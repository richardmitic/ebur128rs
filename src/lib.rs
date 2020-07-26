#[macro_use]
extern crate enum_display_derive;

use biquad::{Biquad, Coefficients, DirectForm1};
use std::error::Error;
use std::fmt::Display;
use std::result::Result;

// Spec defines 400ms block overlapping by 75%
const AUDIO_BLOCK_S: f64 = 0.1;
const MOMENTARY_BLOCK_S: f64 = 0.4;
const SHORT_TERM_BLOCK_S: f64 = 3.;
const GATING_THRESHOLD_ABSOLUTE: f64 = -70.;

#[derive(PartialEq, Eq, Hash, Copy, Clone, Ord, PartialOrd, Debug, Display)]
pub enum Channel {
    Left = 0,
    Right = 1,
    Centre = 2,
    Lfe = 3, // unused
    LeftSurround = 4,
    RightSurround = 5,
}

impl Default for Channel {
    fn default() -> Self {
        Channel::Left
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Display)]
pub enum GatingType {
    None,
    Absolute,
    Relative,
}

impl Default for GatingType {
    fn default() -> Self {
        GatingType::None
    }
}

pub fn channel_from_index(index: usize) -> Channel {
    match index {
        0 => Channel::Left,
        1 => Channel::Right,
        2 => Channel::Centre,
        3 => Channel::Lfe,
        4 => Channel::LeftSurround,
        _ => Channel::RightSurround,
    }
}

#[derive(PartialEq, Eq, Hash, Copy, Clone, Ord, PartialOrd, Debug, Display)]
pub enum Ebur128Error {
    NotEnoughAudio,
    TooQuiet,
    NotAvailable,
    WrongBlockSize,
}

impl Error for Ebur128Error {}

fn root_mean(values: &[f64]) -> f64 {
    values.iter().map(|v| (*v).powi(2)).sum::<f64>() / values.len() as f64
}

fn channel_gain(channel: Channel) -> f64 {
    match channel {
        Channel::Left => 1.,
        Channel::Right => 1.,
        Channel::Centre => 1.,
        Channel::LeftSurround => 1.41,
        Channel::RightSurround => 1.41,
        _ => 0.,
    }
}

fn get_filters() -> (DirectForm1<f64>, DirectForm1<f64>) {
    let stage1 = Coefficients {
        a1: -1.69065929318241,
        a2: 0.73248077421585,
        b0: 1.53512485958697,
        b1: -2.69169618940638,
        b2: 1.19839281085285,
    };

    let stage2 = Coefficients {
        a1: -1.99004745483398,
        a2: 0.99007225036621,
        b0: 1.,
        b1: -2.,
        b2: 1.,
    };

    (
        DirectForm1::<f64>::new(stage1),
        DirectForm1::<f64>::new(stage2),
    )
}

pub fn calculate_channel_loudness(channel: Channel, samples: &[f64]) -> f64 {
    // FIXME: assume the audio block is the correct length
    // FIXME: assume sampling rate is 48kHz
    let mut filtered_audio: Vec<f64> = Vec::with_capacity(samples.len());
    let (mut f1, mut f2) = get_filters();
    for sample in samples {
        filtered_audio.push(f2.run(f1.run(*sample)));
    }
    root_mean(filtered_audio.as_slice()) * channel_gain(channel)
}

pub fn calculate_channel(samples: &[f64], channel: Channel) -> f64 {
    root_mean(samples) * channel_gain(channel)
}

pub fn deinterleave(samples: &[f64], num_channels: usize) -> Vec<Vec<f64>> {
    (0..num_channels)
        .map(|n| {
            samples
                .iter()
                .skip(n)
                .step_by(num_channels)
                .map(|s| *s)
                .collect()
        })
        .collect()
}

pub fn deinterleave_and_filter(interleaved_samples: &[f64], num_channels: usize) -> Vec<Vec<f64>> {
    let mut audio = deinterleave(interleaved_samples, num_channels);
    let mut filters = vec![get_filters(); num_channels];
    for (channel, (mut f1, mut f2)) in audio.iter_mut().zip(filters.iter_mut()) {
        for sample in channel.iter_mut() {
            *sample = f2.run(f1.run(*sample));
        }
    }
    audio
}

pub fn calculate_loudness_multichannel(interleaved_samples: &[f64], num_channels: usize) -> f64 {
    let audio = deinterleave_and_filter(interleaved_samples, num_channels);
    let mut sum = 0f64;
    for (index, channel_audio) in audio.into_iter().enumerate() {
        let channel = channel_from_index(index);
        if channel == Channel::Lfe {
            continue;
        }
        sum += calculate_channel(channel_audio.as_slice(), channel);
    }
    -0.691 + (10. * sum.log10())
}

#[derive(Clone, Debug)]
pub struct State {
    sample_rate: f64,
    channels: usize,
    loudness_blocks: Vec<f64>,
    running_loudness: f64,
    blocks_processed: f64,
    streaming: bool,
}

impl Default for State {
    fn default() -> Self {
        State::new(48000., 2, false)
    }
}

impl State {
    pub fn new(sample_rate: f64, channels: usize, streaming: bool) -> State {
        State {
            sample_rate: sample_rate,
            channels: channels,
            loudness_blocks: vec![],
            running_loudness: 0.,
            blocks_processed: 0.,
            streaming: streaming,
        }
    }

    fn short_term_loudness_blocks() -> usize {
        (SHORT_TERM_BLOCK_S / AUDIO_BLOCK_S) as usize
    }

    fn momentary_loudness_blocks() -> usize {
        (MOMENTARY_BLOCK_S / AUDIO_BLOCK_S) as usize
    }

    fn samples_per_audio_block(&self) -> usize {
        (AUDIO_BLOCK_S * self.sample_rate) as usize * self.channels
    }

    fn gating_threshold(&self, gating: GatingType) -> f64 {
        match gating {
            GatingType::None => f64::NEG_INFINITY,
            GatingType::Absolute => GATING_THRESHOLD_ABSOLUTE,
            GatingType::Relative => self.integrated_loudness(GatingType::Absolute).unwrap() - 10., // Blind unwrap should be fine - function is not public
        }
    }

    fn update_running_loudness(&mut self, val: f64) {
        self.blocks_processed += 1.;
        self.running_loudness += (val - self.running_loudness) / self.blocks_processed;
    }

    fn store_loudness_block(&mut self, loudness: f64) {
        self.loudness_blocks.push(loudness);
        if self.streaming {
            while self.loudness_blocks.len() > Self::short_term_loudness_blocks() {
                self.loudness_blocks.remove(0);
            }
        }
    }

    pub fn process(&mut self, interleaved_samples: &[f64]) -> Result<(), Ebur128Error> {
        if interleaved_samples.len() != self.samples_per_audio_block() {
            return Err(Ebur128Error::WrongBlockSize);
        }
        let loudness = calculate_loudness_multichannel(interleaved_samples, self.channels);
        self.store_loudness_block(loudness);
        if self.streaming {
            self.update_running_loudness(loudness);
        }
        Ok(())
    }

    fn loudness(
        &self,
        gating: GatingType,
        starting_block_index: usize,
    ) -> Result<f64, Ebur128Error> {
        let threshold = self.gating_threshold(gating);

        let blocks_above_threshold = self.loudness_blocks[starting_block_index..]
            .iter()
            .filter(|loudness| **loudness >= threshold)
            .count();

        if blocks_above_threshold == 0 {
            return Err(Ebur128Error::TooQuiet);
        }

        Ok(self.loudness_blocks[starting_block_index..]
            .iter()
            .filter(|loudness| **loudness >= threshold)
            .sum::<f64>()
            / blocks_above_threshold as f64)
    }

    pub fn integrated_loudness(&self, gating: GatingType) -> Result<f64, Ebur128Error> {
        if self.streaming {
            if gating == GatingType::Relative {
                return Err(Ebur128Error::NotAvailable);
            }
            if self.blocks_processed == 0. {
                return Err(Ebur128Error::NotEnoughAudio);
            }
            return Ok(self.running_loudness);
        }
        self.loudness(gating, 0)
    }

    pub fn short_term_loudness(&self, gating: GatingType) -> Result<f64, Ebur128Error> {
        let starting_block_idx = self
            .loudness_blocks
            .len()
            .checked_sub(Self::short_term_loudness_blocks())
            .unwrap_or(0);
        self.loudness(gating, starting_block_idx)
    }

    pub fn momentary_loudness(&self, gating: GatingType) -> Result<f64, Ebur128Error> {
        let starting_block_idx = self
            .loudness_blocks
            .len()
            .checked_sub(Self::momentary_loudness_blocks())
            .unwrap_or(0);
        self.loudness(gating, starting_block_idx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use dasp_signal::{self as signal, Signal};
    use more_asserts::*;
    use rand::{thread_rng, Rng};

    macro_rules! assert_close_enough {
        ($left:expr, $right:expr, $tollerance:expr) => {
            let (left, right, tollerance) = (&($left), &($right), &($tollerance));
            assert_ge!(*left, *right - *tollerance);
            assert_le!(*left, *right + *tollerance);
        };
    }

    fn create_noise(length: usize, scale: f64) -> Vec<f64> {
        let mut rng = thread_rng();
        (0..length).map(|_| rng.gen::<f64>() * scale).collect()
    }

    fn tone_1k() -> Vec<f64> {
        signal::rate(48000.)
            .const_hz(1000.)
            .sine()
            .take(48000)
            .collect::<Vec<f64>>()
    }

    #[test]
    fn new_state() {
        let state = State::default();
        assert!(state.integrated_loudness(GatingType::None).is_err());
    }

    #[test]
    fn input_too_short() {
        let mut state = State::default();
        assert!(state.process(vec![0f64; 9599].as_slice()).is_err());
    }

    #[test]
    fn input_too_long() {
        let mut state = State::default();
        assert!(state.process(vec![0f64; 9601].as_slice()).is_err());
    }

    #[test]
    fn input_correct_length() {
        let mut state = State::default();
        assert!(state.process(vec![0f64; 9600].as_slice()).is_ok());
    }

    #[test]
    fn integrated_loudness_multiple_frames() {
        let mut state = State::default();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness(GatingType::None).unwrap();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness(GatingType::None).unwrap();
        assert_close_enough!(loudness1, loudness2, 0.1);
    }

    #[test]
    fn integrated_loudness_ungated() {
        let mut state = State::default();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness(GatingType::None).unwrap();
        assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness(GatingType::None).unwrap();
        assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        let loudness3 = state.integrated_loudness(GatingType::None).unwrap();
        assert_gt!(loudness1, loudness2);
        assert_lt!(loudness2, loudness3);
    }

    #[test]
    fn integrated_loudness_gated_absolute() {
        let mut state = State::default();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness(GatingType::Absolute).unwrap();
        assert!(state.process(create_noise(9600, 0.0001).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness(GatingType::Absolute).unwrap();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness3 = state.integrated_loudness(GatingType::Absolute).unwrap();
        assert_eq!(loudness1, loudness2);
        assert_close_enough!(loudness1, loudness3, 0.1);
    }

    #[test]
    fn integrated_loudness_gated_relative() {
        let mut state = State::default();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness(GatingType::Relative).unwrap();
        assert!(state.process(create_noise(9600, 0.01).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness(GatingType::Relative).unwrap();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness3 = state.integrated_loudness(GatingType::Relative).unwrap();
        assert_eq!(loudness1, loudness2);
        assert_close_enough!(loudness1, loudness3, 0.1);
    }

    #[test]
    fn short_term_loudness_ungated() {
        let mut state = State::default();
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        }
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        }
        let il = state.integrated_loudness(GatingType::None).unwrap();
        let stl = state.short_term_loudness(GatingType::None).unwrap();
        assert_gt!(stl, il);
    }

    #[test]
    fn momentary_loudness_ungated() {
        let mut state = State::default();
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        }
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        }
        for _ in 0..4 {
            assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        }
        let il = state.integrated_loudness(GatingType::None).unwrap();
        let stl = state.short_term_loudness(GatingType::None).unwrap();
        let ml = state.momentary_loudness(GatingType::None).unwrap();
        assert_lt!(il, stl);
        assert_lt!(stl, ml);
    }

    #[test]
    fn everything_below_threshold() {
        let mut state = State::default();
        for _ in 0..30 {
            assert_eq!(
                state.process(create_noise(9600, 0.0001).as_slice()).is_ok(),
                true
            );
        }
        assert_eq!(
            state.integrated_loudness(GatingType::Absolute).is_err(),
            true
        );
    }

    #[test]
    fn streaming_mode_integrated_loudness() {
        let mut state = State::new(48000., 1, true);
        for _ in 0..30 {
            assert_eq!(
                state.process(create_noise(4800, 0.5).as_slice()).is_ok(),
                true
            );
        }
        let il = state.integrated_loudness(GatingType::Absolute).unwrap();
        assert_close_enough!(il, -13.6, 0.1);
    }

    #[test]
    fn streaming_mode_no_relative_threshold() {
        let mut state = State::new(48000., 1, true);
        for _ in 0..30 {
            assert_eq!(
                state.process(create_noise(4800, 0.5).as_slice()).is_ok(),
                true
            );
        }
        assert_eq!(
            state.integrated_loudness(GatingType::Relative).is_err(),
            true
        );
    }

    #[test]
    fn streaming_mode_short_term_loudness_ungated() {
        let mut state = State::new(48000., 2, true);
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        }
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        }
        let il = state.integrated_loudness(GatingType::None).unwrap();
        let stl = state.short_term_loudness(GatingType::None).unwrap();
        assert_gt!(stl, il);
    }

    #[test]
    fn streaming_mode_momentary_loudness_ungated() {
        let mut state = State::new(48000., 2, true);
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        }
        for _ in 0..30 {
            assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        }
        for _ in 0..4 {
            assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        }
        let il = state.integrated_loudness(GatingType::None).unwrap();
        let stl = state.short_term_loudness(GatingType::None).unwrap();
        let ml = state.momentary_loudness(GatingType::None).unwrap();
        assert_lt!(il, stl);
        assert_lt!(stl, ml);
    }

    #[test]
    fn sine() {
        // Specification states:
        // "If a 0 dB FS 1 kHz sine wave is applied to the left, centre,
        // or right channel input, the indicated loudness will equal -3.01 LKFS."
        let mut state = State::new(48000., 1, false);
        assert!(state.process(&tone_1k()[0..4800]).is_ok());
        let loudness = state.integrated_loudness(GatingType::Absolute).unwrap();
        assert_close_enough!(loudness, -3.01, 0.01);
    }

    #[test]
    fn channel_loudness() {
        assert_gt!(calculate_channel_loudness(Channel::Left, &[1., 1., 1.]), 2.);
        assert_gt!(
            calculate_channel_loudness(Channel::Right, &[1., 1., 1.]),
            2.
        );
        assert_gt!(
            calculate_channel_loudness(Channel::Centre, &[1., 1., 1.]),
            2.
        );
        assert_gt!(
            calculate_channel_loudness(Channel::LeftSurround, &[1., 1., 1.]),
            2.5
        );
        assert_gt!(
            calculate_channel_loudness(Channel::RightSurround, &[1., 1., 1.]),
            2.5
        );
    }

    #[test]
    fn deinterleave_works() {
        assert_eq!(
            deinterleave(&[0., 1., 0., 1.], 2),
            vec![[0., 0.,], [1., 1.,]]
        );
    }

    #[test]
    fn deinterleave_and_filter_works() {
        let (mut f1, mut f2) = get_filters();
        let left_result: Vec<f64> = vec![0., 1., 1., 1., 1., 1.]
            .into_iter()
            .map(|s| f2.run(f1.run(s)))
            .collect();

        let samples = &[0., 0., 1., 2., 1., 2., 1., 2., 1., 2., 1., 2.];
        let result = deinterleave_and_filter(samples, 2);
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], left_result);
    }

    #[test]
    fn filters_work() {
        let mut samples = vec![0f64];
        samples.append(&mut vec![1f64; 19]);
        let (mut f1, mut f2) = get_filters();
        let result: Vec<f64> = samples.into_iter().map(|s| f2.run(f1.run(s))).collect();
        assert_eq!(result[0], 0.);
        assert_gt!(result[1], result[0]);
        for i in 2..20 {
            assert_lt!(result[i], result[i - 1]);
        }
    }

    #[test]
    fn loudness_multichannel_mono() {
        let result = calculate_loudness_multichannel(create_noise(48000, 0.5).as_slice(), 1);
        assert_ne!(result, 0.);
    }

    #[test]
    fn loudness_multichannel_stereo() {
        let result = calculate_loudness_multichannel(create_noise(48000 * 2, 0.5).as_slice(), 2);
        assert_ne!(result, 0.);
    }

    #[test]
    fn loudness_multichannel_5_1() {
        let result = calculate_loudness_multichannel(create_noise(48000 * 6, 0.5).as_slice(), 6);
        assert_ne!(result, 0.);
    }
}
