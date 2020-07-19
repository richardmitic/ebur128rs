use std::error::Error;
use std::fmt;

use biquad::{Coefficients, DirectForm1, Biquad};

// Spec defines 400ms block overlapping by 75%
const AUDIO_BLOCK_S: f64 = 0.1;
const MOMENTARY_BLOCK_S: f64 = 0.4;
const SHORT_TERM_BLOCK_S: f64 = 3.;

#[derive(PartialEq, Eq, Hash, Copy, Clone)]
pub enum Channel {
    Left = 0,
    Right = 1,
    Centre = 2,
    Lfe = 3, // unused
    LeftSurround = 4,
    RightSurround = 5
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


#[derive(Debug)]
struct Ebur128Error {}

impl fmt::Display for Ebur128Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ebur128Error is here!")
    }
}

impl Error for Ebur128Error {}

fn add_vec(a: Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b).map(|(x, y)| x + y).collect::<Vec<f64>>()
}

fn scale_vec(a: Vec<f64>, scale: f64) -> Vec<f64> {
    a.iter().map(|x| x * scale).collect::<Vec<f64>>()
}

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
        _ => 0.
    }
}

fn get_filters() -> (DirectForm1<f64>, DirectForm1<f64>) {
    let stage1 = Coefficients {
        a1: -1.69065929318241,
        a2: 0.73248077421585,
        b0: 1.53512485958697,
        b1: -2.69169618940638,
        b2: 1.19839281085285
    };

    let stage2 = Coefficients {
        a1: -1.99004745483398,
        a2: 0.99007225036621,
        b0: 1.,
        b1: -2.,
        b2: 1.
    };

    (DirectForm1::<f64>::new(stage1), DirectForm1::<f64>::new(stage2))
}

pub fn caluclate_channel_loudness(channel: Channel, samples: &[f64]) -> f64 {
    // FIXME: assume the audio block is the correct length
    // FIXME: assume sampling rate is 48kHz
    let mut filtered_audio: Vec<f64> = Vec::with_capacity(samples.len());
    let (mut f1, mut f2) = get_filters();
    for sample in samples {
        filtered_audio.push(f2.run(f1.run(*sample)));
    }
    root_mean(filtered_audio.as_slice()) * channel_gain(channel)
}

pub fn caluclate_channel(samples: &[f64], channel: Channel) -> f64 {
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
            continue
        }
        sum += caluclate_channel(channel_audio.as_slice(), channel);
    }
    -0.691 + (10. * sum.log10())
}

#[derive(Debug)]
struct State {
    sample_rate: f64,
    channels: usize,
    loudness_blocks: Vec<f64>,
    integrated_loudness: f64,
    blocks_processed: usize
}

impl State {
    pub fn new(sample_rate: f64, channels: usize) -> State {
        State {
            sample_rate: sample_rate,
            channels: channels,
            loudness_blocks: vec![],
            integrated_loudness: 0.,
            blocks_processed: 0
        }
    }

    fn num_loudness_blocks() -> usize {
        (SHORT_TERM_BLOCK_S / AUDIO_BLOCK_S) as usize
    }

    fn samples_per_audio_block(&self) -> usize {
        (AUDIO_BLOCK_S * self.sample_rate) as usize * self.channels
    }

    pub fn process(&mut self, interleaved_samples: &[f64]) -> Result<(), Ebur128Error> {
        if interleaved_samples.len() != self.samples_per_audio_block() {
            return Err(Ebur128Error{});
        }
        let loudness = calculate_loudness_multichannel(interleaved_samples, self.channels);
        self.loudness_blocks.push(loudness);
        Ok(())
    }

    pub fn integrated_loudness(&self) -> Result<f64, Ebur128Error> {
        match self.loudness_blocks.is_empty() {
            true => Err(Ebur128Error{}),
            false => Ok(self.loudness_blocks.iter().sum::<f64>() / self.loudness_blocks.len() as f64)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, thread_rng};
    use more_asserts::*;

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

    #[test]
    fn new_state() {
        let state = State::new(48000., 2);
        assert!(state.integrated_loudness().is_err());
    }

    #[test]
    fn input_too_short() {
        let mut state = State::new(48000., 2);
        assert!(state.process(vec![0f64; 9599].as_slice()).is_err());
    }

    #[test]
    fn input_too_long() {
        let mut state = State::new(48000., 2);
        assert!(state.process(vec![0f64; 9601].as_slice()).is_err());
    }

    #[test]
    fn input_correct_length() {
        let mut state = State::new(48000., 2);
        assert!(state.process(vec![0f64; 9600].as_slice()).is_ok());
    }

    #[test]
    fn integrated_loudness_multiple_frames() {
        let mut state = State::new(48000., 2);
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness().unwrap();
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness().unwrap();
        assert_close_enough!(loudness1, loudness2, 0.1);
    }

    #[test]
    fn integrated_loudness_updates() {
        let mut state = State::new(48000., 2);
        assert!(state.process(create_noise(9600, 0.5).as_slice()).is_ok());
        let loudness1 = state.integrated_loudness().unwrap();
        assert!(state.process(create_noise(9600, 0.1).as_slice()).is_ok());
        let loudness2 = state.integrated_loudness().unwrap();
        assert!(state.process(create_noise(9600, 0.9).as_slice()).is_ok());
        let loudness3 = state.integrated_loudness().unwrap();
        assert_gt!(loudness1, loudness2);
        assert_lt!(loudness2, loudness3);
    }

    #[test]
    fn channel_loudness() {
        assert!(caluclate_channel_loudness(Channel::Left, &[1., 1., 1.]) > 2.);
        assert!(caluclate_channel_loudness(Channel::Right, &[1., 1., 1.]) > 2.);
        assert!(caluclate_channel_loudness(Channel::Centre, &[1., 1., 1.]) > 2.);
        assert!(caluclate_channel_loudness(Channel::LeftSurround, &[1., 1., 1.]) > 2.5);
        assert!(caluclate_channel_loudness(Channel::RightSurround, &[1., 1., 1.]) > 2.5);
    }

    #[test]
    fn deinterleave_works() {
        assert_eq!(deinterleave(&[0., 1., 0., 1.], 2), vec![[0., 0.,], [1., 1.,]]);
    }

    #[test]
    fn deinterleave_and_filter_works() {
        let (mut f1, mut f2) = get_filters();
        let left_result: Vec<f64> = vec![0., 1., 1., 1., 1., 1.].into_iter().map(|s| f2.run(f1.run(s))).collect();

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
