// extern crate itertools;

use std::error::Error;
use std::fmt;

type SAMPLE = f64;

// Spec defines 400ms block overlapping by 75%
const AUDIO_BLOCK_S: f64 = 0.1;
const MOMENTARY_BLOCK_S: f64 = 0.4;
const SHORT_TERM_BLOCK_S: f64 = 3.;

#[derive(Debug)]
struct Ebur128Error {}

impl fmt::Display for Ebur128Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Ebur128Error is here!")
    }
}

impl Error for Ebur128Error {}

fn add_vec(a: Vec<f64>, b: &Vec<f64>) -> Vec<f64> {
    a.iter().zip(b).map(|(x,y)| x + y).collect::<Vec<f64>>()
}

#[derive(Debug)]
struct State {
    sample_rate: f64,
    channels: usize,
    // audio_data: Vec<Vec<SAMPLE>> // non-interleaved, filtered audio
    loudness_blocks: Vec<Vec<f64>> // time x channel
}

impl State {
    pub fn new(sample_rate: f64, channels: usize) -> State {
        State {
            sample_rate: sample_rate,
            channels: channels,
            // audio_data: vec![vec![0.; State::audio_buffer_length(sample_rate)]; channels]
            loudness_blocks: vec![vec![]; channels]
        }
    }

    fn audio_buffer_length(sample_rate: f64) -> usize {
        (sample_rate * SHORT_TERM_BLOCK_S) as usize
    }

    fn num_loudness_blocks() -> usize {
        (SHORT_TERM_BLOCK_S / AUDIO_BLOCK_S) as usize
    }

    fn audio_block_samples(self) -> usize {
        (AUDIO_BLOCK_S * self.sample_rate) as usize
    }

    fn root_mean(values: &[f64]) -> f64 {
        values.iter().map(|v| (*v).powi(2)).sum::<f64>() / values.len() as f64
    }

    pub fn add_frames(&mut self, interleaved_frames: &[f64]) -> Result<(), Ebur128Error> {
        let deinterleaved_channels: Vec<Vec<f64>> = (0..self.channels)
            .map(|n| interleaved_frames.iter().skip(n).step_by(self.channels).map(|s| *s).collect())
            .collect();

        for (lb, ch) in self.loudness_blocks.iter_mut().zip(deinterleaved_channels) {
            lb.push(Self::root_mean(ch.as_slice()));
        }

        Ok(())
    }

    pub fn integrated_loudness(&self) -> f64 {
        self.loudness_blocks
            .iter()
            .fold(
                vec![0.; self.channels],
                |sum, val| { add_vec(sum, val) }
            )
            .iter()
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_state() {
        let state = State::new(48000., 2);
        assert_eq!(state.integrated_loudness(), 0.);
    }

    #[test]
    fn frames_are_added() {
        let mut state = State::new(48000., 2);
        assert!(state.add_frames(&[1., 2., 1., 2., 1., 2.]).is_ok());
        assert_eq!(state.integrated_loudness(), 1. + 4.);
    }
}
