//! Audio preprocessing utilities.
//!
//! Requires the `audio` feature flag.

use hound::{WavReader, WavSpec};
use std::io::Cursor;

use crate::error::{TaskError, TaskResult};

/// Audio sample rate commonly used for ASR models.
pub const SAMPLE_RATE_16K: u32 = 16000;

/// Decode WAV audio from bytes.
pub fn decode_wav(bytes: &[u8]) -> TaskResult<(Vec<f32>, WavSpec)> {
    let cursor = Cursor::new(bytes);
    let reader = WavReader::new(cursor)
        .map_err(|e| TaskError::InvalidInput(format!("Invalid WAV: {}", e)))?;

    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .map(|s| s.unwrap_or(0.0))
            .collect(),
        hound::SampleFormat::Int => {
            let max_value = (1 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.unwrap_or(0) as f32 / max_value)
                .collect()
        }
    };

    Ok((samples, spec))
}

/// Convert stereo to mono by averaging channels.
pub fn stereo_to_mono(samples: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 {
        return samples;
    }

    samples
        .chunks(channels as usize)
        .map(|chunk| chunk.iter().sum::<f32>() / channels as f32)
        .collect()
}

/// Simple linear interpolation resampling.
pub fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
    if from_rate == to_rate {
        return samples.to_vec();
    }

    let ratio = from_rate as f64 / to_rate as f64;
    let new_len = (samples.len() as f64 / ratio) as usize;

    (0..new_len)
        .map(|i| {
            let src_idx = i as f64 * ratio;
            let idx = src_idx as usize;
            let frac = src_idx - idx as f64;

            if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac as f32) + samples[idx + 1] * frac as f32
            } else {
                samples[idx.min(samples.len() - 1)]
            }
        })
        .collect()
}

/// Normalize audio to [-1, 1] range.
pub fn normalize(samples: &mut [f32]) {
    let max = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    if max > 0.0 {
        for s in samples.iter_mut() {
            *s /= max;
        }
    }
}
