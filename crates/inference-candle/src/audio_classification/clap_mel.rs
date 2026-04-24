//! Mel spectrogram computation for CLAP.
//!
//! CLAP (laion/clap-htsat-fused) uses:
//! - Sample rate: 48000 Hz
//! - n_fft: 1024 (window size)
//! - hop_length: 480
//! - num_mel_bins: 64
//! - freq range: 50-14000 Hz
//! - max_length: 10 seconds (480000 samples)
//! - spec_size: 256 (target time frames for the model)

use candle_core::{Device, Tensor};

use crate::{TaskError, TaskResult};

/// CLAP preprocessing parameters from preprocessor_config.json.
pub const CLAP_SAMPLE_RATE: u32 = 48000;
pub const CLAP_N_FFT: usize = 1024;
pub const CLAP_HOP_LENGTH: usize = 480;
pub const CLAP_NUM_MEL_BINS: usize = 64;
pub const CLAP_FREQ_MIN: f64 = 50.0;
pub const CLAP_FREQ_MAX: f64 = 14000.0;
pub const CLAP_MAX_LENGTH_S: f64 = 10.0;

/// Generate mel filterbank matrix.
///
/// Returns a flat Vec<f32> of shape [num_mel_bins, n_freq] where n_freq = n_fft/2 + 1.
/// Uses HTK mel scale: mel = 2595 * log10(1 + f/700).
pub fn generate_mel_filters(
    num_mel_bins: usize,
    n_fft: usize,
    sample_rate: u32,
    freq_min: f64,
    freq_max: f64,
) -> Vec<f32> {
    let n_freq = n_fft / 2 + 1;
    let sr = f64::from(sample_rate);

    let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f64| 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(freq_min);
    let mel_max = hz_to_mel(freq_max);

    // num_mel_bins + 2 boundary points
    let n_points = num_mel_bins + 2;
    let mel_points: Vec<f64> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f64 / (n_points - 1) as f64)
        .collect();

    let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();
    let bin_points: Vec<f64> = hz_points.iter().map(|&f| f * n_fft as f64 / sr).collect();

    let mut filters = vec![0.0f32; num_mel_bins * n_freq];

    for m in 0..num_mel_bins {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freq {
            let kf = k as f64;
            let val = if kf >= left && kf < center {
                (kf - left) / (center - left)
            } else if kf >= center && kf <= right {
                (right - kf) / (right - center)
            } else {
                0.0
            };
            filters[m * n_freq + k] = val as f32;
        }

        // Slaney normalization (area = 1 per triangle)
        let width = hz_points[m + 2] - hz_points[m];
        if width > 0.0 {
            let norm = 2.0 / width as f32;
            for k in 0..n_freq {
                filters[m * n_freq + k] *= norm;
            }
        }
    }

    filters
}

/// Compute STFT magnitude squared (power spectrogram) for a single channel.
///
/// Uses a Hann window. Returns shape [n_freq, num_frames].
fn stft_power(samples: &[f32], n_fft: usize, hop_length: usize) -> Vec<Vec<f32>> {
    let n_freq = n_fft / 2 + 1;

    // Hann window
    let window: Vec<f32> = (0..n_fft)
        .map(|i| {
            let w = (std::f32::consts::PI * i as f32 / n_fft as f32).sin();
            w * w
        })
        .collect();

    let num_frames = if samples.len() >= n_fft {
        (samples.len() - n_fft) / hop_length + 1
    } else {
        1
    };

    let mut spectrogram = Vec::with_capacity(num_frames);

    for frame_idx in 0..num_frames {
        let start = frame_idx * hop_length;
        let mut frame = vec![0.0f32; n_fft];

        // Windowed frame
        for i in 0..n_fft {
            let sample_idx = start + i;
            if sample_idx < samples.len() {
                frame[i] = samples[sample_idx] * window[i];
            }
        }

        // Real FFT via DFT (n_fft is typically 1024, so O(n_fft * n_freq) is fine)
        let mut magnitudes = vec![0.0f32; n_freq];
        for (k, mag) in magnitudes.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (n, &sample) in frame.iter().enumerate().take(n_fft) {
                let angle = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                re += f64::from(sample) * angle.cos();
                im += f64::from(sample) * angle.sin();
            }
            *mag = (re * re + im * im) as f32;
        }
        spectrogram.push(magnitudes);
    }

    spectrogram
}

/// Convert raw waveform to log-mel spectrogram tensor for CLAP.
///
/// Input: mono f32 samples at 48kHz.
/// Output: Tensor of shape `[1, 1, time_frames, num_mel_bins]`.
///
/// The time dimension is the natural STFT frame count (NOT truncated to spec_size).
/// The audio encoder's `reshape_mel2img` will handle interpolation to the
/// target image size `(spec_size, spec_size)`.
///
/// For a 10s clip at 48kHz with hop=480: ~998 frames.
/// The target for reshape is `spec_size * freq_ratio = 256 * 4 = 1024` time frames.
pub fn waveform_to_mel(
    samples: &[f32],
    num_mel_bins: usize,
    mel_filters: &[f32],
    device: &Device,
) -> TaskResult<Tensor> {
    let n_freq = CLAP_N_FFT / 2 + 1;

    // Pad samples to at least n_fft length
    let padded = if samples.len() < CLAP_N_FFT {
        let mut p = samples.to_vec();
        p.resize(CLAP_N_FFT, 0.0);
        p
    } else {
        samples.to_vec()
    };

    // Compute power spectrogram: list of [n_freq] frames
    let power_spec = stft_power(&padded, CLAP_N_FFT, CLAP_HOP_LENGTH);
    let num_frames = power_spec.len();

    // Apply mel filterbank: [num_mel_bins, n_freq] x [n_freq, num_frames] -> [num_mel_bins, num_frames]
    let mut mel_spec = vec![0.0f32; num_mel_bins * num_frames];
    for m in 0..num_mel_bins {
        for t in 0..num_frames {
            let mut sum = 0.0f32;
            for k in 0..n_freq {
                sum += mel_filters[m * n_freq + k] * power_spec[t][k];
            }
            mel_spec[m * num_frames + t] = sum;
        }
    }

    // Log-mel (power to dB): 10 * log10(max(mel, 1e-10))
    for val in &mut mel_spec {
        *val = 10.0 * (*val).max(1e-10).log10();
    }

    // Normalize: shift so max = 0, then scale
    let max_val = mel_spec.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    for val in &mut mel_spec {
        *val = (*val - max_val).max(-80.0); // clip to -80 dB
        *val /= 80.0; // scale to [-1, 0]
        *val += 1.0; // shift to [0, 1]
    }

    // Create tensor [1, 1, time_frames, num_mel_bins]
    // mel_spec is in [mel_bins, time] layout, so we need to transpose:
    // First create [mel_bins, time], then reshape to [1, 1, time, mel_bins]
    let mel_tensor = Tensor::from_vec(mel_spec, (1, num_mel_bins, num_frames), device)
        .map_err(|e| TaskError::Inference(format!("mel tensor creation: {e}")))?;
    // [1, mel_bins, time] → [1, time, mel_bins] → [1, 1, time, mel_bins]
    mel_tensor
        .permute((0, 2, 1))
        .and_then(|t| t.unsqueeze(1))
        .map_err(|e| TaskError::Inference(format!("mel reshape: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mel_filters_shape() {
        let filters = generate_mel_filters(64, 1024, 48000, 50.0, 14000.0);
        let n_freq = 1024 / 2 + 1;
        assert_eq!(filters.len(), 64 * n_freq);
    }

    #[test]
    fn test_mel_filters_non_negative() {
        let filters = generate_mel_filters(64, 1024, 48000, 50.0, 14000.0);
        assert!(filters.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_stft_power_basic() {
        // 1 second of 48kHz silence
        let samples = vec![0.0f32; 48000];
        let spec = stft_power(&samples, 1024, 480);
        let n_freq = 1024 / 2 + 1;
        // Each frame should be all zeros (silence)
        for frame in &spec {
            assert_eq!(frame.len(), n_freq);
            assert!(frame.iter().all(|&v| v.abs() < 1e-10));
        }
    }

    #[test]
    fn test_waveform_to_mel_shape() {
        let filters = generate_mel_filters(64, 1024, 48000, 50.0, 14000.0);
        // 1 second of 48kHz audio → (48000 - 1024) / 480 + 1 = 99 frames
        let samples = vec![0.0f32; 48000];
        let device = Device::Cpu;
        let mel = waveform_to_mel(&samples, 64, &filters, &device).unwrap();
        // Shape: [1, 1, time_frames, 64]
        let dims = mel.dims();
        assert_eq!(dims[0], 1);
        assert_eq!(dims[1], 1);
        assert_eq!(dims[3], 64);
        // time_frames = (48000 - 1024) / 480 + 1 = 99 (approximately)
        assert!(
            dims[2] > 90 && dims[2] < 110,
            "Expected ~99 frames, got {}",
            dims[2]
        );
    }

    #[test]
    fn test_waveform_to_mel_short_audio() {
        let filters = generate_mel_filters(64, 1024, 48000, 50.0, 14000.0);
        // Very short audio (less than one window)
        let samples = vec![0.1f32; 100];
        let device = Device::Cpu;
        let mel = waveform_to_mel(&samples, 64, &filters, &device).unwrap();
        // Shape: [1, 1, 1, 64] (padded to n_fft, gives 1 frame)
        assert_eq!(mel.dims()[0], 1);
        assert_eq!(mel.dims()[1], 1);
        assert_eq!(mel.dims()[3], 64);
    }
}
