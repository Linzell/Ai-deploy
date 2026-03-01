//! Audio preprocessing for ASR and audio classification models.
//!
//! Handles:
//! - Loading WAV files
//! - Resampling to 16kHz
//! - Computing mel spectrograms for Whisper (ASR)
//! - Raw waveform output for wav2vec2 (audio classification)
//! - Converting to tensor format

use crate::error::{PreprocessError, PreprocessResult};
use ndarray::{Array2, Array3};
use rustfft::{num_complex::Complex, FftPlanner};
use std::cmp::Ordering;
use std::f32::consts::PI;
use std::path::Path;
use tracing::{debug, info};

/// DSP-safe numeric conversions.
///
/// These functions handle the numeric conversions needed for audio DSP code.
/// In DSP, we frequently convert between:
/// - Sample counts (usize) and floating-point for ratio calculations
/// - Sample rates (u32) and f32/f64 for frequency calculations
/// - FFT bin indices and floating-point for interpolation
///
/// These conversions are safe for typical audio scenarios:
/// - Audio files under ~16 million samples (277 hours at 16kHz) fit in f32's 24-bit mantissa
/// - Sample rates are always positive and < 200kHz
/// - FFT sizes are typically < 8192
mod dsp_cast {
    /// Convert usize to f32 for DSP calculations.
    /// Safe for typical audio buffer sizes (< 16M samples).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn usize_to_f32(v: usize) -> f32 {
        v as f32
    }

    /// Convert usize to f64 for high-precision DSP calculations.
    /// Safe for all practical audio scenarios.
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn usize_to_f64(v: usize) -> f64 {
        v as f64
    }

    /// Convert u32 to f32 (sample rates, bit depths).
    /// Safe for all audio sample rates (< 16M).
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn u32_to_f32(v: u32) -> f32 {
        v as f32
    }

    /// Convert i32 to f32 (audio samples).
    /// Safe for 24-bit audio and below.
    #[inline]
    #[allow(clippy::cast_precision_loss)]
    pub fn i32_to_f32(v: i32) -> f32 {
        v as f32
    }

    /// Convert f64 to usize for index calculations.
    /// The value must be non-negative (enforced by DSP algorithms).
    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn f64_to_usize(v: f64) -> usize {
        v as usize
    }

    /// Convert f32 to usize for bin index calculations.
    /// The value must be non-negative (enforced by DSP algorithms).
    #[inline]
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub fn f32_to_usize(v: f32) -> usize {
        v as usize
    }

    /// Convert f64 fractional part to f32 for interpolation.
    #[inline]
    #[allow(clippy::cast_possible_truncation)]
    pub fn f64_to_f32(v: f64) -> f32 {
        v as f32
    }
}

/// Whisper audio configuration.
pub const WHISPER_SAMPLE_RATE: u32 = 16000;
pub const WHISPER_N_FFT: usize = 400;
pub const WHISPER_HOP_LENGTH: usize = 160;
pub const WHISPER_N_MELS: usize = 80;
pub const WHISPER_CHUNK_LENGTH: usize = 30; // seconds
/// Whisper expects exactly 3000 time frames for 30 seconds of audio.
/// This is calculated as: (30 * 16000 / 160) = 3000 frames.
/// The mel spectrogram will be padded or truncated to this size.
pub const WHISPER_MAX_FRAMES: usize = 3000;

/// wav2vec2 audio configuration.
pub const WAV2VEC2_SAMPLE_RATE: u32 = 16000;
pub const WAV2VEC2_MAX_LENGTH: usize = 30; // seconds

/// Audio output type - determines what format the model expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AudioOutputType {
    /// Mel spectrogram features for Whisper-style models.
    /// Output key: `input_features` with shape [batch, n_mels, time_frames]
    MelSpectrogram,
    /// Raw waveform samples for wav2vec2-style models.
    /// Output key: `input_values` with shape [batch, samples]
    RawWaveform,
}

/// Processed audio ready for ONNX inference.
#[derive(Debug, Clone)]
pub enum AudioOutput {
    /// Mel spectrogram for Whisper-style models.
    MelSpectrogram {
        /// Mel spectrogram features [batch_size, n_mels, time_frames]
        /// For Whisper: [1, 80, 3000] for 30 seconds of audio
        input_features: Array3<f32>,
        /// Original audio duration in seconds
        duration_seconds: f32,
    },
    /// Raw waveform for wav2vec2-style models.
    RawWaveform {
        /// Raw audio samples [batch_size, samples]
        /// Normalized to [-1, 1] range
        input_values: ndarray::Array2<f32>,
        /// Original audio duration in seconds
        duration_seconds: f32,
    },
}

impl AudioOutput {
    /// Get the duration in seconds.
    pub fn duration_seconds(&self) -> f32 {
        match self {
            Self::MelSpectrogram {
                duration_seconds, ..
            }
            | Self::RawWaveform {
                duration_seconds, ..
            } => *duration_seconds,
        }
    }

    /// Convert to JSON-compatible format for ONNX input.
    pub fn to_json_inputs(&self) -> serde_json::Value {
        match self {
            Self::MelSpectrogram { input_features, .. } => {
                let shape = input_features.shape();
                let mut result = Vec::with_capacity(shape[0]);

                for b in 0..shape[0] {
                    let mut mels = Vec::with_capacity(shape[1]);
                    for m in 0..shape[1] {
                        let frames: Vec<f32> =
                            (0..shape[2]).map(|t| input_features[[b, m, t]]).collect();
                        mels.push(frames);
                    }
                    result.push(mels);
                }

                serde_json::json!({
                    "input_features": result
                })
            }
            Self::RawWaveform { input_values, .. } => {
                let shape = input_values.shape();
                let mut result = Vec::with_capacity(shape[0]);

                for b in 0..shape[0] {
                    let samples: Vec<f32> = (0..shape[1]).map(|s| input_values[[b, s]]).collect();
                    result.push(samples);
                }

                serde_json::json!({
                    "input_values": result
                })
            }
        }
    }
}

/// Audio processing configuration.
#[derive(Debug, Clone)]
pub struct AudioConfig {
    /// Target sample rate
    pub sample_rate: u32,
    /// FFT size (only used for MelSpectrogram output)
    pub n_fft: usize,
    /// Hop length between frames (only used for MelSpectrogram output)
    pub hop_length: usize,
    /// Number of mel filterbanks (only used for MelSpectrogram output)
    pub n_mels: usize,
    /// Maximum audio length in seconds (None = no limit)
    pub max_length_seconds: Option<usize>,
    /// Whether to pad to max_length
    pub pad_to_max: bool,
    /// Output type - determines what format the model expects
    pub output_type: AudioOutputType,
    /// Target number of time frames for mel spectrogram (None = no padding).
    /// Whisper expects exactly 3000 frames for 30s audio.
    /// If set, the mel spectrogram will be padded or truncated to this exact size.
    pub target_time_frames: Option<usize>,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self::whisper()
    }
}

impl AudioConfig {
    /// Create Whisper-compatible config (mel spectrogram output).
    ///
    /// Whisper expects exactly 3000 time frames for 30 seconds of audio.
    /// The mel spectrogram will be padded or truncated to this exact size.
    pub fn whisper() -> Self {
        Self {
            sample_rate: WHISPER_SAMPLE_RATE,
            n_fft: WHISPER_N_FFT,
            hop_length: WHISPER_HOP_LENGTH,
            n_mels: WHISPER_N_MELS,
            max_length_seconds: Some(WHISPER_CHUNK_LENGTH),
            pad_to_max: true,
            output_type: AudioOutputType::MelSpectrogram,
            target_time_frames: Some(WHISPER_MAX_FRAMES),
        }
    }

    /// Create wav2vec2-compatible config (raw waveform output).
    /// Used for audio classification models.
    pub fn wav2vec2() -> Self {
        Self {
            sample_rate: WAV2VEC2_SAMPLE_RATE,
            n_fft: 0,      // Not used for raw waveform
            hop_length: 0, // Not used for raw waveform
            n_mels: 0,     // Not used for raw waveform
            max_length_seconds: Some(WAV2VEC2_MAX_LENGTH),
            pad_to_max: true,
            output_type: AudioOutputType::RawWaveform,
            target_time_frames: None, // Raw waveform doesn't use frames
        }
    }
}

/// Audio processor for ASR models.
pub struct AudioProcessor {
    config: AudioConfig,
    mel_filters: Array2<f32>,
}

impl AudioProcessor {
    /// Create a new audio processor with default (Whisper) config.
    pub fn new() -> Self {
        Self::with_config(AudioConfig::default())
    }

    /// Create with custom config.
    pub fn with_config(config: AudioConfig) -> Self {
        let mel_filters = create_mel_filterbank(config.sample_rate, config.n_fft, config.n_mels);

        Self {
            config,
            mel_filters,
        }
    }

    /// Load and process audio from a WAV file.
    pub fn process_file(&self, path: impl AsRef<Path>) -> PreprocessResult<AudioOutput> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading audio from file");

        let mut reader = hound::WavReader::open(path)
            .map_err(|e| PreprocessError::Audio(format!("Failed to open WAV file: {e}")))?;

        let spec = reader.spec();
        debug!(
            sample_rate = spec.sample_rate,
            channels = spec.channels,
            bits = spec.bits_per_sample,
            "Audio file specs"
        );

        // Read samples and convert to f32 mono
        let samples = Self::read_samples(&mut reader)?;

        // Resample if necessary
        let samples = if spec.sample_rate == self.config.sample_rate {
            samples
        } else {
            Self::resample(&samples, spec.sample_rate, self.config.sample_rate)
        };

        self.process_samples(&samples)
    }

    /// Process raw audio bytes (WAV format).
    pub fn process_bytes(&self, bytes: &[u8]) -> PreprocessResult<AudioOutput> {
        debug!(size = bytes.len(), "Processing audio from bytes");

        let cursor = std::io::Cursor::new(bytes);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| PreprocessError::Audio(format!("Failed to parse WAV: {e}")))?;

        let spec = reader.spec();
        let samples = Self::read_samples(&mut reader)?;

        let samples = if spec.sample_rate == self.config.sample_rate {
            samples
        } else {
            Self::resample(&samples, spec.sample_rate, self.config.sample_rate)
        };

        self.process_samples(&samples)
    }

    /// Process pre-loaded samples.
    pub fn process_samples(&self, samples: &[f32]) -> PreprocessResult<AudioOutput> {
        let duration_seconds =
            dsp_cast::usize_to_f32(samples.len()) / dsp_cast::u32_to_f32(self.config.sample_rate);
        debug!(
            num_samples = samples.len(),
            duration_seconds = duration_seconds,
            output_type = ?self.config.output_type,
            "Processing audio samples"
        );

        // Truncate or pad to max length
        let samples = if let Some(max_secs) = self.config.max_length_seconds {
            let max_samples = max_secs * self.config.sample_rate as usize;
            if samples.len() > max_samples {
                samples[..max_samples].to_vec()
            } else if self.config.pad_to_max && samples.len() < max_samples {
                let mut padded = samples.to_vec();
                padded.resize(max_samples, 0.0);
                padded
            } else {
                samples.to_vec()
            }
        } else {
            samples.to_vec()
        };

        match self.config.output_type {
            AudioOutputType::MelSpectrogram => {
                // Compute mel spectrogram
                let mel_spec = self.compute_mel_spectrogram(&samples)?;

                // Pad or truncate to target frame count if specified
                let mel_spec = if let Some(target_frames) = self.config.target_time_frames {
                    let (n_mels, n_frames) = (mel_spec.nrows(), mel_spec.ncols());

                    match n_frames.cmp(&target_frames) {
                        Ordering::Equal => mel_spec,
                        Ordering::Less => {
                            // Pad with zeros (log mel of silence)
                            let mut padded = Array2::<f32>::zeros((n_mels, target_frames));
                            padded
                                .slice_mut(ndarray::s![.., ..n_frames])
                                .assign(&mel_spec);
                            debug!(
                                original_frames = n_frames,
                                target_frames = target_frames,
                                "Padded mel spectrogram"
                            );
                            padded
                        }
                        Ordering::Greater => {
                            // Truncate
                            debug!(
                                original_frames = n_frames,
                                target_frames = target_frames,
                                "Truncated mel spectrogram"
                            );
                            mel_spec.slice(ndarray::s![.., ..target_frames]).to_owned()
                        }
                    }
                } else {
                    mel_spec
                };

                // Add batch dimension [1, n_mels, time]
                let (n_mels, n_frames) = (mel_spec.nrows(), mel_spec.ncols());
                let input_features = mel_spec
                    .into_shape_with_order((1, n_mels, n_frames))
                    .map_err(|e| PreprocessError::Audio(format!("Shape error: {e}")))?;

                Ok(AudioOutput::MelSpectrogram {
                    input_features,
                    duration_seconds,
                })
            }
            AudioOutputType::RawWaveform => {
                // Return raw samples as [1, samples] - already normalized to [-1, 1] from WAV loading
                let n_samples = samples.len();
                let input_values = ndarray::Array2::from_shape_vec((1, n_samples), samples)
                    .map_err(|e| PreprocessError::Audio(format!("Shape error: {e}")))?;

                Ok(AudioOutput::RawWaveform {
                    input_values,
                    duration_seconds,
                })
            }
        }
    }

    /// Read samples from WAV reader and convert to mono f32.
    fn read_samples<R: std::io::Read>(
        reader: &mut hound::WavReader<R>,
    ) -> PreprocessResult<Vec<f32>> {
        let spec = reader.spec();

        // Validate channel count - only mono and stereo are supported
        if spec.channels > 2 {
            return Err(PreprocessError::Audio(format!(
                "Unsupported channel count: {}. Only mono (1) and stereo (2) are supported",
                spec.channels
            )));
        }

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = dsp_cast::i32_to_f32(1 << (spec.bits_per_sample - 1));
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| dsp_cast::i32_to_f32(v) / max_val))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| PreprocessError::Audio(format!("Failed to read samples: {e}")))?
            }
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| PreprocessError::Audio(format!("Failed to read samples: {e}")))?,
        };

        // Convert to mono if stereo
        let samples = if spec.channels == 2 {
            samples
                .chunks(2)
                .map(|chunk| f32::midpoint(chunk[0], chunk.get(1).copied().unwrap_or(0.0)))
                .collect()
        } else {
            samples
        };

        Ok(samples)
    }

    /// Simple linear resampling.
    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = f64::from(to_rate) / f64::from(from_rate);
        let new_len = dsp_cast::f64_to_usize(dsp_cast::usize_to_f64(samples.len()) * ratio);
        let mut result = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = dsp_cast::usize_to_f64(i) / ratio;
            let idx = dsp_cast::f64_to_usize(src_idx);
            let frac = src_idx - dsp_cast::usize_to_f64(idx);

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - dsp_cast::f64_to_f32(frac))
                    + samples[idx + 1] * dsp_cast::f64_to_f32(frac)
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };
            result.push(sample);
        }

        result
    }

    /// Compute mel spectrogram using STFT.
    fn compute_mel_spectrogram(&self, samples: &[f32]) -> PreprocessResult<Array2<f32>> {
        let n_fft = self.config.n_fft;
        let hop_length = self.config.hop_length;

        // Number of frames
        let n_frames = 1 + (samples.len().saturating_sub(n_fft)) / hop_length;
        if n_frames == 0 {
            return Err(PreprocessError::Audio("Audio too short".into()));
        }

        // Compute STFT
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n_fft);

        // Hann window
        let n_fft_f32 = dsp_cast::usize_to_f32(n_fft);
        let window: Vec<f32> = (0..n_fft)
            .map(|i| 0.5 * (1.0 - (2.0 * PI * dsp_cast::usize_to_f32(i) / n_fft_f32).cos()))
            .collect();

        let n_freq = n_fft / 2 + 1;
        let mut power_spec = Array2::<f32>::zeros((n_freq, n_frames));

        // Pre-allocate FFT buffer outside the loop to avoid per-frame allocation
        let mut buffer: Vec<Complex<f32>> = vec![Complex::new(0.0, 0.0); n_fft];

        for frame_idx in 0..n_frames {
            let start = frame_idx * hop_length;

            // Fill buffer with windowed samples (reusing pre-allocated buffer)
            for (i, buf_elem) in buffer.iter_mut().enumerate() {
                let sample = if start + i < samples.len() {
                    samples[start + i]
                } else {
                    0.0
                };
                *buf_elem = Complex::new(sample * window[i], 0.0);
            }

            fft.process(&mut buffer);

            // Compute power spectrum
            for (freq_idx, c) in buffer.iter().take(n_freq).enumerate() {
                power_spec[[freq_idx, frame_idx]] = c.norm_sqr();
            }
        }

        // Apply mel filterbank
        let mel_spec = self.mel_filters.dot(&power_spec);

        // Log mel spectrogram (add small epsilon for numerical stability)
        let log_mel = mel_spec.mapv(|v| (v.max(1e-10)).ln());

        // Normalize (Whisper-style)
        let max_val = log_mel.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let log_mel = log_mel.mapv(|v| (v.max(max_val - 8.0) + 4.0) / 4.0);

        Ok(log_mel)
    }
}

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for AudioProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AudioProcessor")
            .field("config", &self.config)
            .finish_non_exhaustive()
    }
}

/// Create mel filterbank matrix.
fn create_mel_filterbank(sample_rate: u32, n_fft: usize, n_mels: usize) -> Array2<f32> {
    let n_freq = n_fft / 2 + 1;
    let f_min = 0.0;
    let f_max = dsp_cast::u32_to_f32(sample_rate) / 2.0;

    // Convert Hz to mel scale
    let hz_to_mel = |f: f32| 2595.0 * (1.0 + f / 700.0).log10();
    let mel_to_hz = |m: f32| 700.0 * (10.0_f32.powf(m / 2595.0) - 1.0);

    let mel_min = hz_to_mel(f_min);
    let mel_max = hz_to_mel(f_max);

    // Mel points
    let n_mels_plus_1_f32 = dsp_cast::usize_to_f32(n_mels + 1);
    let mel_points: Vec<f32> = (0..=n_mels + 1)
        .map(|i| mel_min + dsp_cast::usize_to_f32(i) * (mel_max - mel_min) / n_mels_plus_1_f32)
        .collect();

    // Convert back to Hz
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Convert to FFT bin indices
    let n_fft_plus_1_f32 = dsp_cast::usize_to_f32(n_fft + 1);
    let sample_rate_f32 = dsp_cast::u32_to_f32(sample_rate);
    let bin_points: Vec<usize> = hz_points
        .iter()
        .map(|&f| dsp_cast::f32_to_usize((n_fft_plus_1_f32 * f / sample_rate_f32).floor()))
        .collect();

    // Create filterbank
    let mut filterbank = Array2::<f32>::zeros((n_mels, n_freq));

    for m in 0..n_mels {
        for k in bin_points[m]..bin_points[m + 1] {
            if k < n_freq {
                filterbank[[m, k]] = dsp_cast::usize_to_f32(k - bin_points[m])
                    / dsp_cast::usize_to_f32((bin_points[m + 1] - bin_points[m]).max(1));
            }
        }
        for k in bin_points[m + 1]..bin_points[m + 2] {
            if k < n_freq {
                filterbank[[m, k]] = dsp_cast::usize_to_f32(bin_points[m + 2] - k)
                    / dsp_cast::usize_to_f32((bin_points[m + 2] - bin_points[m + 1]).max(1));
            }
        }
    }

    filterbank
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_config() {
        let config = AudioConfig::whisper();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.output_type, AudioOutputType::MelSpectrogram);
    }

    #[test]
    fn test_wav2vec2_config() {
        let config = AudioConfig::wav2vec2();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.output_type, AudioOutputType::RawWaveform);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let fb = create_mel_filterbank(16000, 400, 80);
        assert_eq!(fb.shape(), &[80, 201]);
    }
}
