//! Audio classification via Wav2Vec2 / HuBERT / CLAP.
//!
//! Implements the `Task` trait for audio classification models:
//! - **Wav2Vec2 / HuBERT**: supervised classification (CNN → Transformer → classifier head)
//! - **CLAP**: zero-shot classification (dual-encoder: audio + text, cosine similarity)
//!
//! ## Supported input formats
//!
//! - Raw PCM samples as a JSON array of floats: `{"inputs": [0.1, -0.2, ...]}`
//! - Base64-encoded WAV file: `{"inputs": "UklGRi..."}`
//!
//! ## Output format
//!
//! Returns a JSON array of `{label, score}` objects sorted by score descending,
//! matching the HuggingFace Inference API format.

pub mod config;
pub mod encoder;
pub mod feature_extractor;
pub mod model;

pub mod clap_audio_encoder;
pub mod clap_config;
pub mod clap_mel;
pub mod clap_model;
pub mod clap_swin;
pub mod clap_text_encoder;

use async_trait::async_trait;
use base64::Engine;
use candle_core::{DType, Device, Tensor};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

use crate::error::{TaskError, TaskResult};
use crate::utils;
use config::Wav2Vec2Config;

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

/// Input for audio classification.
///
/// Accepts either:
/// - `inputs` as a JSON array of f32 samples (raw PCM) or base64-encoded WAV string
/// - `audio` as a data URI (`data:audio/wav;base64,...`) or base64 string
#[derive(Debug, Deserialize)]
pub struct AudioClassificationInput {
    /// Raw float samples or base64-encoded WAV
    pub inputs: Option<serde_json::Value>,
    /// Alternative: data URI or base64 audio (from `audio` field)
    pub audio: Option<serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Audio decoding helpers
// ---------------------------------------------------------------------------

/// Decode WAV bytes (PCM) into f32 samples.
fn decode_wav_bytes(data: &[u8]) -> TaskResult<(Vec<f32>, u32)> {
    // Minimal WAV parser: RIFF header + fmt chunk + data chunk
    if data.len() < 44 || &data[0..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        return Err(TaskError::InvalidInput("Invalid WAV header".into()));
    }

    // Find fmt chunk
    let mut pos = 12;
    let mut sample_rate = 16000u32;
    let mut bits_per_sample = 16u16;
    let mut num_channels = 1u16;

    while pos + 8 <= data.len() {
        let chunk_id = &data[pos..pos + 4];
        let chunk_size =
            u32::from_le_bytes([data[pos + 4], data[pos + 5], data[pos + 6], data[pos + 7]])
                as usize;

        if chunk_id == b"fmt " && pos + 8 + chunk_size <= data.len() {
            let fmt = &data[pos + 8..pos + 8 + chunk_size];
            if fmt.len() >= 16 {
                num_channels = u16::from_le_bytes([fmt[2], fmt[3]]);
                sample_rate = u32::from_le_bytes([fmt[4], fmt[5], fmt[6], fmt[7]]);
                bits_per_sample = u16::from_le_bytes([fmt[14], fmt[15]]);
            }
            pos += 8 + chunk_size;
        } else if chunk_id == b"data" && pos + 8 <= data.len() {
            let audio_data = &data[pos + 8..data.len().min(pos + 8 + chunk_size)];
            let samples = pcm_to_f32(audio_data, bits_per_sample, num_channels)?;
            return Ok((samples, sample_rate));
        } else {
            pos += 8 + chunk_size;
        }
    }

    Err(TaskError::InvalidInput(
        "WAV file missing data chunk".into(),
    ))
}

/// Convert raw PCM bytes to mono f32 samples in [-1, 1].
fn pcm_to_f32(data: &[u8], bits: u16, channels: u16) -> TaskResult<Vec<f32>> {
    match bits {
        16 => {
            let samples: Vec<f32> = data
                .chunks_exact(2)
                .map(|c| f32::from(i16::from_le_bytes([c[0], c[1]])) / 32768.0)
                .collect();
            // Mix to mono if stereo
            if channels > 1 {
                let ch = channels as usize;
                Ok(samples
                    .chunks(ch)
                    .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                    .collect())
            } else {
                Ok(samples)
            }
        }
        32 => {
            let samples: Vec<f32> = data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if channels > 1 {
                let ch = channels as usize;
                Ok(samples
                    .chunks(ch)
                    .map(|frame| frame.iter().sum::<f32>() / ch as f32)
                    .collect())
            } else {
                Ok(samples)
            }
        }
        _ => Err(TaskError::InvalidInput(format!(
            "Unsupported bits per sample: {bits}"
        ))),
    }
}

/// Simple linear resampling from `src_rate` to `target_rate`.
fn resample(samples: &[f32], src_rate: u32, target_rate: u32) -> Vec<f32> {
    if src_rate == target_rate {
        return samples.to_vec();
    }
    let ratio = f64::from(src_rate) / f64::from(target_rate);
    let out_len = (samples.len() as f64 / ratio) as usize;
    (0..out_len)
        .map(|i| {
            let src_pos = i as f64 * ratio;
            let idx = src_pos as usize;
            let frac = src_pos - idx as f64;
            let s0 = samples[idx.min(samples.len() - 1)];
            let s1 = samples[(idx + 1).min(samples.len() - 1)];
            s0 + (s1 - s0) * frac as f32
        })
        .collect()
}

/// Normalize audio to zero mean, unit variance.
fn normalize_audio(samples: &mut [f32]) {
    if samples.is_empty() {
        return;
    }
    let n = samples.len() as f64;
    let mean = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / n;
    let var = samples
        .iter()
        .map(|&s| {
            let d = f64::from(s) - mean;
            d * d
        })
        .sum::<f64>()
        / n;
    let std = (var + 1e-7).sqrt();
    for s in samples.iter_mut() {
        *s = ((f64::from(*s) - mean) / std) as f32;
    }
}

// ---------------------------------------------------------------------------
// Main task struct — supports both Wav2Vec2 and CLAP backends
// ---------------------------------------------------------------------------

/// Internal model backend.
enum AudioModelBackend {
    /// Supervised classification: Wav2Vec2 / HuBERT / UniSpeech
    Wav2Vec2(model::Wav2Vec2ForSequenceClassification),
    /// Zero-shot classification: CLAP (audio + text encoders)
    Clap {
        model: Box<clap_model::ClapModel>,
        tokenizer: Box<tokenizers::Tokenizer>,
        candidate_labels: Vec<String>,
        mel_filters: Vec<f32>,
        num_mel_bins: usize,
    },
}

/// Candle-based audio classification task.
pub struct CandleAudioClassifierTask {
    name: String,
    backend: AudioModelBackend,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    /// For Wav2Vec2 models (CLAP uses zero-shot labels instead).
    id2label: Option<HashMap<usize, String>>,
}

impl CandleAudioClassifierTask {
    /// Create from a model directory containing config.json and safetensors.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading candle audio classification model"
        );

        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Parse config.json to determine model type
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;
        let model_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let model_type = model_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match model_type {
            "clap" => Self::load_clap(model_dir, name, config, &config_content, &device),
            "wav2vec2" | "hubert" | "unispeech" | "unispeech-sat" => Self::load_wav2vec2(
                model_dir,
                name,
                config,
                &config_content,
                &model_json,
                &device,
            ),
            other => Err(TaskError::ModelLoad(format!(
                "Unsupported audio classification architecture: '{other}'. \
                 Supported: wav2vec2, hubert, unispeech, clap."
            ))),
        }
    }

    /// Load a Wav2Vec2 / HuBERT model (supervised classification).
    fn load_wav2vec2(
        model_dir: &Path,
        name: String,
        _config: &Config,
        config_content: &str,
        model_json: &serde_json::Value,
        device: &Device,
    ) -> TaskResult<Self> {
        let wav2vec2_config: Wav2Vec2Config = serde_json::from_str(config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse Wav2Vec2Config: {e}")))?;

        let id2label = model_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let idx: usize = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect::<HashMap<usize, String>>()
            });

        info!(
            model_type = wav2vec2_config.model_type.as_deref().unwrap_or("unknown"),
            hidden_size = wav2vec2_config.hidden_size,
            num_labels = wav2vec2_config.num_labels,
            num_layers = wav2vec2_config.num_hidden_layers,
            do_stable_layer_norm = wav2vec2_config.do_stable_layer_norm,
            has_id2label = id2label.is_some(),
            backbone_prefix = wav2vec2_config.backbone_prefix(),
            "Audio classification model configuration"
        );

        let config_path = model_dir.join("config.json");
        let weight_files = utils::find_weight_files(model_dir)?;
        let is_pytorch = utils::is_pytorch_bin(&weight_files);
        info!(
            num_files = weight_files.len(),
            format = if is_pytorch {
                "pytorch_model.bin"
            } else {
                "safetensors"
            },
            "Found weight files"
        );

        let model_dtype = utils::read_model_dtype(&config_path);
        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => model_dtype.unwrap_or(DType::F32),
        };
        info!(dtype = ?dtype, "Compute dtype");

        let vb = if is_pytorch {
            utils::load_pytorch_bin(&weight_files[0], dtype, device)?
        } else {
            utils::load_safetensors_safe(&weight_files, dtype, device)?
        };

        let model = model::Wav2Vec2ForSequenceClassification::load(&vb, &wav2vec2_config)?;
        info!("Wav2Vec2 model loaded");

        Ok(Self {
            name,
            backend: AudioModelBackend::Wav2Vec2(model),
            device: device.clone(),
            dtype,
            id2label,
        })
    }

    /// Load a CLAP model (zero-shot audio classification).
    fn load_clap(
        model_dir: &Path,
        name: String,
        _config: &Config,
        config_content: &str,
        device: &Device,
    ) -> TaskResult<Self> {
        let clap_config: clap_config::ClapConfig = serde_json::from_str(config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse ClapConfig: {e}")))?;

        info!(
            projection_dim = clap_config.projection_dim,
            num_mel_bins = clap_config.audio_config.num_mel_bins,
            spec_size = clap_config.audio_config.spec_size,
            enable_fusion = clap_config.audio_config.enable_fusion,
            text_hidden_size = clap_config.text_config.hidden_size,
            text_layers = clap_config.text_config.num_hidden_layers,
            "CLAP model configuration"
        );

        // Load tokenizer (required for CLAP's text encoder)
        let tokenizer = utils::load_tokenizer(model_dir)?;

        let config_path = model_dir.join("config.json");
        let weight_files = utils::find_weight_files(model_dir)?;
        let is_pytorch = utils::is_pytorch_bin(&weight_files);
        info!(
            num_files = weight_files.len(),
            format = if is_pytorch {
                "pytorch_model.bin"
            } else {
                "safetensors"
            },
            "Found weight files"
        );

        let model_dtype = utils::read_model_dtype(&config_path);
        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => model_dtype.unwrap_or(DType::F32),
        };
        info!(dtype = ?dtype, "Compute dtype");

        let vb = if is_pytorch {
            utils::load_pytorch_bin(&weight_files[0], dtype, device)?
        } else {
            utils::load_safetensors_safe(&weight_files, dtype, device)?
        };

        let clap_model = clap_model::ClapModel::load(&vb, &clap_config)?;
        info!("CLAP model loaded");

        // Default candidate labels for zero-shot classification (AudioSet ontology common sounds).
        // Users can override by sending `candidate_labels` in the request.
        let candidate_labels = default_audio_labels();

        // Pre-compute mel filterbank
        let mel_filters = clap_mel::generate_mel_filters(
            clap_config.audio_config.num_mel_bins,
            clap_mel::CLAP_N_FFT,
            clap_mel::CLAP_SAMPLE_RATE,
            clap_mel::CLAP_FREQ_MIN,
            clap_mel::CLAP_FREQ_MAX,
        );

        Ok(Self {
            name,
            backend: AudioModelBackend::Clap {
                model: Box::new(clap_model),
                tokenizer: Box::new(tokenizer),
                candidate_labels,
                mel_filters,
                num_mel_bins: clap_config.audio_config.num_mel_bins,
            },
            device: device.clone(),
            dtype,
            id2label: None,
        })
    }

    /// Parse input JSON into f32 samples, resampling to `target_rate`.
    fn parse_audio(input: &AudioClassificationInput, target_rate: u32) -> TaskResult<Vec<f32>> {
        // Resolve the audio value: prefer `inputs`, fall back to `audio`
        let value = input
            .inputs
            .as_ref()
            .or(input.audio.as_ref())
            .ok_or_else(|| {
                TaskError::InvalidInput("Missing 'inputs' or 'audio' field in request".into())
            })?;

        match value {
            // Array of floats — raw PCM samples
            serde_json::Value::Array(arr) => {
                let samples: Vec<f32> = arr
                    .iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| f as f32)
                            .ok_or_else(|| TaskError::InvalidInput("Non-numeric sample".into()))
                    })
                    .collect::<TaskResult<_>>()?;
                Ok(samples)
            }
            // String — base64-encoded WAV, possibly with data URI prefix
            serde_json::Value::String(s) => {
                // Strip data URI prefix if present (e.g., "data:audio/wav;base64,...")
                let b64 = if let Some(idx) = s.find(",base64,") {
                    &s[idx + 8..]
                } else if s.starts_with("data:") {
                    // "data:audio/wav;base64," format
                    s.split(',').nth(1).unwrap_or(s)
                } else {
                    s.as_str()
                };

                let bytes = base64::engine::general_purpose::STANDARD
                    .decode(b64.as_bytes())
                    .map_err(|e| TaskError::InvalidInput(format!("Invalid base64: {e}")))?;
                let (mut samples, sample_rate) = decode_wav_bytes(&bytes)?;
                // Resample to target rate if needed
                if sample_rate != target_rate {
                    debug!(
                        src_rate = sample_rate,
                        target_rate = target_rate,
                        "Resampling audio"
                    );
                    samples = resample(&samples, sample_rate, target_rate);
                }
                // Normalize
                normalize_audio(&mut samples);
                Ok(samples)
            }
            _ => Err(TaskError::InvalidInput(
                "Expected 'inputs' as array of floats or base64 WAV string".into(),
            )),
        }
    }

    /// Format logits into classification results (Wav2Vec2 path).
    fn format_output(&self, logits: &Tensor) -> TaskResult<serde_json::Value> {
        // logits: [1, num_labels] → squeeze to [num_labels]
        let logits = logits
            .squeeze(0)
            .map_err(|e| TaskError::Inference(format!("squeeze: {e}")))?;

        // Softmax
        let max_val = logits
            .max(0)
            .map_err(|e| TaskError::Inference(format!("max: {e}")))?;
        let shifted = logits
            .broadcast_sub(&max_val)
            .map_err(|e| TaskError::Inference(format!("sub: {e}")))?;
        let exp = shifted
            .exp()
            .map_err(|e| TaskError::Inference(format!("exp: {e}")))?;
        let sum = exp
            .sum_all()
            .map_err(|e| TaskError::Inference(format!("sum: {e}")))?;
        let probs: Vec<f32> = exp
            .broadcast_div(&sum)
            .map_err(|e| TaskError::Inference(format!("div: {e}")))?
            .to_dtype(DType::F32)
            .map_err(|e| TaskError::Inference(format!("dtype: {e}")))?
            .to_vec1()
            .map_err(|e| TaskError::Inference(format!("to_vec1: {e}")))?;

        let mut results: Vec<serde_json::Value> = probs
            .iter()
            .enumerate()
            .map(|(i, &prob)| {
                let label = self
                    .id2label
                    .as_ref()
                    .and_then(|m| m.get(&i))
                    .cloned()
                    .unwrap_or_else(|| format!("LABEL_{i}"));
                serde_json::json!({
                    "label": label,
                    "score": prob,
                })
            })
            .collect();

        sort_results_desc(&mut results);
        Ok(serde_json::json!(results))
    }

    /// Format CLAP zero-shot scores into classification results.
    fn format_clap_output(labels: &[String], scores: &[f32]) -> serde_json::Value {
        let mut results: Vec<serde_json::Value> = labels
            .iter()
            .zip(scores.iter())
            .map(|(label, &score)| {
                serde_json::json!({
                    "label": label,
                    "score": score,
                })
            })
            .collect();

        sort_results_desc(&mut results);
        serde_json::json!(results)
    }
}

/// Sort results by score descending.
fn sort_results_desc(results: &mut [serde_json::Value]) {
    results.sort_by(|a, b| {
        b.get("score")
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
            .partial_cmp(
                &a.get("score")
                    .and_then(serde_json::Value::as_f64)
                    .unwrap_or(0.0),
            )
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

/// Wrap a result value in `{"outputs": {key: value}}` to match integration test expectations.
fn wrap_outputs(key: &str, value: serde_json::Value) -> serde_json::Value {
    serde_json::json!({ "outputs": { key: value } })
}

/// Default candidate labels for CLAP zero-shot classification.
///
/// Covers common AudioSet categories. Users can override via request payload.
fn default_audio_labels() -> Vec<String> {
    [
        "Speech",
        "Music",
        "Silence",
        "Dog",
        "Cat",
        "Bird",
        "Vehicle",
        "Alarm",
        "Laughter",
        "Crying",
        "Footsteps",
        "Water",
        "Wind",
        "Thunder",
        "Gunshot",
        "Siren",
        "Applause",
        "Knock",
        "Cough",
        "Snoring",
    ]
    .iter()
    .map(std::string::ToString::to_string)
    .collect()
}

// ---------------------------------------------------------------------------
// Task trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Task for CandleAudioClassifierTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "Audio classification execute");

        let input: AudioClassificationInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input JSON: {e}")),
        };

        match &self.backend {
            AudioModelBackend::Wav2Vec2(model) => self.execute_wav2vec2(model, &input),
            AudioModelBackend::Clap {
                model,
                tokenizer,
                candidate_labels,
                mel_filters,
                num_mel_bins,
            } => {
                // Check for user-provided candidate labels
                let input_json: serde_json::Value =
                    serde_json::from_str(payload).unwrap_or_default();
                let user_labels: Option<Vec<String>> = input_json
                    .get("candidate_labels")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(String::from))
                            .collect()
                    });
                let labels = user_labels.as_deref().unwrap_or(candidate_labels);

                self.execute_clap(model, tokenizer, labels, mel_filters, *num_mel_bins, &input)
            }
        }
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

impl CandleAudioClassifierTask {
    /// Execute Wav2Vec2/HuBERT classification.
    fn execute_wav2vec2(
        &self,
        model: &model::Wav2Vec2ForSequenceClassification,
        input: &AudioClassificationInput,
    ) -> GrpcTaskResult {
        let mut samples = match Self::parse_audio(input, 16000) {
            Ok(s) => s,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        // Only normalize if raw PCM array (WAV path normalizes in parse_audio)
        let is_raw_array = input
            .inputs
            .as_ref()
            .is_some_and(serde_json::Value::is_array);
        if is_raw_array {
            normalize_audio(&mut samples);
        }

        let num_samples = samples.len();
        debug!(num_samples, "Audio parsed (wav2vec2)");

        let waveform = match Tensor::from_vec(samples, (1, num_samples), &self.device) {
            Ok(t) => t,
            Err(e) => return GrpcTaskResult::err(format!("Failed to create tensor: {e}")),
        };

        let logits = match model.forward(&waveform) {
            Ok(l) => l,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        match self.format_output(&logits) {
            Ok(value) => GrpcTaskResult::ok(wrap_outputs("classifications", value).to_string()),
            Err(e) => GrpcTaskResult::err(e.to_string()),
        }
    }

    /// Execute CLAP zero-shot classification.
    #[allow(clippy::too_many_arguments)]
    fn execute_clap(
        &self,
        clap: &clap_model::ClapModel,
        tokenizer: &tokenizers::Tokenizer,
        labels: &[String],
        mel_filters: &[f32],
        num_mel_bins: usize,
        input: &AudioClassificationInput,
    ) -> GrpcTaskResult {
        // Parse audio at CLAP's 48kHz sample rate
        let samples = match Self::parse_audio(input, clap_mel::CLAP_SAMPLE_RATE) {
            Ok(s) => s,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        debug!(num_samples = samples.len(), "Audio parsed (CLAP)");

        // Convert to mel spectrogram: output is [1, 1, time_frames, num_mel_bins]
        let mel = match clap_mel::waveform_to_mel(&samples, num_mel_bins, mel_filters, &self.device)
        {
            Ok(m) => m,
            Err(e) => return GrpcTaskResult::err(format!("Mel spectrogram: {e}")),
        };

        // mel is already [1, 1, time, freq] — ready for audio encoder

        // Tokenize candidate labels
        let label_input_ids = match tokenize_labels(tokenizer, labels, &self.device) {
            Ok(ids) => ids,
            Err(e) => return GrpcTaskResult::err(format!("Tokenize labels: {e}")),
        };

        // Zero-shot classification
        let scores = match clap.classify_zero_shot(&mel, &label_input_ids) {
            Ok(s) => s,
            Err(e) => return GrpcTaskResult::err(format!("CLAP classify: {e}")),
        };

        let output = Self::format_clap_output(labels, &scores);
        GrpcTaskResult::ok(wrap_outputs("classifications", output).to_string())
    }
}

/// Tokenize a list of label strings into a batched tensor `[N, max_seq_len]`.
fn tokenize_labels(
    tokenizer: &tokenizers::Tokenizer,
    labels: &[String],
    device: &Device,
) -> TaskResult<Tensor> {
    let encodings: Vec<tokenizers::Encoding> = labels
        .iter()
        .map(|label| {
            // Prefix with "This is a sound of " for better CLAP zero-shot performance
            let text = format!("This is a sound of {label}");
            tokenizer
                .encode(text, true)
                .map_err(|e| TaskError::Inference(format!("tokenize '{label}': {e}")))
        })
        .collect::<TaskResult<_>>()?;

    // Find max length and pad
    let max_len = encodings
        .iter()
        .map(|e| e.get_ids().len())
        .max()
        .unwrap_or(0);

    let mut all_ids = Vec::with_capacity(labels.len() * max_len);
    for enc in &encodings {
        let ids = enc.get_ids();
        all_ids.extend(ids.iter().map(|&id| i64::from(id)));
        // Pad with 1 (RoBERTa pad token)
        all_ids.extend(std::iter::repeat_n(1i64, max_len - ids.len()));
    }

    Tensor::from_vec(all_ids, (labels.len(), max_len), device)
        .map_err(|e| TaskError::Inference(format!("label tensor: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_audio() {
        let mut samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        normalize_audio(&mut samples);
        // Mean should be ~0
        let mean: f64 = samples.iter().map(|&s| f64::from(s)).sum::<f64>() / samples.len() as f64;
        assert!(mean.abs() < 1e-5);
        // Std should be ~1
        let var: f64 = samples
            .iter()
            .map(|&s| {
                let d = f64::from(s) - mean;
                d * d
            })
            .sum::<f64>()
            / samples.len() as f64;
        assert!((var.sqrt() - 1.0).abs() < 0.1);
    }

    #[test]
    fn test_resample_same_rate() {
        let samples = vec![1.0, 2.0, 3.0];
        let out = resample(&samples, 16000, 16000);
        assert_eq!(out, samples);
    }

    #[test]
    fn test_resample_downsample() {
        let samples: Vec<f32> = (0..32000).map(|i| (i as f32) / 32000.0).collect();
        let out = resample(&samples, 32000, 16000);
        // Should be roughly half the length
        assert!(out.len().abs_diff(16000) < 2);
    }

    #[test]
    fn test_pcm_to_f32_16bit() {
        // A single 16-bit sample: 16384 → 0.5
        let data = 16384i16.to_le_bytes();
        let samples = pcm_to_f32(&data, 16, 1).unwrap();
        assert_eq!(samples.len(), 1);
        assert!((samples[0] - 0.5).abs() < 0.001);
    }
}
