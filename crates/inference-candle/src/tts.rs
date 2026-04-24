//! Candle-based Text-to-Speech (TTS) task.
//!
//! Uses candle-transformers to run TTS models natively in Rust,
//! supporting high-quality speech synthesis from text.
//!
//! ## Supported Models
//!
//! - **Parler TTS**: High-quality TTS with voice descriptions
//!   - `parler-tts/parler-tts-mini-v1` (recommended, smaller)
//!   - `parler-tts/parler-tts-large-v1` (higher quality)
//!
//! ## Architecture
//!
//! Parler TTS uses:
//! - T5 text encoder for processing input text and voice descriptions
//! - Decoder transformer for generating audio tokens
//! - DAC (Descript Audio Codec) for converting tokens to waveform
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_candle::CandleTtsTask;
//! use inference_core::Config;
//!
//! let task = CandleTtsTask::from_model_dir("/path/to/model", "task-name", &config)?;
//! let result = task.execute(r#"{"text": "Hello world"}"#, "req-1").await;
//! ```

use crate::error::{TaskError, TaskResult};
use crate::utils;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::parler_tts::{Config as ParlerConfig, Model as ParlerModel};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::{Config as AppConfig, KvCacheConfig};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Default voice description for TTS.
const DEFAULT_VOICE_DESCRIPTION: &str =
    "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up.";

/// TTS generation configuration.
#[derive(Debug, Clone)]
pub struct TtsGenConfig {
    /// Maximum generation steps.
    pub max_steps: usize,
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f64,
    /// Top-p (nucleus) sampling.
    pub top_p: Option<f64>,
    /// Random seed.
    pub seed: u64,
    /// KV cache configuration (max length, dtype preferences, etc.)
    pub kv_cache: KvCacheConfig,
}

impl Default for TtsGenConfig {
    fn default() -> Self {
        Self {
            max_steps: 2048,
            temperature: 0.0, // Greedy by default for consistent output
            top_p: None,
            seed: 299_792_458,
            kv_cache: KvCacheConfig::default(),
        }
    }
}

impl TtsGenConfig {
    /// Create from app config.
    pub fn from_config(config: &AppConfig) -> Self {
        Self {
            max_steps: config.max_tokens.min(5000),
            temperature: f64::from(config.temperature),
            top_p: if config.top_p < 1.0 {
                Some(f64::from(config.top_p))
            } else {
                None
            },
            kv_cache: KvCacheConfig::from_config(config),
            ..Default::default()
        }
    }
}

/// Internal TTS backend variant.
enum TtsBackend {
    Parler {
        model: Arc<Mutex<ParlerModel>>,
        tokenizer: Tokenizer,
    },
    Qwen3 {
        model: Arc<Mutex<crate::qwen3_tts::Qwen3TtsModel>>,
        tokenizer: Tokenizer,
    },
}

/// Candle-based TTS task supporting Parler TTS and Qwen3 TTS.
pub struct CandleTtsTask {
    /// Task name.
    name: String,
    /// Backend variant.
    backend: TtsBackend,
    /// Generation config.
    gen_config: TtsGenConfig,
    /// Candle device.
    device: Device,
    /// Compute dtype (controls weights + KV cache in Candle).
    /// Currently used only at load time; kept for diagnostics and future use.
    #[allow(dead_code)]
    dtype: DType,
    /// Audio sample rate (from model config).
    sample_rate: u32,
}

impl CandleTtsTask {
    /// Load TTS model from a directory.
    ///
    /// Expects:
    /// - `config.json` - Model configuration
    /// - `model.safetensors` or `model.safetensors.index.json` - Model weights
    /// - `tokenizer.json` - Tokenizer
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &AppConfig,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading Candle TTS model"
        );

        // Determine device
        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device for TTS");

        // Check architecture — only Parler TTS is supported
        let config_path = model_dir.join("config.json");
        let raw_config: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .map_err(|e| TaskError::ModelLoad(format!("Cannot open config.json: {e}")))?,
        )
        .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let model_type = raw_config["model_type"].as_str().unwrap_or("");
        let architectures = raw_config["architectures"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            })
            .unwrap_or_default();

        // Route to Qwen3 TTS if detected
        if model_type == "qwen3_tts"
            || architectures.contains("Qwen3TTS")
            || architectures.contains("Qwen3TTSForConditionalGeneration")
        {
            info!("Detected Qwen3 TTS architecture, delegating to qwen3_tts module");
            return Self::load_qwen3_tts(model_dir, name, config);
        }

        if model_type != "parler_tts" && !architectures.contains("Parler") {
            return Err(TaskError::ModelLoad(format!(
                "Unsupported TTS architecture: model_type={model_type:?}, architectures=[{architectures}]. \
                 Supported: Parler TTS, Qwen3 TTS."
            )));
        }

        // Load model config (Parler TTS)
        let parler_config: ParlerConfig = serde_json::from_value(raw_config)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid Parler TTS config: {e}")))?;

        let sample_rate = parler_config.audio_encoder.sampling_rate;
        info!(sample_rate = sample_rate, "Audio sample rate from config");

        // Load tokenizer
        let tokenizer = utils::load_tokenizer(model_dir)?;

        // Build generation config first — we need kv_cache settings to pick dtype
        let gen_config = TtsGenConfig::from_config(config);

        // Determine compute dtype from KV cache config + model's native dtype.
        // In Candle, KV cache dtype = model compute dtype (they can't differ).
        // TTS models are small, pass 0 for weight size.
        let model_dtype = utils::read_model_dtype(&model_dir.join("config.json"));
        let dtype = utils::resolve_compute_dtype(&gen_config.kv_cache, model_dtype, &device, 0);
        info!(dtype = ?dtype, "Compute dtype for TTS (controls weights + KV cache)");

        // Load model weights
        let model = Self::load_model(model_dir, &parler_config, &device, dtype)?;

        Ok(Self {
            name,
            backend: TtsBackend::Parler {
                model: Arc::new(Mutex::new(model)),
                tokenizer,
            },
            gen_config,
            device,
            dtype,
            sample_rate,
        })
    }

    /// Load model weights from safetensors.
    fn load_model(
        model_dir: &Path,
        config: &ParlerConfig,
        device: &Device,
        dtype: DType,
    ) -> TaskResult<ParlerModel> {
        // Check for single file or sharded weights
        let single_file = model_dir.join("model.safetensors");
        let index_file = model_dir.join("model.safetensors.index.json");

        let weight_files = if single_file.exists() {
            info!(path = %single_file.display(), "Loading single safetensors file");
            vec![single_file]
        } else if index_file.exists() {
            info!(path = %index_file.display(), "Loading sharded safetensors");
            Self::get_sharded_files(model_dir, &index_file)?
        } else {
            return Err(TaskError::ModelLoad(
                "No model.safetensors or model.safetensors.index.json found".into(),
            ));
        };

        let vb = utils::load_safetensors_safe(&weight_files, dtype, device)?;

        ParlerModel::new(config, vb)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot create model: {e}")))
    }

    /// Get list of sharded weight files from index.
    fn get_sharded_files(model_dir: &Path, index_path: &Path) -> TaskResult<Vec<PathBuf>> {
        let index: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(index_path)
                .map_err(|e| TaskError::ModelLoad(format!("Cannot open index file: {e}")))?,
        )
        .map_err(|e| TaskError::ModelLoad(format!("Invalid index file: {e}")))?;

        let weight_map = index
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| TaskError::ModelLoad("Missing weight_map in index".into()))?;

        // Collect unique shard files
        let mut shard_files: Vec<String> = weight_map
            .values()
            .filter_map(|v| v.as_str())
            .map(String::from)
            .collect();
        shard_files.sort();
        shard_files.dedup();

        let paths: Vec<PathBuf> = shard_files.into_iter().map(|f| model_dir.join(f)).collect();

        info!(num_shards = paths.len(), "Found sharded weight files");
        Ok(paths)
    }

    /// Generate speech from text.
    async fn generate_speech(
        &self,
        text: &str,
        voice_description: Option<&str>,
    ) -> TaskResult<Vec<f32>> {
        match &self.backend {
            TtsBackend::Parler { model, tokenizer } => {
                self.generate_parler(text, voice_description, model, tokenizer)
                    .await
            }
            TtsBackend::Qwen3 { model, tokenizer } => {
                self.generate_qwen3(text, model, tokenizer).await
            }
        }
    }

    /// Generate speech using Parler TTS backend.
    async fn generate_parler(
        &self,
        text: &str,
        voice_description: Option<&str>,
        model: &Arc<Mutex<ParlerModel>>,
        tokenizer: &Tokenizer,
    ) -> TaskResult<Vec<f32>> {
        let voice_desc = voice_description.unwrap_or(DEFAULT_VOICE_DESCRIPTION);

        // Tokenize description (voice style)
        let description_tokens = tokenizer
            .encode(voice_desc, true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?
            .get_ids()
            .to_vec();

        // Tokenize prompt (text to speak)
        let prompt_tokens = tokenizer
            .encode(text, true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?
            .get_ids()
            .to_vec();

        if prompt_tokens.is_empty() {
            return Err(TaskError::InvalidInput("Empty input text".into()));
        }

        debug!(
            prompt_tokens = prompt_tokens.len(),
            description_tokens = description_tokens.len(),
            "Tokenized TTS input"
        );

        // Create tensors
        let description_tensor = Tensor::new(description_tokens, &self.device)
            .map_err(|e| TaskError::Inference(format!("Tensor error: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?;

        let prompt_tensor = Tensor::new(prompt_tokens, &self.device)
            .map_err(|e| TaskError::Inference(format!("Tensor error: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?;

        // Create logits processor
        let lp = LogitsProcessor::new(
            self.gen_config.seed,
            Some(self.gen_config.temperature),
            self.gen_config.top_p,
        );

        // Generate audio codes
        let mut model = model.lock().await;

        // Cap max_steps by KV cache max_length to prevent unbounded growth
        let effective_max_steps = self
            .gen_config
            .max_steps
            .min(self.gen_config.kv_cache.max_length);
        if effective_max_steps < self.gen_config.max_steps {
            debug!(
                configured = self.gen_config.max_steps,
                capped_to = effective_max_steps,
                "TTS max_steps capped by kv_cache.max_length"
            );
        }

        let codes = model
            .generate(&prompt_tensor, &description_tensor, lp, effective_max_steps)
            .map_err(|e| TaskError::Inference(format!("Generation failed: {e}")))?;

        debug!(codes_shape = ?codes.shape(), "Generated audio codes");

        // Decode codes to audio
        let codes = codes
            .to_dtype(DType::I64)
            .map_err(|e| TaskError::Inference(format!("Dtype conversion failed: {e}")))?;

        let codes = codes
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Unsqueeze failed: {e}")))?;

        let pcm = model
            .audio_encoder
            .decode_codes(
                &codes
                    .to_device(&self.device)
                    .map_err(|e| TaskError::Inference(format!("Device transfer failed: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("Audio decoding failed: {e}")))?;

        // Extract single channel
        let pcm = pcm
            .i((0, 0))
            .map_err(|e| TaskError::Inference(format!("Index failed: {e}")))?;

        // Normalize loudness (optional, improves quality)
        let pcm = Self::normalize_loudness(&pcm)?;

        // Convert to Vec<f32>
        let samples: Vec<f32> = pcm
            .to_vec1()
            .map_err(|e| TaskError::Inference(format!("Tensor to vec failed: {e}")))?;

        info!(
            samples = samples.len(),
            duration_secs = samples.len() as f32 / self.sample_rate as f32,
            "Generated audio"
        );

        Ok(samples)
    }

    /// Normalize audio loudness using simple peak normalization.
    fn normalize_loudness(pcm: &Tensor) -> TaskResult<Tensor> {
        let max_val = pcm
            .abs()
            .map_err(|e| TaskError::Inference(format!("Abs failed: {e}")))?
            .max(0)
            .map_err(|e| TaskError::Inference(format!("Max failed: {e}")))?;

        let max_scalar: f32 = max_val
            .to_scalar()
            .map_err(|e| TaskError::Inference(format!("Scalar conversion failed: {e}")))?;

        if max_scalar > 0.99 {
            // Only normalize if clipping
            let scale = 0.95 / max_scalar;
            pcm.affine(f64::from(scale), 0.0)
                .map_err(|e| TaskError::Inference(format!("Affine failed: {e}")))
        } else {
            Ok(pcm.clone())
        }
    }

    /// Load Qwen3 TTS model.
    fn load_qwen3_tts(model_dir: &Path, name: String, config: &AppConfig) -> TaskResult<Self> {
        let device = utils::resolve_device(&config.device)?;
        let gen_config = TtsGenConfig::from_config(config);
        let model_dtype = utils::read_model_dtype(&model_dir.join("config.json"));
        let dtype = utils::resolve_compute_dtype(&gen_config.kv_cache, model_dtype, &device, 0);
        info!(dtype = ?dtype, "Compute dtype for Qwen3 TTS");

        let tokenizer = utils::load_tokenizer(model_dir)?;

        let qwen3_model =
            crate::qwen3_tts::Qwen3TtsModel::from_model_dir(model_dir, &device, dtype)?;
        let sample_rate = 24000; // Qwen3 TTS default

        Ok(Self {
            name,
            backend: TtsBackend::Qwen3 {
                model: Arc::new(Mutex::new(qwen3_model)),
                tokenizer,
            },
            gen_config,
            device,
            dtype,
            sample_rate,
        })
    }

    /// Generate speech using Qwen3 TTS backend.
    async fn generate_qwen3(
        &self,
        text: &str,
        model: &Arc<Mutex<crate::qwen3_tts::Qwen3TtsModel>>,
        tokenizer: &Tokenizer,
    ) -> TaskResult<Vec<f32>> {
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?
            .get_ids()
            .to_vec();

        if tokens.is_empty() {
            return Err(TaskError::InvalidInput("Empty input text".into()));
        }

        debug!(tokens = tokens.len(), "Tokenized Qwen3 TTS input");

        let effective_max_steps = self
            .gen_config
            .max_steps
            .min(self.gen_config.kv_cache.max_length);

        let mut model = model.lock().await;
        let output = model.generate(
            &tokens,
            None, // speaker_id — TODO: expose in input
            None, // language_id — TODO: expose in input
            effective_max_steps,
            self.gen_config.temperature,
        )?;

        if let Some(samples) = output.waveform {
            info!(
                samples = samples.len(),
                duration_secs = samples.len() as f32 / output.sample_rate as f32,
                "Qwen3 TTS generated audio"
            );
            Ok(samples)
        } else {
            // No speech tokenizer available — return silence with a warning
            tracing::warn!(
                "Speech tokenizer not loaded. Returning codec tokens as empty audio. \
                 Download speech_tokenizer/ files to enable audio output."
            );
            Ok(vec![0.0f32; 24000]) // 1 second of silence
        }
    }
}

/// Input for TTS task.
#[derive(Debug, Deserialize)]
pub struct CandleTtsInput {
    /// Text to synthesize.
    pub text: String,
    /// Optional voice description (for Parler TTS).
    /// Example: "A warm female voice with moderate pace"
    pub voice_description: Option<String>,
}

/// Output from TTS task.
#[derive(Debug, Serialize)]
pub struct CandleTtsOutput {
    /// Generated audio waveform samples.
    pub waveform: Vec<Vec<f32>>,
    /// Sample rate of the audio.
    pub sample_rate: u32,
}

#[async_trait]
impl Task for CandleTtsTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, input: &str, request_id: &str) -> GrpcTaskResult {
        debug!(
            request_id = request_id,
            input_len = input.len(),
            "Executing TTS task"
        );

        // Parse input
        let parsed: CandleTtsInput = match serde_json::from_str(input) {
            Ok(v) => v,
            Err(e) => {
                return GrpcTaskResult::err(format!("Invalid input JSON: {e}"));
            }
        };

        // Generate speech
        let samples = match self
            .generate_speech(&parsed.text, parsed.voice_description.as_deref())
            .await
        {
            Ok(s) => s,
            Err(e) => {
                return GrpcTaskResult::err(format!("TTS generation failed: {e}"));
            }
        };

        // Format output (compatible with HuggingFace TTS format)
        let output = CandleTtsOutput {
            waveform: vec![samples],
            sample_rate: self.sample_rate,
        };

        let output_json = match serde_json::to_string(&output) {
            Ok(j) => j,
            Err(e) => {
                return GrpcTaskResult::err(format!("Output serialization failed: {e}"));
            }
        };

        info!(request_id = request_id, "TTS task completed");

        GrpcTaskResult::ok(output_json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tts_input_parsing() {
        let input = r#"{"text": "Hello world"}"#;
        let parsed: CandleTtsInput = serde_json::from_str(input).unwrap();
        assert_eq!(parsed.text, "Hello world");
        assert!(parsed.voice_description.is_none());
    }

    #[test]
    fn test_tts_input_with_voice() {
        let input = r#"{"text": "Hello", "voice_description": "A deep male voice with slow pace"}"#;
        let parsed: CandleTtsInput = serde_json::from_str(input).unwrap();
        assert_eq!(parsed.text, "Hello");
        assert_eq!(
            parsed.voice_description,
            Some("A deep male voice with slow pace".to_string())
        );
    }

    #[test]
    fn test_gen_config_defaults() {
        let config = TtsGenConfig::default();
        assert_eq!(config.max_steps, 2048);
        assert!(config.temperature.abs() < f64::EPSILON);
        assert!(config.top_p.is_none());
        // KV cache should have sensible defaults
        assert_eq!(config.kv_cache.max_length, 2048);
        assert!(config.kv_cache.cache_dtype_k.is_none());
        assert!(config.kv_cache.cache_dtype_v.is_none());
    }
}
