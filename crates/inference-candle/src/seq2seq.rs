//! Generic Candle-based Seq2Seq (encoder-decoder) task.
//!
//! Uses candle-transformers to run encoder-decoder models natively in Rust,
//! avoiding ONNX KV-cache issues that affect some exported models.
//!
//! ## Supported Architectures
//!
//! - **Whisper**: Speech-to-text (ASR)
//! - **Voxtral**: Mistral's multimodal speech-to-text (ASR)
//! - **T5/FlanT5/MADLAD**: Text-to-text (translation, summarization, etc.)
//! - **Marian**: Neural machine translation
//! - **TrOCR**: Image-to-text (OCR)
//!
//! ## Architecture Detection
//!
//! Model architecture is auto-detected from `config.json` based on:
//! - `model_type` field
//! - `architectures` field
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_candle::CandleSeq2SeqTask;
//! use inference_core::Config;
//!
//! let task = CandleSeq2SeqTask::from_model_dir("/path/to/model", "task-name", &config)?;
//! let result = task.execute(r#"{"text": "translate to German: Hello"}"#, "req-1").await;
//! ```

use crate::error::{TaskError, TaskResult};
use crate::utils;
use async_trait::async_trait;
use base64::Engine;
use candle_core::{DType, Device, IndexOp, Tensor};
use inference_core::task::{Task, TaskChunk, TaskResult as GrpcTaskResult, TaskStream};
use inference_core::{Config as AppConfig, KvCacheConfig};
use rand::distributions::weighted::WeightedIndex;
use rand::distributions::Distribution;
use rand::SeedableRng;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info};

// Audio processing constants
const SAMPLE_RATE: u32 = 16000;
const TARGET_SAMPLE_RATE: u32 = 16000;

// Re-export candle model types
use candle_transformers::models::t5::{self, T5ForConditionalGeneration};
use candle_transformers::models::voxtral::{
    self, VoxtralCache, VoxtralConfig, VoxtralEncoderConfig, VoxtralForConditionalGeneration,
    VoxtralGenerationConfig, VoxtralLlamaConfig,
};
use candle_transformers::models::whisper::{self as whisper_model, audio as whisper_audio};

// Tekken tokenizer for Voxtral
use tekken::{SpecialTokenPolicy, Tekkenizer};

/// Helper to extract usize from JSON value with default.
fn json_usize(json: &serde_json::Value, key: &str, default: usize) -> usize {
    json.get(key)
        .and_then(serde_json::Value::as_u64)
        .and_then(|v| usize::try_from(v).ok())
        .unwrap_or(default)
}

/// Helper to extract f64 from JSON value with default.
fn json_f64(json: &serde_json::Value, key: &str, default: f64) -> f64 {
    json.get(key)
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(default)
}

/// Helper to extract f32 from JSON value with default.
fn json_f32(json: &serde_json::Value, key: &str, default: f32) -> f32 {
    json.get(key)
        .and_then(serde_json::Value::as_f64)
        .map_or(default, |v| v as f32)
}

/// Helper to extract bool from JSON value with default.
fn json_bool(json: &serde_json::Value, key: &str, default: bool) -> bool {
    json.get(key)
        .and_then(serde_json::Value::as_bool)
        .unwrap_or(default)
}

/// Helper to extract string from JSON value with default.
fn json_str(json: &serde_json::Value, key: &str, default: &str) -> String {
    json.get(key)
        .and_then(serde_json::Value::as_str)
        .unwrap_or(default)
        .to_string()
}

/// Tokenizer wrapper that supports both HuggingFace tokenizers and Tekken.
pub enum Seq2SeqTokenizer {
    /// Standard HuggingFace tokenizer (Whisper, T5, etc.).
    HuggingFace(Box<Tokenizer>),
    /// Mistral's Tekken tokenizer (Voxtral).
    Tekken(Box<Tekkenizer>),
}

impl Seq2SeqTokenizer {
    /// Encode text to token IDs.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> TaskResult<Vec<u32>> {
        match self {
            Self::HuggingFace(tok) => {
                let encoding = tok
                    .encode(text, add_special_tokens)
                    .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;
                Ok(encoding.get_ids().to_vec())
            }
            Self::Tekken(tok) => {
                // Tekken uses (text, add_bos, add_eos) signature, returns Result<Vec<u32>>
                tok.encode(text, add_special_tokens, false)
                    .map_err(|e| TaskError::InvalidInput(format!("Tekken encode failed: {e}")))
            }
        }
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> TaskResult<String> {
        match self {
            Self::HuggingFace(tok) => tok
                .decode(ids, skip_special_tokens)
                .map_err(|e| TaskError::Inference(format!("Decode failed: {e}"))),
            Self::Tekken(tok) => {
                let policy = if skip_special_tokens {
                    SpecialTokenPolicy::Ignore
                } else {
                    SpecialTokenPolicy::Keep
                };
                tok.decode(ids, policy)
                    .map_err(|e| TaskError::Inference(format!("Tekken decode failed: {e}")))
            }
        }
    }

    /// Get token ID for a string (HuggingFace tokenizers only).
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        match self {
            Self::HuggingFace(tok) => tok.token_to_id(token),
            Self::Tekken(_) => None, // Tekken uses hardcoded special tokens
        }
    }
}

/// Supported encoder-decoder model architectures.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Seq2SeqArch {
    /// Whisper speech recognition models.
    Whisper,
    /// Voxtral multimodal speech recognition (Mistral).
    Voxtral,
    /// T5 family (T5, FlanT5, MADLAD, CoEdit, UL2).
    T5,
    /// Marian neural machine translation.
    Marian,
}

impl Seq2SeqArch {
    /// Auto-detect architecture from config.json.
    pub fn detect_from_config(config_path: &Path) -> TaskResult<Self> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;

        let config: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let architectures = config
            .get("architectures")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_str())
                    .collect::<Vec<_>>()
                    .join(",")
            })
            .unwrap_or_default();

        info!(
            model_type = model_type,
            architectures = architectures,
            "Detecting seq2seq architecture"
        );

        let combined = format!("{model_type} {architectures}").to_lowercase();

        if combined.contains("voxtral") {
            Ok(Self::Voxtral)
        } else if combined.contains("whisper") {
            Ok(Self::Whisper)
        } else if combined.contains("t5") || combined.contains("mt5") {
            Ok(Self::T5)
        } else if combined.contains("marian") {
            Ok(Self::Marian)
        } else {
            Err(TaskError::ModelLoad(format!(
                "Unknown seq2seq architecture: model_type={model_type}, architectures={architectures}"
            )))
        }
    }
}

/// Generation configuration for seq2seq models.
#[derive(Debug, Clone)]
pub struct Seq2SeqGenConfig {
    /// Maximum tokens to generate.
    pub max_new_tokens: usize,
    /// Temperature for sampling (0.0 = greedy).
    pub temperature: f64,
    /// Top-p (nucleus) sampling.
    pub top_p: f64,
    /// Repetition penalty.
    pub repetition_penalty: f32,
    /// Random seed.
    pub seed: u64,
    /// KV cache configuration (max length, dtype preferences, etc.)
    pub kv_cache: KvCacheConfig,
}

impl Default for Seq2SeqGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.0, // Greedy by default for seq2seq
            top_p: 0.9,
            repetition_penalty: 1.0,
            seed: 299_792_458,
            kv_cache: KvCacheConfig::default(),
        }
    }
}

impl Seq2SeqGenConfig {
    /// Create from app config.
    pub fn from_config(config: &AppConfig) -> Self {
        Self {
            max_new_tokens: config.max_tokens,
            temperature: f64::from(config.temperature),
            top_p: f64::from(config.top_p),
            kv_cache: KvCacheConfig::from_config(config),
            ..Default::default()
        }
    }
}

/// Internal model wrapper for different architectures.
enum Seq2SeqModel {
    Whisper {
        model: whisper_model::model::Whisper,
        mel_filters: Vec<f32>,
        config: whisper_model::Config,
    },
    WhisperQuantized {
        model: whisper_model::quantized_model::Whisper,
        mel_filters: Vec<f32>,
        config: whisper_model::Config,
    },
    T5 {
        model: T5ForConditionalGeneration,
        config: t5::Config,
    },
    Voxtral {
        model: Box<VoxtralForConditionalGeneration>,
        config: VoxtralConfig,
        cache: VoxtralCache,
        mel_filters: Vec<f32>,
        audio_token_id: usize,
    },
}

impl Seq2SeqModel {
    /// Get EOS token ID.
    fn eos_token_id(&self) -> u32 {
        match self {
            Self::Whisper { .. } | Self::WhisperQuantized { .. } => {
                // Whisper uses <|endoftext|> which is typically 50257
                50257
            }
            Self::T5 { config, .. } => {
                // Safe conversion: vocab sizes are always < u32::MAX
                u32::try_from(config.eos_token_id).unwrap_or(1)
            }
            Self::Voxtral { .. } => {
                // Voxtral uses </s> which is token 2
                2
            }
        }
    }

    /// Get decoder start token ID.
    fn decoder_start_token_id(&self) -> u32 {
        match self {
            Self::Whisper { .. } | Self::WhisperQuantized { .. } => {
                // Whisper SOT token
                50258
            }
            Self::T5 { config, .. } => {
                // Safe conversion: token IDs are always < u32::MAX
                config
                    .decoder_start_token_id
                    .and_then(|id| u32::try_from(id).ok())
                    .unwrap_or(0)
            }
            Self::Voxtral { .. } => {
                // Voxtral uses <s> which is token 1
                1
            }
        }
    }

    /// Clear KV cache for new generation.
    fn clear_kv_cache(&mut self) {
        match self {
            Self::Whisper { model, .. } => model.reset_kv_cache(),
            Self::WhisperQuantized { model, .. } => model.reset_kv_cache(),
            Self::T5 { model, .. } => model.clear_kv_cache(),
            Self::Voxtral { cache, .. } => cache.reset(),
        }
    }
}

/// Candle-based generic seq2seq task.
pub struct CandleSeq2SeqTask {
    name: String,
    model: Arc<Mutex<Seq2SeqModel>>,
    tokenizer: Seq2SeqTokenizer,
    arch: Seq2SeqArch,
    device: Device,
    dtype: DType,
    gen_config: Seq2SeqGenConfig,
}

impl CandleSeq2SeqTask {
    /// Create from a model directory.
    ///
    /// The directory should contain:
    /// - `config.json` - Model configuration
    /// - `model.safetensors` or `*.gguf` - Model weights
    /// - `tokenizer.json` or `tekken.json` - Tokenizer (Voxtral uses tekken.json)
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        app_config: &AppConfig,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading Candle seq2seq model"
        );

        // Resolve device
        let device = utils::resolve_device(&app_config.device)?;
        info!(device = ?device, "Using device");

        // Detect architecture
        let config_path = model_dir.join("config.json");
        let arch = Seq2SeqArch::detect_from_config(&config_path)?;
        info!(architecture = ?arch, "Detected architecture");

        // Build generation config first — we need kv_cache settings to pick dtype
        let gen_config = Seq2SeqGenConfig::from_config(app_config);

        // Determine compute dtype from KV cache config.
        // In Candle, KV cache dtype = model compute dtype (they can't differ).
        let dtype = utils::resolve_compute_dtype(&gen_config.kv_cache);
        info!(dtype = ?dtype, "Compute dtype (controls weights + KV cache)");

        // Load tokenizer and model based on architecture
        let (tokenizer, model) = match arch {
            Seq2SeqArch::Voxtral => {
                // Voxtral uses Tekken tokenizer (tekken.json)
                let tekken_path = model_dir.join("tekken.json");
                let tokenizer = Tekkenizer::from_file(
                    tekken_path
                        .to_str()
                        .ok_or_else(|| TaskError::ModelLoad("Invalid tekken.json path".into()))?,
                )
                .map_err(|e| TaskError::ModelLoad(format!("Failed to load tekken.json: {e}")))?;
                info!(path = %tekken_path.display(), "Loaded Tekken tokenizer");

                let model = Self::load_voxtral(model_dir, &device, dtype)?;
                (Seq2SeqTokenizer::Tekken(Box::new(tokenizer)), model)
            }
            Seq2SeqArch::Whisper => {
                let tokenizer = Self::load_hf_tokenizer(model_dir)?;
                let model = Self::load_whisper(model_dir, &device, dtype)?;
                (Seq2SeqTokenizer::HuggingFace(Box::new(tokenizer)), model)
            }
            Seq2SeqArch::T5 => {
                let tokenizer = Self::load_hf_tokenizer(model_dir)?;
                let model = Self::load_t5(model_dir, &device, dtype)?;
                (Seq2SeqTokenizer::HuggingFace(Box::new(tokenizer)), model)
            }
            Seq2SeqArch::Marian => {
                return Err(TaskError::ModelLoad(
                    "Marian not yet implemented - use T5 or Whisper".into(),
                ));
            }
        };

        Ok(Self {
            name,
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            arch,
            device,
            dtype,
            gen_config,
        })
    }

    /// Load HuggingFace tokenizer from model directory.
    fn load_hf_tokenizer(model_dir: &Path) -> TaskResult<Tokenizer> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load tokenizer.json: {e}")))
    }

    /// Load Whisper model.
    fn load_whisper(model_dir: &Path, device: &Device, dtype: DType) -> TaskResult<Seq2SeqModel> {
        let config_path = model_dir.join("config.json");
        let config: whisper_model::Config = {
            let content = std::fs::read_to_string(&config_path)
                .map_err(|e| TaskError::ModelLoad(format!("Cannot read config: {e}")))?;
            serde_json::from_str(&content)
                .map_err(|e| TaskError::ModelLoad(format!("Invalid Whisper config: {e}")))?
        };

        // Generate mel filters
        let mel_filters = Self::generate_mel_filters(config.num_mel_bins);

        // Find weights file(s)
        let weights_paths = utils::find_weight_files(model_dir)?;

        // Check if GGUF (quantized)
        if utils::is_gguf(&weights_paths) {
            info!("Loading quantized Whisper (GGUF)");
            let vb = candle_transformers::quantized_var_builder::VarBuilder::from_gguf(
                &weights_paths[0],
                device,
            )
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load GGUF: {e}")))?;

            let model = whisper_model::quantized_model::Whisper::load(&vb, config.clone())
                .map_err(|e| TaskError::ModelLoad(format!("Failed to load Whisper: {e}")))?;

            Ok(Seq2SeqModel::WhisperQuantized {
                model,
                mel_filters,
                config,
            })
        } else {
            info!(
                num_shards = weights_paths.len(),
                "Loading Whisper (safetensors)"
            );
            let vb = utils::load_safetensors_safe(&weights_paths, dtype, device)?;

            let model = whisper_model::model::Whisper::load(&vb, config.clone())
                .map_err(|e| TaskError::ModelLoad(format!("Failed to load Whisper: {e}")))?;

            Ok(Seq2SeqModel::Whisper {
                model,
                mel_filters,
                config,
            })
        }
    }

    /// Load T5 model.
    fn load_t5(model_dir: &Path, device: &Device, dtype: DType) -> TaskResult<Seq2SeqModel> {
        let config_path = model_dir.join("config.json");
        let config: t5::Config = {
            let content = std::fs::read_to_string(&config_path)
                .map_err(|e| TaskError::ModelLoad(format!("Cannot read config: {e}")))?;
            serde_json::from_str(&content)
                .map_err(|e| TaskError::ModelLoad(format!("Invalid T5 config: {e}")))?
        };

        let weights_paths = utils::find_weight_files(model_dir)?;
        let vb = utils::load_safetensors_safe(&weights_paths, dtype, device)?;

        let model = T5ForConditionalGeneration::load(vb, &config)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load T5: {e}")))?;

        Ok(Seq2SeqModel::T5 { model, config })
    }

    /// Load Voxtral model.
    ///
    /// Voxtral is Mistral's multimodal speech-to-text model based on Ministral-3B.
    /// It uses a 128-mel filterbank and requires audio padded to multiples of 480000 samples.
    fn load_voxtral(model_dir: &Path, device: &Device, dtype: DType) -> TaskResult<Seq2SeqModel> {
        let config_path = model_dir.join("config.json");
        let config = Self::parse_voxtral_config(&config_path)?;

        info!(
            audio_token_id = config.audio_token_id,
            hidden_size = config.text_config.hidden_size,
            "Loading Voxtral model"
        );

        // Voxtral uses 128 mel bins - generate filters for audio processing
        let mel_filters = Self::generate_mel_filters(voxtral::N_MELS);

        // Load weights (sharded safetensors)
        let weights_paths = utils::find_weight_files(model_dir)?;
        info!(num_shards = weights_paths.len(), "Loading Voxtral weights");
        let vb = utils::load_safetensors_safe(&weights_paths, dtype, device)?;

        // Create model
        let model = VoxtralForConditionalGeneration::new(&config, vb)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load Voxtral: {e}")))?;

        // Create cache for KV-cache (requires text_config and device)
        let cache = VoxtralCache::new(true, dtype, &config.text_config, device)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to create Voxtral cache: {e}")))?;

        // Audio token ID from config (typically 24 in Voxtral)
        let audio_token_id = config.audio_token_id;

        Ok(Seq2SeqModel::Voxtral {
            model: Box::new(model),
            config,
            cache,
            mel_filters,
            audio_token_id,
        })
    }

    /// Parse Voxtral config from JSON file.
    ///
    /// `VoxtralConfig` doesn't implement `Deserialize`, so we parse manually.
    fn parse_voxtral_config(config_path: &Path) -> TaskResult<VoxtralConfig> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config: {e}")))?;

        let json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid JSON config: {e}")))?;

        // Parse audio config from JSON
        let audio_config = Self::parse_voxtral_audio_config(&json)?;

        // Parse text config from JSON
        let text_config = Self::parse_voxtral_text_config(&json)?;

        Ok(VoxtralConfig {
            audio_config,
            text_config,
            audio_token_id: json_usize(&json, "audio_token_id", 24),
            projector_hidden_act: json_str(&json, "projector_hidden_act", "gelu"),
        })
    }

    /// Parse audio encoder config from JSON.
    fn parse_voxtral_audio_config(json: &serde_json::Value) -> TaskResult<VoxtralEncoderConfig> {
        let audio = json
            .get("audio_config")
            .ok_or_else(|| TaskError::ModelLoad("Missing audio_config in configuration".into()))?;

        Ok(VoxtralEncoderConfig {
            vocab_size: json_usize(audio, "vocab_size", 51866),
            hidden_size: json_usize(audio, "hidden_size", 1280),
            num_hidden_layers: json_usize(audio, "num_hidden_layers", 32),
            num_attention_heads: json_usize(audio, "num_attention_heads", 20),
            num_key_value_heads: json_usize(audio, "num_key_value_heads", 20),
            intermediate_size: json_usize(audio, "intermediate_size", 5120),
            dropout: json_f64(audio, "dropout", 0.0),
            attention_dropout: json_f64(audio, "attention_dropout", 0.0),
            activation_dropout: json_f64(audio, "activation_dropout", 0.0),
            activation_function: json_str(audio, "activation_function", "gelu"),
            max_source_positions: json_usize(audio, "max_source_positions", 1500),
            layerdrop: json_f64(audio, "layerdrop", 0.0),
            initializer_range: json_f64(audio, "initializer_range", 0.02),
            scale_embedding: json_bool(audio, "scale_embedding", false),
            num_mel_bins: json_usize(audio, "num_mel_bins", 128),
            head_dim: json_usize(audio, "head_dim", 64),
        })
    }

    /// Parse text model (LLaMA) config from JSON.
    fn parse_voxtral_text_config(json: &serde_json::Value) -> TaskResult<VoxtralLlamaConfig> {
        let text = json
            .get("text_config")
            .ok_or_else(|| TaskError::ModelLoad("Missing text_config in configuration".into()))?;

        Ok(VoxtralLlamaConfig {
            vocab_size: json_usize(text, "vocab_size", 131_072),
            hidden_size: json_usize(text, "hidden_size", 3072),
            intermediate_size: json_usize(text, "intermediate_size", 8192),
            num_hidden_layers: json_usize(text, "num_hidden_layers", 30),
            num_attention_heads: json_usize(text, "num_attention_heads", 32),
            num_key_value_heads: json_usize(text, "num_key_value_heads", 8),
            head_dim: text
                .get("head_dim")
                .and_then(serde_json::Value::as_u64)
                .and_then(|v| usize::try_from(v).ok()),
            rms_norm_eps: json_f64(text, "rms_norm_eps", 1e-5),
            rope_theta: json_f32(text, "rope_theta", 100_000_000.0),
            max_position_embeddings: json_usize(text, "max_position_embeddings", 131_072),
            use_flash_attn: false,
            tie_word_embeddings: json_bool(text, "tie_word_embeddings", false),
        })
    }

    /// Generate mel filterbank for Whisper.
    ///
    /// This implements the standard mel-scale filterbank used in audio processing.
    /// All numeric conversions are safe because:
    /// - num_mel_bins is always small (80 or 128)
    /// - N_FFT is constant 400
    /// - bin_points values are bounded by N_FFT/2
    fn generate_mel_filters(num_mel_bins: usize) -> Vec<f32> {
        const N_FFT: u32 = 400;
        let n_freq = (N_FFT / 2 + 1) as usize;

        let hz_to_mel = |f: f64| 2595.0 * (1.0 + f / 700.0).log10();
        let mel_to_hz = |m: f64| 700.0 * (10.0_f64.powf(m / 2595.0) - 1.0);

        let f_max = f64::from(SAMPLE_RATE) / 2.0;
        let mel_min = hz_to_mel(0.0);
        let mel_max = hz_to_mel(f_max);

        // num_mel_bins fits in u32 (always < 256)
        let num_bins_u32 = u32::try_from(num_mel_bins).unwrap_or(128);
        let mel_points: Vec<f64> = (0..=num_bins_u32 + 1)
            .map(|i| mel_min + f64::from(i) * (mel_max - mel_min) / f64::from(num_bins_u32 + 1))
            .collect();

        let hz_points: Vec<f64> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

        // Compute bin points (always positive, bounded by N_FFT/2+1)
        let nfft_f64 = f64::from(N_FFT + 1);
        let sample_rate_f64 = f64::from(SAMPLE_RATE);
        let bin_points: Vec<usize> = hz_points
            .iter()
            .map(|&f| {
                let val = (nfft_f64 * f / sample_rate_f64).floor();
                // Clamp to valid range [0, N_FFT] then convert to usize
                val.clamp(0.0, f64::from(N_FFT)) as usize
            })
            .collect();

        let mut filterbank = vec![0.0f32; num_mel_bins * n_freq];

        for m in 0..num_mel_bins {
            let bp_m = bin_points[m];
            let bp_m1 = bin_points[m + 1];
            let bp_m2 = bin_points[m + 2];

            for k in bp_m..bp_m1 {
                if k < n_freq {
                    let denom = (bp_m1 - bp_m).max(1);
                    filterbank[m * n_freq + k] = (k - bp_m) as f32 / denom as f32;
                }
            }
            for k in bp_m1..bp_m2 {
                if k < n_freq {
                    let denom = (bp_m2 - bp_m1).max(1);
                    filterbank[m * n_freq + k] = (bp_m2 - k) as f32 / denom as f32;
                }
            }
        }

        filterbank
    }

    /// Generate text from encoder-decoder model.
    async fn generate_t5(&self, input_text: &str) -> TaskResult<String> {
        let input_ids = self.tokenizer.encode(input_text, true)?;
        if input_ids.is_empty() {
            return Err(TaskError::InvalidInput("Empty input".into()));
        }

        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|e| TaskError::Inference(format!("Input tensor error: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?;

        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let eos_token_id = model.eos_token_id();
        let decoder_start_id = model.decoder_start_token_id();

        // Encode input
        let encoder_output = match &mut *model {
            Seq2SeqModel::T5 { model, .. } => model
                .encode(&input_tensor)
                .map_err(|e| TaskError::Inference(format!("Encode failed: {e}")))?,
            Seq2SeqModel::Whisper { .. }
            | Seq2SeqModel::WhisperQuantized { .. }
            | Seq2SeqModel::Voxtral { .. } => {
                return Err(TaskError::Inference(
                    "T5 generate called on non-T5 model".into(),
                ))
            }
        };

        // Autoregressive decoding
        let mut decoder_ids = vec![decoder_start_id];
        let seed = self.gen_config.seed;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        for step in 0..self.gen_config.max_new_tokens {
            // Check if we've hit the KV cache length limit.
            if decoder_ids.len() >= self.gen_config.kv_cache.max_length {
                debug!(
                    decoder_len = decoder_ids.len(),
                    max_cache = self.gen_config.kv_cache.max_length,
                    "Stopping T5 generation: KV cache length limit reached"
                );
                break;
            }

            // On step 0, feed the full decoder sequence (e.g. [start_token]).
            // On subsequent steps, feed only the last token — the KV cache
            // already holds keys/values for all previous positions, so
            // re-feeding them would be O(n²) redundant work and would
            // duplicate entries in the cache.
            let decoder_tensor = if step == 0 {
                Tensor::new(decoder_ids.as_slice(), &self.device)
                    .map_err(|e| TaskError::Inference(format!("Decoder tensor error: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?
            } else {
                let last_token = *decoder_ids.last().unwrap();
                Tensor::new(&[last_token], &self.device)
                    .map_err(|e| TaskError::Inference(format!("Decoder tensor error: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?
            };

            let logits = match &mut *model {
                Seq2SeqModel::T5 { model, .. } => model
                    .decode(&decoder_tensor, &encoder_output)
                    .map_err(|e| TaskError::Inference(format!("Decode failed: {e}")))?,
                Seq2SeqModel::Whisper { .. }
                | Seq2SeqModel::WhisperQuantized { .. }
                | Seq2SeqModel::Voxtral { .. } => {
                    unreachable!()
                }
            };

            let next_token = self.sample_token(&logits, &mut rng)?;

            if next_token == eos_token_id {
                break;
            }

            decoder_ids.push(next_token);
        }

        // Decode tokens (skip decoder start token)
        let output_ids: Vec<u32> = decoder_ids.into_iter().skip(1).collect();
        let output_text = self.tokenizer.decode(&output_ids, true)?;

        Ok(output_text)
    }

    /// Transcribe audio with Whisper.
    async fn transcribe_whisper(&self, pcm_data: &[f32]) -> TaskResult<String> {
        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let (mel_filters, config) = match &*model {
            Seq2SeqModel::Whisper {
                mel_filters,
                config,
                ..
            }
            | Seq2SeqModel::WhisperQuantized {
                mel_filters,
                config,
                ..
            } => (mel_filters.clone(), config.clone()),
            Seq2SeqModel::T5 { .. } | Seq2SeqModel::Voxtral { .. } => {
                return Err(TaskError::Inference("Not a Whisper model".into()))
            }
        };

        // Convert PCM to mel spectrogram
        let mel = whisper_audio::pcm_to_mel(&config, pcm_data, &mel_filters);
        let mel_len = mel.len();
        let mel_tensor = Tensor::from_vec(
            mel,
            (1, config.num_mel_bins, mel_len / config.num_mel_bins),
            &self.device,
        )
        .map_err(|e| TaskError::Inference(format!("Mel tensor error: {e}")))?;

        // Encode audio
        let audio_features = match &mut *model {
            Seq2SeqModel::Whisper { model, .. } => model
                .encoder
                .forward(&mel_tensor, true)
                .map_err(|e| TaskError::Inference(format!("Encoder failed: {e}")))?,
            Seq2SeqModel::WhisperQuantized { model, .. } => model
                .encoder
                .forward(&mel_tensor, true)
                .map_err(|e| TaskError::Inference(format!("Encoder failed: {e}")))?,
            Seq2SeqModel::T5 { .. } | Seq2SeqModel::Voxtral { .. } => unreachable!(),
        };

        // Get special tokens
        let sot_token = self
            .tokenizer
            .token_to_id(whisper_model::SOT_TOKEN)
            .ok_or_else(|| TaskError::Inference("SOT token not found".into()))?;
        let eot_token = self
            .tokenizer
            .token_to_id(whisper_model::EOT_TOKEN)
            .ok_or_else(|| TaskError::Inference("EOT token not found".into()))?;
        let transcribe_token = self
            .tokenizer
            .token_to_id(whisper_model::TRANSCRIBE_TOKEN)
            .ok_or_else(|| TaskError::Inference("Transcribe token not found".into()))?;
        let no_timestamps_token = self
            .tokenizer
            .token_to_id(whisper_model::NO_TIMESTAMPS_TOKEN)
            .ok_or_else(|| TaskError::Inference("No timestamps token not found".into()))?;

        // Build initial decoder tokens
        let mut tokens = vec![sot_token, transcribe_token, no_timestamps_token];

        let seed = self.gen_config.seed;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let sample_len = config.max_target_positions / 2;

        // Autoregressive decoding
        for i in 0..sample_len {
            // Check if we've hit the KV cache length limit.
            if tokens.len() >= self.gen_config.kv_cache.max_length {
                debug!(
                    tokens_len = tokens.len(),
                    max_cache = self.gen_config.kv_cache.max_length,
                    "Stopping Whisper decoding: KV cache length limit reached"
                );
                break;
            }

            let tokens_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| TaskError::Inference(format!("Token tensor error: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?;

            let ys = match &mut *model {
                Seq2SeqModel::Whisper { model, .. } => model
                    .decoder
                    .forward(&tokens_tensor, &audio_features, i == 0)
                    .map_err(|e| TaskError::Inference(format!("Decoder failed: {e}")))?,
                Seq2SeqModel::WhisperQuantized { model, .. } => model
                    .decoder
                    .forward(&tokens_tensor, &audio_features, i == 0)
                    .map_err(|e| TaskError::Inference(format!("Decoder failed: {e}")))?,
                Seq2SeqModel::T5 { .. } | Seq2SeqModel::Voxtral { .. } => unreachable!(),
            };

            let (_, seq_len, _) = ys
                .dims3()
                .map_err(|e| TaskError::Inference(format!("Dims error: {e}")))?;

            let logits = match &*model {
                Seq2SeqModel::Whisper { model, .. } => model
                    .decoder
                    .final_linear(
                        &ys.i((..1, seq_len - 1..))
                            .map_err(|e| TaskError::Inference(format!("Index error: {e}")))?,
                    )
                    .map_err(|e| TaskError::Inference(format!("Final linear error: {e}")))?,
                Seq2SeqModel::WhisperQuantized { model, .. } => model
                    .decoder
                    .final_linear(
                        &ys.i((..1, seq_len - 1..))
                            .map_err(|e| TaskError::Inference(format!("Index error: {e}")))?,
                    )
                    .map_err(|e| TaskError::Inference(format!("Final linear error: {e}")))?,
                Seq2SeqModel::T5 { .. } | Seq2SeqModel::Voxtral { .. } => unreachable!(),
            };

            let logits = logits
                .i(0)
                .map_err(|e| TaskError::Inference(format!("Index error: {e}")))?
                .i(0)
                .map_err(|e| TaskError::Inference(format!("Index error: {e}")))?;

            let next_token = self.sample_token(&logits, &mut rng)?;

            if next_token == eot_token || tokens.len() > config.max_target_positions {
                break;
            }

            tokens.push(next_token);
        }

        // Decode tokens
        let text = self.tokenizer.decode(&tokens, true)?;

        Ok(text)
    }

    /// Transcribe audio with Voxtral.
    ///
    /// Voxtral is a multimodal LLM that processes audio through a speech encoder
    /// and generates text autoregressively. The model uses the `generate()` method
    /// which handles the audio token replacement internally.
    ///
    /// Input token sequence: `<s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:en[TRANSCRIBE]`
    /// Special tokens: BOS=1, INST=3, BEGIN_AUDIO=25, AUDIO=24, /INST=4
    async fn transcribe_voxtral(&self, pcm_data: &[f32]) -> TaskResult<String> {
        // Voxtral special token IDs (from candle-transformers example)
        const BOS_TOKEN: u32 = 1; // <s>
        const INST_TOKEN: u32 = 3; // [INST]
        const BEGIN_AUDIO: u32 = 25; // [BEGIN_AUDIO]
        const END_INST: u32 = 4; // [/INST]
        const TRANSCRIBE: u32 = 34; // [TRANSCRIBE]

        // Tokens per 30s audio chunk (fixed value from Voxtral)
        const TOKENS_PER_CHUNK: usize = 375;

        let mut model = self.model.lock().await;
        model.clear_kv_cache();

        let (mel_filters, audio_token_id, config) = match &*model {
            Seq2SeqModel::Voxtral {
                mel_filters,
                audio_token_id,
                config,
                ..
            } => (mel_filters.clone(), *audio_token_id, config.clone()),
            _ => return Err(TaskError::Inference("Not a Voxtral model".into())),
        };

        // Extract audio features using Voxtral's audio processing
        // This returns a tensor of shape (num_chunks, N_MELS, max_source_positions)
        let audio_features = voxtral::extract_features(pcm_data, &mel_filters, &self.device)
            .map_err(|e| TaskError::Inference(format!("Audio feature extraction failed: {e}")))?;

        // Get the number of audio chunks from the feature tensor
        let num_chunks = audio_features
            .dim(0)
            .map_err(|e| TaskError::Inference(format!("Audio features dim error: {e}")))?;

        // Calculate total audio tokens: chunks × 375 tokens per chunk
        let num_audio_tokens = num_chunks * TOKENS_PER_CHUNK;

        // Build prompt tokens following HuggingFace processor format:
        // <s>[INST][BEGIN_AUDIO][AUDIO]*N[/INST]lang:en[TRANSCRIBE]
        let mut input_tokens: Vec<u32> = vec![BOS_TOKEN, INST_TOKEN, BEGIN_AUDIO];

        // Add AUDIO tokens (one per position needed)
        // Safe: audio_token_id is always small (typically 24)
        let audio_token = u32::try_from(audio_token_id).unwrap_or(24);
        for _ in 0..num_audio_tokens {
            input_tokens.push(audio_token);
        }

        input_tokens.push(END_INST);

        // Add language tokens: "lang:en" encoded with tekken
        // From candle example: lang=9909, :=1058, en=1262
        input_tokens.push(9909); // lang
        input_tokens.push(1058); // :
        input_tokens.push(1262); // en

        input_tokens.push(TRANSCRIBE);

        let input_len = input_tokens.len();
        let input_ids = Tensor::new(input_tokens.as_slice(), &self.device)
            .map_err(|e| TaskError::Inference(format!("Input tensor error: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Unsqueeze error: {e}")))?;

        // Configure generation - use a fresh cache for each transcription
        let gen_cache = VoxtralCache::new(true, self.dtype, &config.text_config, &self.device)
            .map_err(|e| TaskError::Inference(format!("Failed to create generation cache: {e}")))?;

        // Cap max_new_tokens by KV cache max_length to prevent OOM.
        let max_cache = self.gen_config.kv_cache.max_length;
        let max_new_tokens = if input_len < max_cache {
            self.gen_config.max_new_tokens.min(max_cache - input_len)
        } else {
            debug!(
                input_len = input_len,
                max_cache = max_cache,
                "Voxtral input already exceeds KV cache max_length"
            );
            0
        };

        let gen_config = VoxtralGenerationConfig {
            max_new_tokens,
            temperature: self.gen_config.temperature,
            top_p: if self.gen_config.top_p > 0.0 && self.gen_config.top_p < 1.0 {
                Some(self.gen_config.top_p)
            } else {
                None
            },
            device: self.device.clone(),
            cache: Some(gen_cache),
        };

        // Generate using the model's built-in generate method
        let generated_tokens = match &*model {
            Seq2SeqModel::Voxtral { model, .. } => model
                .generate(&input_ids, Some(&audio_features), gen_config)
                .map_err(|e| TaskError::Inference(format!("Generation failed: {e}")))?,
            _ => unreachable!(),
        };

        // Extract only the newly generated tokens (skip input prompt)
        let output_tokens: Vec<u32> = if generated_tokens.len() > input_len {
            generated_tokens[input_len..].to_vec()
        } else {
            generated_tokens
        };

        // Decode the generated tokens
        let text = self.tokenizer.decode(&output_tokens, true)?;

        Ok(text.trim().to_string())
    }

    /// Sample next token from logits.
    fn sample_token(&self, logits: &Tensor, rng: &mut rand::rngs::StdRng) -> TaskResult<u32> {
        let temperature = self.gen_config.temperature;

        if temperature <= 0.0 {
            // Greedy decoding
            let logits_v: Vec<f32> = logits
                .to_vec1()
                .map_err(|e| TaskError::Inference(format!("To vec error: {e}")))?;
            let next_token = logits_v
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.total_cmp(b))
                .map_or(0, |(i, _)| u32::try_from(i).unwrap_or(0));
            Ok(next_token)
        } else {
            // Temperature sampling
            let scaled = (logits / temperature)
                .map_err(|e| TaskError::Inference(format!("Scale error: {e}")))?;
            let probs = candle_nn::ops::softmax(&scaled, 0)
                .map_err(|e| TaskError::Inference(format!("Softmax error: {e}")))?;
            let probs_v: Vec<f32> = probs
                .to_vec1()
                .map_err(|e| TaskError::Inference(format!("To vec error: {e}")))?;
            let dist = WeightedIndex::new(&probs_v)
                .map_err(|e| TaskError::Inference(format!("Distribution error: {e}")))?;
            Ok(u32::try_from(dist.sample(rng)).unwrap_or(0))
        }
    }

    /// Decode WAV bytes to PCM f32 samples at 16kHz.
    fn decode_wav(bytes: &[u8]) -> TaskResult<Vec<f32>> {
        let cursor = std::io::Cursor::new(bytes);
        let mut reader = hound::WavReader::new(cursor)
            .map_err(|e| TaskError::InvalidInput(format!("Invalid WAV: {e}")))?;

        let spec = reader.spec();

        let samples: Vec<f32> = match spec.sample_format {
            hound::SampleFormat::Int => {
                let max_val = f64::from(1i32 << (spec.bits_per_sample - 1));
                reader
                    .samples::<i32>()
                    .map(|s| s.map(|v| (f64::from(v) / max_val) as f32))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|e| TaskError::InvalidInput(format!("WAV read error: {e}")))?
            }
            hound::SampleFormat::Float => reader
                .samples::<f32>()
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| TaskError::InvalidInput(format!("WAV read error: {e}")))?,
        };

        // Convert to mono
        let samples = if spec.channels == 2 {
            samples
                .chunks(2)
                .map(|c| f32::midpoint(c[0], c.get(1).copied().unwrap_or(0.0)))
                .collect()
        } else {
            samples
        };

        // Resample to 16kHz if needed
        let samples = if spec.sample_rate == TARGET_SAMPLE_RATE {
            samples
        } else {
            Self::resample(&samples, spec.sample_rate, TARGET_SAMPLE_RATE)
        };

        Ok(samples)
    }

    /// Linear resampling.
    fn resample(samples: &[f32], from_rate: u32, to_rate: u32) -> Vec<f32> {
        let ratio = f64::from(to_rate) / f64::from(from_rate);
        let new_len = ((samples.len() as f64) * ratio) as usize;
        let mut result = Vec::with_capacity(new_len);

        for i in 0..new_len {
            let src_idx = (i as f64) / ratio;
            let idx = src_idx.floor() as usize;
            let frac = (src_idx - src_idx.floor()) as f32;

            let sample = if idx + 1 < samples.len() {
                samples[idx] * (1.0 - frac) + samples[idx + 1] * frac
            } else if idx < samples.len() {
                samples[idx]
            } else {
                0.0
            };
            result.push(sample);
        }

        result
    }

    /// Strip data URI prefix from base64 string.
    ///
    /// Handles formats like:
    /// - `data:audio/wav;base64,<data>`
    /// - `data:audio/mpeg;base64,<data>`
    /// - Raw base64 (returned as-is)
    fn strip_data_uri_prefix(data: &str) -> &str {
        if let Some(pos) = data.find(";base64,") {
            &data[pos + 8..] // Skip ";base64,"
        } else if let Some(pos) = data.find(',') {
            // Handle "data:...,<data>" without explicit base64 marker
            if data.starts_with("data:") {
                &data[pos + 1..]
            } else {
                data
            }
        } else {
            data
        }
    }
}

/// Input for seq2seq task.
#[derive(Debug, Deserialize)]
pub struct CandleSeq2SeqInput {
    /// Text input (for T5, Marian).
    pub text: Option<String>,
    /// Base64-encoded WAV audio (for Whisper).
    pub audio: Option<String>,
}

/// Output from seq2seq task.
#[derive(Debug, Serialize)]
pub struct CandleSeq2SeqOutput {
    /// Generated/transcribed text.
    pub text: String,
}

#[async_trait]
impl Task for CandleSeq2SeqTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, arch = ?self.arch, "Seq2seq task executing");

        let input: CandleSeq2SeqInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input: {e}")),
        };

        let result = match self.arch {
            Seq2SeqArch::Whisper | Seq2SeqArch::Voxtral => {
                let Some(audio_data) = &input.audio else {
                    return GrpcTaskResult::err(format!(
                        "Missing 'audio' field for {:?}",
                        self.arch
                    ));
                };

                // Strip data URI prefix if present (e.g., "data:audio/wav;base64,...")
                let base64_data = Self::strip_data_uri_prefix(audio_data);

                // Decode base64
                let bytes = match base64::engine::general_purpose::STANDARD.decode(base64_data) {
                    Ok(b) => b,
                    Err(e) => return GrpcTaskResult::err(format!("Invalid base64: {e}")),
                };

                let pcm = match Self::decode_wav(&bytes) {
                    Ok(p) => p,
                    Err(e) => return GrpcTaskResult::err(format!("WAV decode error: {e}")),
                };

                match self.arch {
                    Seq2SeqArch::Whisper => self.transcribe_whisper(&pcm).await,
                    Seq2SeqArch::Voxtral => self.transcribe_voxtral(&pcm).await,
                    _ => unreachable!(),
                }
            }
            Seq2SeqArch::T5 | Seq2SeqArch::Marian => {
                let Some(text) = &input.text else {
                    return GrpcTaskResult::err("Missing 'text' field".into());
                };

                self.generate_t5(text).await
            }
        };

        match result {
            Ok(text) => {
                let output = CandleSeq2SeqOutput { text };
                match serde_json::to_string(&output) {
                    Ok(json) => GrpcTaskResult::ok(json),
                    Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
                }
            }
            Err(e) => GrpcTaskResult::err(format!("Generation failed: {e}")),
        }
    }

    async fn execute_stream(&self, payload: &str, request_id: &str) -> TaskStream {
        // Seq2seq models don't support true streaming, return complete result
        let result = self.execute(payload, request_id).await;
        let chunk = if result.success {
            TaskChunk::final_data(result.result.unwrap_or_default())
        } else {
            TaskChunk::error(result.error.unwrap_or_default())
        };
        Box::pin(tokio_stream::once(chunk))
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn is_ready(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for CandleSeq2SeqTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleSeq2SeqTask")
            .field("name", &self.name)
            .field("arch", &self.arch)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_config_default() {
        let config = Seq2SeqGenConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert!((config.temperature - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_mel_filters_generation() {
        let filters = CandleSeq2SeqTask::generate_mel_filters(80);
        // 80 mel bins * 201 freq bins = 16080 values
        assert_eq!(filters.len(), 80 * 201);
    }
}
