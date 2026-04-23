//! Generation configuration for autoregressive text generation.
//!
//! This module provides shared generation parameters used across all backends
//! (ONNX seq2seq, Candle, llama.cpp), including KV cache configuration.

use crate::Config;
use serde::{Deserialize, Serialize};
use std::fmt;

/// Generation configuration for autoregressive decoding.
///
/// Used by all text generation backends to control sampling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, <1.0 = sharper, >1.0 = smoother)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (0.0-1.0)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repeats)
    pub repetition_penalty: f32,
    /// Context length for repetition penalty
    pub repeat_last_n: usize,
    /// End-of-sequence token IDs
    pub eos_token_ids: Vec<i64>,
    /// Pad token ID
    pub pad_token_id: i64,
    /// Decoder start token ID (for encoder-decoder models)
    pub decoder_start_token_id: i64,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            eos_token_ids: vec![2], // Common EOS token
            pad_token_id: 0,
            decoder_start_token_id: 0,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create from inference config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_new_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            ..Default::default()
        }
    }

    /// Builder: set max_new_tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Builder: set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder: set top_p
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Builder: set top_k
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder: set repetition_penalty
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Builder: set EOS token IDs
    #[must_use]
    pub fn with_eos_tokens(mut self, eos_token_ids: Vec<i64>) -> Self {
        self.eos_token_ids = eos_token_ids;
        self
    }

    /// Builder: set seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Model architecture type for generation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Decoder-only (GPT2, Llama, Qwen, Mistral, etc.)
    DecoderOnly,
    /// Encoder-decoder (T5, BART, Whisper, Florence-2, etc.)
    EncoderDecoder,
}

impl ModelArchitecture {
    /// Check if this is a decoder-only architecture
    pub fn is_decoder_only(&self) -> bool {
        matches!(self, Self::DecoderOnly)
    }

    /// Check if this is an encoder-decoder architecture
    pub fn is_encoder_decoder(&self) -> bool {
        matches!(self, Self::EncoderDecoder)
    }
}

// ============================================================================
// KV Cache Configuration
// ============================================================================

/// Data type for KV cache storage.
///
/// Controls how key and value tensors are stored in the attention cache.
/// Lower precision types reduce memory usage at the cost of some accuracy.
///
/// Not all backends support all types — unsupported types are logged as
/// warnings and silently fall back to the backend default.
///
/// ## Backend support matrix
///
/// | CacheDType | llama.cpp | Candle | ONNX |
/// |------------|-----------|--------|------|
/// | F32        | Yes       | Yes    | Yes  |
/// | F16        | Yes       | Possible (cast) | No |
/// | BF16       | Yes       | Device-dependent | No |
/// | Q8_0 / Q8_K| Yes       | No     | No   |
/// | Q4_0 / Q4_K| Yes       | No     | No   |
/// | Q5_K / Q6_K| Yes       | No     | No   |
/// | TQ1_0 / TQ2_0 | Yes  | No     | No   |
/// | MXFP4      | Yes       | No     | No   |
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum CacheDType {
    /// Full precision — maximum quality, most memory.
    F32,
    /// Half precision — 2x memory savings, near-lossless for most models.
    F16,
    /// BFloat16 — similar to F16, better dynamic range, hardware-dependent.
    BF16,
    /// 8-bit block-quantized — ~4x savings, nearly lossless.
    /// Recommended as a safe default for llama.cpp.
    Q8_0,
    /// 4-bit block-quantized — ~8x savings.
    /// Good for V cache; K cache is more sensitive to quantization.
    Q4_0,
    /// 4-bit K-quantized (improved, mixed precision). Better quality than Q4_0.
    Q4_K,
    /// 5-bit K-quantized (improved, mixed precision).
    Q5_K,
    /// 6-bit K-quantized (improved, mixed precision).
    Q6_K,
    /// 8-bit K-quantized (improved, mixed precision). Higher quality than Q8_0.
    Q8_K,
    /// ~1-bit TurboQuant (Google Research). Extreme compression, model-dependent quality.
    TQ1_0,
    /// ~2-bit TurboQuant (Google Research). Better quality than TQ1_0.
    TQ2_0,
    /// Microscaling FP4 (4-bit float). Experimental, hardware-dependent.
    MXFP4,
}

impl fmt::Display for CacheDType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::F32 => write!(f, "f32"),
            Self::F16 => write!(f, "f16"),
            Self::BF16 => write!(f, "bf16"),
            Self::Q8_0 => write!(f, "q8_0"),
            Self::Q4_0 => write!(f, "q4_0"),
            Self::Q4_K => write!(f, "q4_k"),
            Self::Q5_K => write!(f, "q5_k"),
            Self::Q6_K => write!(f, "q6_k"),
            Self::Q8_K => write!(f, "q8_k"),
            Self::TQ1_0 => write!(f, "tq1_0"),
            Self::TQ2_0 => write!(f, "tq2_0"),
            Self::MXFP4 => write!(f, "mxfp4"),
        }
    }
}

impl CacheDType {
    /// Parse from a string (case-insensitive).
    ///
    /// Returns `None` for unrecognized values.
    pub fn from_str_opt(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "f32" => Some(Self::F32),
            "f16" => Some(Self::F16),
            "bf16" => Some(Self::BF16),
            "q8_0" | "q8" => Some(Self::Q8_0),
            "q4_0" | "q4" => Some(Self::Q4_0),
            "q4_k" => Some(Self::Q4_K),
            "q5_k" => Some(Self::Q5_K),
            "q6_k" => Some(Self::Q6_K),
            "q8_k" => Some(Self::Q8_K),
            "tq1_0" | "turbo1" | "tq1" => Some(Self::TQ1_0),
            "tq2_0" | "turbo2" | "tq2" => Some(Self::TQ2_0),
            "mxfp4" => Some(Self::MXFP4),
            _ => None,
        }
    }
}

/// KV cache configuration shared across all inference backends.
///
/// Controls how the attention key/value cache is allocated, stored, and
/// managed. Each backend maps these settings to its native API:
///
/// - **llama.cpp**: Maps to `LlamaContextParams` (`with_type_k`, `with_type_v`,
///   `with_flash_attention_policy`, `with_offload_kqv`)
/// - **Candle**: Cache dtype controls tensor casting before storage; max length
///   enables cache truncation to prevent OOM
/// - **ONNX**: Most settings are informational; max length limits generation
///
/// ## Configuration
///
/// Via TOML:
/// ```toml
/// [inference.kv_cache]
/// cache_dtype_k = "q8_0"
/// cache_dtype_v = "q8_0"
/// max_length = 4096
/// flash_attention = true
/// offload_to_gpu = true
/// ```
///
/// Via environment variables:
/// ```text
/// MAIIA_AI_KV_CACHE_DTYPE_K=q8_0
/// MAIIA_AI_KV_CACHE_DTYPE_V=q8_0
/// MAIIA_AI_KV_CACHE_MAX_LENGTH=4096
/// MAIIA_AI_KV_CACHE_FLASH_ATTENTION=true
/// MAIIA_AI_KV_CACHE_OFFLOAD_TO_GPU=true
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KvCacheConfig {
    /// Data type for K (key) cache. `None` = backend default.
    ///
    /// The K cache is more sensitive to quantization than the V cache.
    /// Q8_0 is a safe choice; Q4_0 may degrade quality for K.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dtype_k: Option<CacheDType>,

    /// Data type for V (value) cache. `None` = backend default.
    ///
    /// The V cache tolerates more aggressive quantization than K.
    /// Both Q8_0 and Q4_0 are viable for V cache.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cache_dtype_v: Option<CacheDType>,

    /// Maximum sequence length for the KV cache.
    ///
    /// - **llama.cpp**: Maps to `n_ctx` (hard pre-allocation limit)
    /// - **Candle**: Used for cache truncation / OOM protection
    /// - **ONNX**: Limits total generation length
    ///
    /// Default: 2048.
    #[serde(default = "default_kv_cache_max_length")]
    pub max_length: usize,

    /// Enable flash attention if supported by the backend and model.
    ///
    /// Flash attention reduces memory usage and improves throughput for
    /// long sequences. Requires compatible hardware and model architecture.
    ///
    /// Note: In llama.cpp, V cache quantization requires flash attention
    /// to be enabled.
    #[serde(default)]
    pub flash_attention: bool,

    /// Offload KV cache and KQV operations to GPU (if available).
    ///
    /// Only applies to llama.cpp backend. Default: true (llama.cpp default).
    #[serde(default = "default_true_kv")]
    pub offload_to_gpu: bool,
}

fn default_kv_cache_max_length() -> usize {
    2048
}

fn default_true_kv() -> bool {
    true
}

impl Default for KvCacheConfig {
    fn default() -> Self {
        Self {
            cache_dtype_k: None,
            cache_dtype_v: None,
            max_length: default_kv_cache_max_length(),
            flash_attention: false,
            offload_to_gpu: true,
        }
    }
}

impl KvCacheConfig {
    /// Create from the main `Config`, preserving backward compatibility
    /// with `max_cache_length`.
    pub fn from_config(config: &Config) -> Self {
        config.kv_cache.clone()
    }

    /// Builder: set K cache dtype.
    #[must_use]
    pub fn with_cache_dtype_k(mut self, dtype: CacheDType) -> Self {
        self.cache_dtype_k = Some(dtype);
        self
    }

    /// Builder: set V cache dtype.
    #[must_use]
    pub fn with_cache_dtype_v(mut self, dtype: CacheDType) -> Self {
        self.cache_dtype_v = Some(dtype);
        self
    }

    /// Builder: set both K and V cache dtype to the same value.
    #[must_use]
    pub fn with_cache_dtype(mut self, dtype: CacheDType) -> Self {
        self.cache_dtype_k = Some(dtype);
        self.cache_dtype_v = Some(dtype);
        self
    }

    /// Builder: set max cache length.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = max_length;
        self
    }

    /// Builder: enable/disable flash attention.
    #[must_use]
    pub fn with_flash_attention(mut self, enabled: bool) -> Self {
        self.flash_attention = enabled;
        self
    }

    /// Builder: enable/disable GPU offloading for KV cache.
    #[must_use]
    pub fn with_offload_to_gpu(mut self, enabled: bool) -> Self {
        self.offload_to_gpu = enabled;
        self
    }
}
