//! Configuration management via environment variables and TOML files.
//!
//! Configuration is loaded in the following order (later values override earlier):
//! 1. Default values (hardcoded Rust defaults)
//! 2. Base defaults TOML (`configs/defaults.toml`, auto-discovered)
//! 3. Model-specific TOML config file (if `MAIIA_AI_CONFIG_PATH` is set)
//! 4. Environment variables with `MAIIA_AI_` prefix
//!
//! The base defaults file eliminates boilerplate across model configs. It is
//! auto-discovered relative to the model config path (same dir → parent → grandparent),
//! or can be set explicitly via `MAIIA_AI_DEFAULTS_PATH`.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

use crate::generation::{CacheDType, KvCacheConfig};

/// Environment variable prefix for all settings.
pub const ENV_PREFIX: &str = "MAIIA_AI_";

/// Data source type for model weights and data files.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DataSourceType {
    #[default]
    HuggingFace,
    S3,
    Local,
}

impl From<&str> for DataSourceType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "s3" => DataSourceType::S3,
            "local" => DataSourceType::Local,
            // Default to HuggingFace for any unrecognized value
            _ => DataSourceType::HuggingFace,
        }
    }
}

/// Task type for inference.
///
/// This is now a simple string-based type that allows ANY task type to be configured.
/// Only "echo" is treated specially (no model required). All other values create
/// a generic `OnnxTask` that can run any ONNX model.
///
/// ## Common task types (conventions, not enforced):
/// - `asr` / `speech` / `audio` - Automatic Speech Recognition (Whisper, Voxtral)
/// - `ocr` / `vision` / `image` - Optical Character Recognition (PaddleOCR, TrOCR)
/// - `chat` / `llm` / `text-generation` - Chat/LLM completion (GPT, Llama, Mistral)
/// - `embed` / `embedding` / `feature-extraction` - Embeddings (BGE-M3, all-MiniLM)
/// - `classification` / `text-classification` - Text classification
/// - `ner` / `token-classification` - Named entity recognition
/// - `qa` / `question-answering` - Question answering
/// - `summarization` - Text summarization
/// - `translation` - Machine translation
/// - `echo` / `test` - Echo task for testing (no model)
///
/// You can use ANY string value - the service is fully generic.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(transparent)]
pub struct TaskType(pub String);

impl TaskType {
    /// Create a new task type from a string.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Check if this is the echo/test task type.
    pub fn is_echo(&self) -> bool {
        let s = self.0.to_lowercase();
        s == "echo" || s == "test"
    }

    /// Check if this task type requires seq2seq (autoregressive) generation.
    ///
    /// Seq2seq tasks include:
    /// - text-generation (decoder-only, GPT2/Llama)
    /// - translation (encoder-decoder, T5/MarianMT)
    /// - summarization (encoder-decoder, BART/T5)
    /// - automatic-speech-recognition (encoder-decoder, Whisper)
    /// - image-to-text (encoder-decoder, TrOCR)
    /// - text-to-speech (encoder-decoder, SpeechT5)
    /// - document-question-answering (encoder-decoder)
    /// - image-text-to-text (encoder-decoder, vision-language)
    /// - audio-text-to-text (encoder-decoder)
    /// - visual-question-answering (encoder-decoder, Florence-2)
    pub fn is_seq2seq(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        matches!(
            s.as_str(),
            "text_generation"
                | "translation"
                | "summarization"
                | "automatic_speech_recognition"
                | "image_to_text"
                | "text_to_speech"
                | "document_question_answering"
                | "image_text_to_text"
                | "audio_text_to_text"
                | "visual_question_answering"
        )
    }

    /// Check if this is a decoder-only seq2seq task (GPT2, Llama, etc.).
    /// These models only have a decoder, no separate encoder.
    pub fn is_decoder_only(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        s == "text_generation"
    }

    /// Check if this is a text-to-speech task.
    pub fn is_tts(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        s == "text_to_speech"
    }

    /// Check if this is an encoder-only task (single forward pass, no autoregressive loop).
    ///
    /// Currently limited to text-based encoder models (BERT, RoBERTa, DistilBERT):
    /// - token-classification (NER, POS tagging)
    /// - text-classification / sentiment-analysis
    /// - fill-mask (masked language modeling)
    /// - feature-extraction / sentence-similarity (embeddings)
    /// - question-answering (extractive QA)
    /// - zero-shot-classification
    pub fn is_encoder_only(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        matches!(
            s.as_str(),
            "token_classification"
                | "text_classification"
                | "sentiment_analysis"
                | "fill_mask"
                | "feature_extraction"
                | "sentence_similarity"
                | "question_answering"
                | "zero_shot_classification"
        )
    }

    /// Check if this is an audio classification task.
    pub fn is_audio_classification(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        s == "audio_classification"
    }

    /// Check if this is an object detection task.
    pub fn is_object_detection(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        s == "object_detection"
    }

    /// Check if this is an image-to-text (image captioning) task.
    pub fn is_image_to_text(&self) -> bool {
        let s = self.0.to_lowercase().replace('-', "_");
        s == "image_to_text"
    }

    /// Get the raw string value.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for TaskType {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for TaskType {
    fn from(s: String) -> Self {
        Self(s)
    }
}

impl std::fmt::Display for TaskType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Device type for inference.
///
/// - `Auto` (default): probe for GPU at runtime, fall back to CPU if unavailable.
/// - `Cpu`: force CPU inference.
/// - `Gpu`: use the platform-preferred GPU (Metal on macOS, CUDA elsewhere).
/// - `Cuda` / `Metal`: force a specific GPU backend.
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    #[default]
    Auto,
    Cpu,
    /// Platform-preferred GPU: Metal on macOS, CUDA elsewhere.
    Gpu,
    Cuda,
    Metal,
}

impl DeviceType {
    /// Resolve `Gpu` to platform-specific backend.
    /// Returns `Metal` on macOS, `Cuda` elsewhere.
    /// `Auto` is **not** resolved here — use [`auto_detect_device`] first.
    #[must_use]
    pub fn resolve(&self) -> Self {
        match self {
            Self::Gpu => {
                if cfg!(target_os = "macos") {
                    Self::Metal
                } else {
                    Self::Cuda
                }
            }
            other => other.clone(),
        }
    }

    /// Returns `true` if no explicit device preference was set.
    #[must_use]
    pub const fn is_auto(&self) -> bool {
        matches!(self, Self::Auto)
    }

    /// Returns `true` if this device type is a GPU variant (Metal, Cuda, or generic Gpu).
    #[must_use]
    pub const fn is_gpu(&self) -> bool {
        matches!(self, Self::Metal | Self::Cuda | Self::Gpu)
    }
}

impl From<&str> for DeviceType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gpu" => Self::Gpu,
            "cuda" => Self::Cuda,
            "metal" | "mps" => Self::Metal,
            "cpu" => Self::Cpu,
            // "auto" and any unrecognized value → auto-detect
            _ => Self::Auto,
        }
    }
}

/// Probe for GPU availability at runtime and return the best device.
///
/// Detection strategy:
/// - **macOS**: assume Metal is available (Apple Silicon Macs all have Metal).
/// - **Linux/Windows**: assume CUDA may be available.
/// - Each backend validates the actual hardware in its own `resolve_device`
///   and falls back to CPU if the GPU probe fails.
///
/// This is a lightweight platform check. The actual device creation and
/// hardware validation happens in each backend's device resolver.
#[must_use]
pub fn auto_detect_device() -> DeviceType {
    if cfg!(target_os = "macos") {
        info!("Auto-detected device: Metal (macOS)");
        DeviceType::Metal
    } else {
        info!("Auto-detected device: CUDA (non-macOS, will fall back to CPU if unavailable)");
        DeviceType::Cuda
    }
}

/// Backend type for inference.
///
/// - `Onnx`: Use ONNX Runtime (good for embeddings, classification, older models)
/// - `Candle`: Use Candle (Rust-native, good for modern LLMs like Qwen3, Llama 3.x)
/// - `Llama`: Use llama.cpp via llama-cpp-2 (for GGUF models like GPT-OSS-20B)
/// - `Auto`: Auto-detect based on task type (text-generation → Candle, others → ONNX)
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum BackendType {
    #[default]
    Auto,
    Onnx,
    Candle,
    Llama,
}

impl From<&str> for BackendType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "onnx" => BackendType::Onnx,
            "candle" => BackendType::Candle,
            "llama" | "gguf" | "llama.cpp" | "llama-cpp" => BackendType::Llama,
            _ => BackendType::Auto,
        }
    }
}

// ============================================================================
// TOML Config File Structure
// ============================================================================

/// Task configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TaskConfig {
    /// Task type: any string value (e.g., "asr", "ocr", "chat", "embed", "classification", etc.)
    /// Only "echo" is special (no model required). All other values use OnnxTask.
    #[serde(default)]
    pub r#type: TaskType,

    /// Task name (e.g., "maiia.chat.v1")
    #[serde(default)]
    pub name: Option<String>,
}

/// Model configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ModelConfig {
    /// Model source: huggingface, s3, local
    #[serde(default)]
    pub source: DataSourceType,

    /// Model path/ID (HF repo ID, S3 prefix, or local path)
    #[serde(default)]
    pub path: Option<String>,

    /// Model revision (for HuggingFace)
    #[serde(default)]
    pub revision: Option<String>,

    /// ONNX model filename (if using ONNX)
    #[serde(default)]
    pub onnx_file: Option<String>,

    /// GGUF model filename (if using Llama backend)
    /// e.g., "gpt-oss-20b-Q4_K_M.gguf"
    #[serde(default)]
    pub gguf_file: Option<String>,
}

/// Inference configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceConfig {
    /// Backend: auto, onnx, candle, llama
    /// Auto selects candle for text-generation, onnx for others
    #[serde(default)]
    pub backend: BackendType,

    /// Device: auto (default), cpu, cuda, metal, gpu.
    /// When absent from TOML, auto-detection picks the best available device.
    #[serde(default)]
    pub device: Option<DeviceType>,

    /// Maximum tokens for generation (chat/LLM)
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,

    /// Temperature for generation
    #[serde(default = "default_temperature")]
    pub temperature: f32,

    /// Top-p sampling
    #[serde(default = "default_top_p")]
    pub top_p: f32,

    /// Number of threads for CPU inference
    #[serde(default = "default_num_threads")]
    pub num_threads: usize,

    /// Maximum KV-cache sequence length for pre-allocation.
    /// Required for CoreML/Metal which cannot handle zero-length dynamic dimensions.
    /// Set to expected max context length (prompt + generation). Default: 2048.
    #[serde(default = "default_max_cache_length")]
    pub max_cache_length: usize,

    /// Number of GPU layers to offload for llama.cpp backend.
    /// Set to a high value (e.g., 99) to offload all layers to GPU.
    /// Default: 0 (CPU only).
    #[serde(default)]
    pub n_gpu_layers: u32,

    /// Idle timeout in seconds before model is unloaded automatically.
    /// Set to 0 to disable automatic unloading (model stays loaded indefinitely).
    /// Default: 0 (disabled).
    #[serde(default)]
    pub idle_timeout_seconds: u64,

    /// KV cache configuration for attention cache management.
    ///
    /// Controls cache dtype quantization, max length, flash attention,
    /// and GPU offloading. See [`KvCacheConfig`] for details.
    ///
    /// ```toml
    /// [inference.kv_cache]
    /// cache_dtype_k = "q8_0"
    /// cache_dtype_v = "q8_0"
    /// max_length = 4096
    /// flash_attention = true
    /// ```
    #[serde(default)]
    pub kv_cache: KvCacheConfig,

    /// Extra model-specific parameters
    #[serde(default, flatten)]
    pub extra: HashMap<String, toml::Value>,
}

fn default_max_tokens() -> usize {
    2048
}
fn default_temperature() -> f32 {
    0.7
}
fn default_top_p() -> f32 {
    0.9
}
fn default_num_threads() -> usize {
    4
}
fn default_max_cache_length() -> usize {
    2048
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            backend: BackendType::default(),
            device: None,
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            num_threads: default_num_threads(),
            max_cache_length: default_max_cache_length(),
            n_gpu_layers: 0,
            kv_cache: KvCacheConfig::default(),
            extra: HashMap::new(),
            idle_timeout_seconds: 300,
        }
    }
}

/// S3 configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct S3Config {
    /// S3 bucket name
    #[serde(default)]
    pub bucket: Option<String>,

    /// S3 prefix for model files
    #[serde(default)]
    pub model_prefix: Option<String>,

    /// S3 prefix for data files
    #[serde(default)]
    pub data_prefix: Option<String>,

    /// AWS region
    #[serde(default)]
    pub region: Option<String>,

    /// Custom endpoint URL (for MinIO)
    #[serde(default)]
    pub endpoint: Option<String>,
}

/// Service configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceConfig {
    /// gRPC server port
    #[serde(default = "default_grpc_port")]
    pub grpc_port: u16,

    /// HTTP server port
    #[serde(default = "default_http_port")]
    pub http_port: u16,

    /// Health check port
    #[serde(default = "default_health_port")]
    pub health_port: u16,

    /// Service name for registration
    #[serde(default = "default_service_name")]
    pub name: String,

    /// Enable request batching
    #[serde(default = "default_true")]
    pub enable_batching: bool,

    /// Maximum batch size
    #[serde(default = "default_max_batch_size")]
    pub max_batch_size: usize,

    /// Batch timeout in milliseconds
    #[serde(default = "default_batch_timeout_ms")]
    pub batch_timeout_ms: u64,

    /// Cache directory
    #[serde(default = "default_cache_dir")]
    pub cache_dir: String,

    /// Enable caching
    #[serde(default = "default_true")]
    pub enable_cache: bool,

    /// Per-request timeout in milliseconds (0 = no timeout). Default: 300_000 (5 min).
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,

    /// Maximum payload size in bytes. Default: 10 MiB.
    #[serde(default = "default_max_payload_size_bytes")]
    pub max_payload_size_bytes: usize,

    /// Maximum concurrent in-flight requests (0 = unlimited). Default: 64.
    #[serde(default = "default_max_concurrent_requests")]
    pub max_concurrent_requests: usize,

    /// Path to TLS certificate PEM file. Both `tls_cert_path` and `tls_key_path`
    /// must be set to enable TLS.
    #[serde(default)]
    pub tls_cert_path: Option<String>,

    /// Path to TLS private key PEM file. Both `tls_cert_path` and `tls_key_path`
    /// must be set to enable TLS.
    #[serde(default)]
    pub tls_key_path: Option<String>,
}

fn default_grpc_port() -> u16 {
    50051
}
fn default_http_port() -> u16 {
    8000
}
fn default_health_port() -> u16 {
    8080
}
fn default_service_name() -> String {
    "maiia-ai-worker".to_string()
}
fn default_true() -> bool {
    true
}
fn default_max_batch_size() -> usize {
    32
}
fn default_batch_timeout_ms() -> u64 {
    100
}
fn default_cache_dir() -> String {
    "/tmp/inference-cache".to_string()
}
fn default_request_timeout_ms() -> u64 {
    300_000 // 5 minutes
}
fn default_max_payload_size_bytes() -> usize {
    10 * 1024 * 1024 // 10 MiB
}
fn default_max_concurrent_requests() -> usize {
    64
}

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            grpc_port: default_grpc_port(),
            http_port: default_http_port(),
            health_port: default_health_port(),
            name: default_service_name(),
            enable_batching: default_true(),
            max_batch_size: default_max_batch_size(),
            batch_timeout_ms: default_batch_timeout_ms(),
            cache_dir: default_cache_dir(),
            enable_cache: default_true(),
            request_timeout_ms: default_request_timeout_ms(),
            max_payload_size_bytes: default_max_payload_size_bytes(),
            max_concurrent_requests: default_max_concurrent_requests(),
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

/// HuggingFace configuration section in TOML.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HuggingFaceConfig {
    /// HuggingFace API token
    #[serde(default)]
    pub token: Option<String>,

    /// Cache directory for HF downloads
    #[serde(default)]
    pub cache_dir: Option<String>,
}

// ============================================================================
// Full TOML Config Structure
// ============================================================================

/// Complete TOML configuration file structure.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TomlConfig {
    /// Task configuration
    #[serde(default)]
    pub task: TaskConfig,

    /// Model configuration
    #[serde(default)]
    pub model: ModelConfig,

    /// Inference configuration
    #[serde(default)]
    pub inference: InferenceConfig,

    /// S3 configuration
    #[serde(default)]
    pub s3: S3Config,

    /// Service configuration
    #[serde(default)]
    pub service: ServiceConfig,

    /// HuggingFace configuration
    #[serde(default)]
    pub huggingface: HuggingFaceConfig,
}

impl TomlConfig {
    /// Load configuration from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> crate::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .map_err(|e| crate::Error::Config(format!("Failed to read config file: {e}")))?;

        toml::from_str(&content)
            .map_err(|e| crate::Error::Config(format!("Failed to parse TOML config: {e}")))
    }
}

// ============================================================================
// Main Config (merged from TOML + ENV)
// ============================================================================

/// Main configuration struct, populated from TOML file and environment variables.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    // Task Configuration
    pub task_type: TaskType,
    pub task_name: String,

    // Model Configuration
    pub model_source: DataSourceType,
    pub model_path: Option<String>,
    pub model_revision: Option<String>,
    pub onnx_file: Option<String>,
    pub gguf_file: Option<String>,

    // Inference Configuration
    pub backend: BackendType,
    pub device: DeviceType,
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: f32,
    pub num_threads: usize,
    pub max_cache_length: usize,
    pub n_gpu_layers: u32,

    // KV Cache Configuration
    pub kv_cache: KvCacheConfig,

    // Idle timeout configuration
    pub idle_timeout_seconds: u64,

    // HuggingFace Configuration
    pub hf_token: Option<String>,
    pub hf_cache_dir: Option<String>,

    // S3 Configuration
    pub s3_bucket: Option<String>,
    pub s3_model_prefix: Option<String>,
    pub s3_data_prefix: Option<String>,
    pub s3_region: Option<String>,
    pub s3_endpoint: Option<String>,

    // Service Configuration
    pub grpc_port: u16,
    #[serde(default = "default_http_port")]
    pub http_port: u16,
    pub health_port: u16,
    pub service_name: String,
    pub enable_batching: bool,
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub cache_dir: String,
    pub enable_cache: bool,

    // Server hardening
    /// Per-request timeout in milliseconds (0 = no timeout). Default: 300_000 (5 min).
    pub request_timeout_ms: u64,
    /// Maximum payload size in bytes. Default: 10 MiB.
    pub max_payload_size_bytes: usize,
    /// Maximum concurrent in-flight requests (0 = unlimited). Default: 64.
    pub max_concurrent_requests: usize,

    // TLS
    /// Path to TLS certificate PEM file. Both must be set to enable TLS.
    pub tls_cert_path: Option<String>,
    /// Path to TLS private key PEM file. Both must be set to enable TLS.
    pub tls_key_path: Option<String>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            task_type: TaskType::default(),
            task_name: "maiia.echo.v1".to_string(),
            model_source: DataSourceType::default(),
            model_path: None,
            model_revision: None,
            onnx_file: None,
            gguf_file: None,
            backend: BackendType::default(),
            device: DeviceType::default(),
            max_tokens: 2048,
            temperature: 0.7,
            top_p: 0.9,
            num_threads: 4,
            max_cache_length: 2048,
            n_gpu_layers: 0,
            idle_timeout_seconds: 0,
            kv_cache: KvCacheConfig::default(),
            hf_token: None,
            hf_cache_dir: None,
            s3_bucket: None,
            s3_model_prefix: None,
            s3_data_prefix: None,
            s3_region: None,
            s3_endpoint: None,
            grpc_port: 50051,
            http_port: 8000,
            health_port: 8080,
            service_name: "maiia-ai-worker".to_string(),
            enable_batching: true,
            max_batch_size: 32,
            batch_timeout_ms: 100,
            cache_dir: "/tmp/inference-cache".to_string(),
            enable_cache: true,
            request_timeout_ms: 300_000,              // 5 minutes
            max_payload_size_bytes: 10 * 1024 * 1024, // 10 MiB
            max_concurrent_requests: 64,
            tls_cert_path: None,
            tls_key_path: None,
        }
    }
}

impl Config {
    /// Load configuration from environment variables and optional TOML file.
    ///
    /// Load order:
    /// 1. Default values (hardcoded Rust defaults)
    /// 2. Base defaults TOML (`configs/defaults.toml` sibling to the model config)
    /// 3. Model-specific TOML file (if `MAIIA_AI_CONFIG_PATH` is set)
    /// 4. Environment variables with `MAIIA_AI_` prefix
    ///
    /// The base defaults file is auto-discovered: if the model config is at
    /// `configs/nlp/text-classification.toml`, the loader looks for
    /// `configs/defaults.toml` (parent or grandparent directory).
    /// This can be overridden with `MAIIA_AI_DEFAULTS_PATH`.
    pub fn load() -> crate::Result<Self> {
        let mut config = Self::default();

        // Try to load from TOML file first
        if let Ok(config_path) = std::env::var("MAIIA_AI_CONFIG_PATH") {
            info!("Loading config from TOML file: {config_path}");

            // Auto-discover defaults.toml relative to the model config
            let defaults_path = Self::find_defaults_path(&config_path);
            if let Some(ref dp) = defaults_path {
                info!("Loading base defaults from: {}", dp.display());
                let defaults_toml = TomlConfig::from_file(dp)?;
                config.apply_toml(&defaults_toml);
            }

            // Apply model-specific config (overrides defaults)
            let toml_config = TomlConfig::from_file(&config_path)?;
            config.apply_toml(&toml_config);
        }

        // Override with environment variables
        config.apply_env();

        // If device is still Auto after all layers, probe for GPU
        config.resolve_auto_device();

        Ok(config)
    }

    /// Find the `defaults.toml` file relative to a model config path.
    ///
    /// Search order:
    /// 1. `MAIIA_AI_DEFAULTS_PATH` env var (explicit override)
    /// 2. Same directory as the config file (e.g., `configs/nlp/defaults.toml`)
    /// 3. Parent directory (e.g., `configs/defaults.toml` when config is `configs/nlp/foo.toml`)
    /// 4. Grandparent directory (for deeply nested configs)
    ///
    /// Returns `None` if no defaults file is found (not an error — defaults are optional).
    fn find_defaults_path(config_path: &str) -> Option<std::path::PathBuf> {
        // Explicit override via env var
        if let Ok(explicit) = std::env::var("MAIIA_AI_DEFAULTS_PATH") {
            let p = std::path::PathBuf::from(&explicit);
            if p.exists() {
                return Some(p);
            }
            tracing::warn!("MAIIA_AI_DEFAULTS_PATH={explicit} does not exist, ignoring");
        }

        let config = std::path::Path::new(config_path);
        let config_dir = config.parent()?;

        // Check same directory
        let candidate = config_dir.join("defaults.toml");
        if candidate.exists() && candidate.as_path() != config {
            return Some(candidate);
        }

        // Check parent directory
        if let Some(parent) = config_dir.parent() {
            let candidate = parent.join("defaults.toml");
            if candidate.exists() {
                return Some(candidate);
            }

            // Check grandparent directory
            if let Some(grandparent) = parent.parent() {
                let candidate = grandparent.join("defaults.toml");
                if candidate.exists() {
                    return Some(candidate);
                }
            }
        }

        None
    }

    /// Load from environment only (for backwards compatibility).
    pub fn from_env() -> Self {
        let mut config = Self::default();
        config.apply_env();
        config
    }

    /// Apply TOML configuration values.
    fn apply_toml(&mut self, toml: &TomlConfig) {
        // Task
        self.task_type.clone_from(&toml.task.r#type);
        if let Some(ref name) = toml.task.name {
            self.task_name.clone_from(name);
        }

        // Model
        self.model_source.clone_from(&toml.model.source);
        self.model_path.clone_from(&toml.model.path);
        self.model_revision.clone_from(&toml.model.revision);
        self.onnx_file.clone_from(&toml.model.onnx_file);
        self.gguf_file.clone_from(&toml.model.gguf_file);

        // Inference
        self.backend.clone_from(&toml.inference.backend);
        if let Some(ref d) = toml.inference.device {
            self.device = d.clone();
        }
        self.max_tokens = toml.inference.max_tokens;
        self.temperature = toml.inference.temperature;
        self.top_p = toml.inference.top_p;
        self.num_threads = toml.inference.num_threads;
        self.max_cache_length = toml.inference.max_cache_length;
        self.n_gpu_layers = toml.inference.n_gpu_layers;

        // KV Cache — use the nested [inference.kv_cache] section.
        // For backward compatibility: if kv_cache.max_length is still the default (2048)
        // but max_cache_length was explicitly set to something else, use max_cache_length.
        self.kv_cache = toml.inference.kv_cache.clone();
        if self.kv_cache.max_length == 2048 && toml.inference.max_cache_length != 2048 {
            self.kv_cache.max_length = toml.inference.max_cache_length;
        }

        // HuggingFace
        self.hf_token.clone_from(&toml.huggingface.token);
        self.hf_cache_dir.clone_from(&toml.huggingface.cache_dir);

        // S3
        self.s3_bucket.clone_from(&toml.s3.bucket);
        self.s3_model_prefix.clone_from(&toml.s3.model_prefix);
        self.s3_data_prefix.clone_from(&toml.s3.data_prefix);
        self.s3_region.clone_from(&toml.s3.region);
        self.s3_endpoint.clone_from(&toml.s3.endpoint);

        // Service
        self.grpc_port = toml.service.grpc_port;
        self.http_port = toml.service.http_port;
        self.health_port = toml.service.health_port;
        self.service_name.clone_from(&toml.service.name);
        self.enable_batching = toml.service.enable_batching;
        self.max_batch_size = toml.service.max_batch_size;
        self.batch_timeout_ms = toml.service.batch_timeout_ms;
        self.cache_dir.clone_from(&toml.service.cache_dir);
        self.enable_cache = toml.service.enable_cache;
        self.request_timeout_ms = toml.service.request_timeout_ms;
        self.max_payload_size_bytes = toml.service.max_payload_size_bytes;
        self.max_concurrent_requests = toml.service.max_concurrent_requests;
        self.tls_cert_path.clone_from(&toml.service.tls_cert_path);
        self.tls_key_path.clone_from(&toml.service.tls_key_path);
    }

    /// Apply environment variable overrides.
    fn apply_env(&mut self) {
        // Helper to get env var with MAIIA_AI_ prefix
        let get_env = |key: &str| std::env::var(format!("{ENV_PREFIX}{key}")).ok();

        // Task
        if let Some(v) = get_env("TASK_TYPE") {
            self.task_type = TaskType::from(v.as_str());
        }
        if let Some(v) = get_env("TASK_NAME") {
            self.task_name = v;
        }

        // Model
        if let Some(v) = get_env("MODEL_SOURCE") {
            self.model_source = DataSourceType::from(v.as_str());
        }
        if let Some(v) = get_env("MODEL_PATH") {
            self.model_path = Some(v);
        }
        if let Some(v) = get_env("MODEL_REVISION") {
            self.model_revision = Some(v);
        }
        if let Some(v) = get_env("ONNX_FILE") {
            self.onnx_file = Some(v);
        }
        if let Some(v) = get_env("GGUF_FILE") {
            self.gguf_file = Some(v);
        }

        // Inference
        if let Some(v) = get_env("BACKEND") {
            self.backend = BackendType::from(v.as_str());
        }
        if let Some(v) = get_env("DEVICE") {
            self.device = DeviceType::from(v.as_str());
        }
        if let Some(v) = get_env("MAX_TOKENS").and_then(|s| s.parse().ok()) {
            self.max_tokens = v;
        }
        if let Some(v) = get_env("TEMPERATURE").and_then(|s| s.parse().ok()) {
            self.temperature = v;
        }
        if let Some(v) = get_env("TOP_P").and_then(|s| s.parse().ok()) {
            self.top_p = v;
        }
        if let Some(v) = get_env("NUM_THREADS").and_then(|s| s.parse().ok()) {
            self.num_threads = v;
        }
        if let Some(v) = get_env("MAX_CACHE_LENGTH").and_then(|s| s.parse().ok()) {
            self.max_cache_length = v;
            // Also update kv_cache.max_length for backward compatibility
            self.kv_cache.max_length = v;
        }
        if let Some(v) = get_env("N_GPU_LAYERS").and_then(|s| s.parse().ok()) {
            self.n_gpu_layers = v;
        }

        // KV Cache specific env vars (override TOML [inference.kv_cache] section)
        if let Some(v) = get_env("KV_CACHE_DTYPE_K").and_then(|s| CacheDType::from_str_opt(&s)) {
            self.kv_cache.cache_dtype_k = Some(v);
        }
        if let Some(v) = get_env("KV_CACHE_DTYPE_V").and_then(|s| CacheDType::from_str_opt(&s)) {
            self.kv_cache.cache_dtype_v = Some(v);
        }
        if let Some(v) = get_env("KV_CACHE_MAX_LENGTH").and_then(|s| s.parse().ok()) {
            self.kv_cache.max_length = v;
        }
        if let Some(v) = get_env("KV_CACHE_FLASH_ATTENTION") {
            self.kv_cache.flash_attention = v.to_lowercase() == "true" || v == "1";
        }
        if let Some(v) = get_env("KV_CACHE_OFFLOAD_TO_GPU") {
            self.kv_cache.offload_to_gpu = v.to_lowercase() == "true" || v == "1";
        }

        // HuggingFace (also check HF_TOKEN without prefix for compatibility)
        if let Some(v) = get_env("HF_TOKEN").or_else(|| std::env::var("HF_TOKEN").ok()) {
            self.hf_token = Some(v);
        }
        if let Some(v) = get_env("HF_CACHE_DIR") {
            self.hf_cache_dir = Some(v);
        }

        // S3
        if let Some(v) = get_env("S3_BUCKET") {
            self.s3_bucket = Some(v);
        }
        if let Some(v) = get_env("S3_MODEL_PREFIX") {
            self.s3_model_prefix = Some(v);
        }
        if let Some(v) = get_env("S3_DATA_PREFIX") {
            self.s3_data_prefix = Some(v);
        }
        if let Some(v) = get_env("S3_REGION").or_else(|| std::env::var("AWS_REGION").ok()) {
            self.s3_region = Some(v);
        }
        if let Some(v) = get_env("S3_ENDPOINT") {
            self.s3_endpoint = Some(v);
        }

        // Service
        if let Some(v) = get_env("GRPC_PORT").and_then(|s| s.parse().ok()) {
            self.grpc_port = v;
        }
        if let Some(v) = get_env("HTTP_PORT").and_then(|s| s.parse().ok()) {
            self.http_port = v;
        }
        if let Some(v) = get_env("HEALTH_PORT").and_then(|s| s.parse().ok()) {
            self.health_port = v;
        }
        if let Some(v) = get_env("SERVICE_NAME") {
            self.service_name = v;
        }
        if let Some(v) = get_env("ENABLE_BATCHING") {
            self.enable_batching = v.to_lowercase() == "true" || v == "1";
        }
        if let Some(v) = get_env("MAX_BATCH_SIZE").and_then(|s| s.parse().ok()) {
            self.max_batch_size = v;
        }
        if let Some(v) = get_env("BATCH_TIMEOUT_MS").and_then(|s| s.parse().ok()) {
            self.batch_timeout_ms = v;
        }
        if let Some(v) = get_env("CACHE_DIR") {
            self.cache_dir = v;
        }
        if let Some(v) = get_env("ENABLE_CACHE") {
            self.enable_cache = v.to_lowercase() == "true" || v == "1";
        }
        if let Some(v) = get_env("REQUEST_TIMEOUT_MS").and_then(|s| s.parse().ok()) {
            self.request_timeout_ms = v;
        }
        if let Some(v) = get_env("MAX_PAYLOAD_SIZE_BYTES").and_then(|s| s.parse().ok()) {
            self.max_payload_size_bytes = v;
        }
        if let Some(v) = get_env("MAX_CONCURRENT_REQUESTS").and_then(|s| s.parse().ok()) {
            self.max_concurrent_requests = v;
        }
        if let Some(v) = get_env("TLS_CERT_PATH") {
            self.tls_cert_path = Some(v);
        }
        if let Some(v) = get_env("TLS_KEY_PATH") {
            self.tls_key_path = Some(v);
        }
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
        // === Inference parameter range checks ===
        if self.temperature < 0.0 || self.temperature > 2.0 {
            return Err(crate::Error::Config(format!(
                "temperature must be in [0.0, 2.0], got {}",
                self.temperature
            )));
        }
        if self.top_p < 0.0 || self.top_p > 1.0 {
            return Err(crate::Error::Config(format!(
                "top_p must be in [0.0, 1.0], got {}",
                self.top_p
            )));
        }
        if self.max_tokens == 0 || self.max_tokens > 1_000_000 {
            return Err(crate::Error::Config(format!(
                "max_tokens must be in [1, 1_000_000], got {}",
                self.max_tokens
            )));
        }
        if self.num_threads == 0 || self.num_threads > 256 {
            return Err(crate::Error::Config(format!(
                "num_threads must be in [1, 256], got {}",
                self.num_threads
            )));
        }
        if self.max_cache_length == 0 {
            return Err(crate::Error::Config(
                "max_cache_length must be > 0".to_string(),
            ));
        }

        // === Service parameter range checks ===
        if self.grpc_port == 0 {
            return Err(crate::Error::Config("grpc_port must be > 0".to_string()));
        }
        if self.max_batch_size == 0 {
            return Err(crate::Error::Config(
                "max_batch_size must be > 0".to_string(),
            ));
        }
        if self.max_payload_size_bytes == 0 {
            return Err(crate::Error::Config(
                "max_payload_size_bytes must be > 0".to_string(),
            ));
        }

        // === TLS validation ===
        match (&self.tls_cert_path, &self.tls_key_path) {
            (Some(_), None) => {
                return Err(crate::Error::Config(
                    "tls_cert_path is set but tls_key_path is missing — both are required for TLS"
                        .to_string(),
                ));
            }
            (None, Some(_)) => {
                return Err(crate::Error::Config(
                    "tls_key_path is set but tls_cert_path is missing — both are required for TLS"
                        .to_string(),
                ));
            }
            (Some(cert), Some(key)) => {
                let cert_path = std::path::Path::new(cert);
                let key_path = std::path::Path::new(key);
                if !cert_path.exists() {
                    return Err(crate::Error::Config(format!(
                        "tls_cert_path does not exist: {cert}"
                    )));
                }
                if !key_path.exists() {
                    return Err(crate::Error::Config(format!(
                        "tls_key_path does not exist: {key}"
                    )));
                }
            }
            (None, None) => {} // TLS disabled, fine
        }

        // Echo task doesn't need model validation
        if self.task_type.is_echo() {
            return Ok(());
        }

        // Validate based on model source
        match self.model_source {
            DataSourceType::HuggingFace => {
                if self.model_path.is_none() {
                    return Err(crate::Error::Config(
                        "MAIIA_AI_MODEL_PATH (HF model ID) required when MODEL_SOURCE=huggingface"
                            .to_string(),
                    ));
                }
            }
            DataSourceType::S3 => {
                if self.s3_bucket.is_none() {
                    return Err(crate::Error::Config(
                        "MAIIA_AI_S3_BUCKET required when MODEL_SOURCE=s3".to_string(),
                    ));
                }
            }
            DataSourceType::Local => {
                if self.model_path.is_none() {
                    return Err(crate::Error::Config(
                        "MAIIA_AI_MODEL_PATH required when MODEL_SOURCE=local".to_string(),
                    ));
                }
            }
        }

        Ok(())
    }

    /// Generate task name: explicit > model-derived > task-type fallback.
    pub fn effective_task_name(&self) -> String {
        // If task_name was explicitly set (not the default sentinel), use it.
        if self.task_name != "maiia.echo.v1" || self.task_type.is_echo() {
            return self.task_name.clone();
        }

        // Derive from model_path: "onnx-community/nsfw_image_detection-ONNX" → "onnx-community.nsfw_image_detection-ONNX.v1"
        if let Some(ref model_path) = self.model_path {
            if !model_path.is_empty() {
                let model_slug = model_path.replace('/', ".");
                return format!("{model_slug}.v1");
            }
        }

        // Fallback to task type
        format!("maiia.{}.v1", self.task_type)
    }

    /// Returns true if TLS is configured (both cert and key paths are set).
    pub fn tls_enabled(&self) -> bool {
        self.tls_cert_path.is_some() && self.tls_key_path.is_some()
    }

    /// Resolve `DeviceType::Auto` to a concrete device by probing for GPU.
    ///
    /// Also sets performance defaults for Llama backend when a GPU is detected:
    /// - `n_gpu_layers = 99` (offload all layers)
    /// - `flash_attention = true` (faster attention on Metal/CUDA)
    /// - `cache_dtype_k/v = Q8_0` (reduces memory bandwidth, ~lossless)
    fn resolve_auto_device(&mut self) {
        // If device is still Auto, probe for GPU
        if self.device.is_auto() {
            self.device = auto_detect_device();
        }

        // Resolve generic "Gpu" to the platform-specific variant (Metal/CUDA).
        // This ensures downstream code sees a concrete device type.
        if self.device == DeviceType::Gpu {
            self.device = self.device.resolve();
        }

        // For llama backend: if a GPU device is selected (or auto-detected) and
        // n_gpu_layers is still 0 (default = no GPU layers), offload all layers
        // to GPU automatically. This applies regardless of whether the user set
        // --device gpu explicitly or it was auto-detected.
        if self.device.is_gpu()
            && self.n_gpu_layers == 0
            && matches!(self.backend, BackendType::Llama)
        {
            info!("Auto-setting n_gpu_layers=99 for llama backend with GPU");
            self.n_gpu_layers = 99;
        }

        // For llama backend with GPU: auto-enable flash attention and Q8_0
        // KV cache if the user hasn't explicitly configured them.
        // Flash attention is a major throughput win on Metal/CUDA.
        // Q8_0 KV cache reduces memory bandwidth with negligible quality loss.
        if self.device.is_gpu() && matches!(self.backend, BackendType::Llama) {
            if !self.kv_cache.flash_attention {
                info!("Auto-enabling flash attention for llama backend with GPU");
                self.kv_cache.flash_attention = true;
            }
            if self.kv_cache.cache_dtype_k.is_none() {
                info!("Auto-setting KV cache dtype to Q8_0 for llama backend with GPU");
                self.kv_cache.cache_dtype_k = Some(crate::generation::CacheDType::Q8_0);
            }
            if self.kv_cache.cache_dtype_v.is_none() {
                self.kv_cache.cache_dtype_v = Some(crate::generation::CacheDType::Q8_0);
            }
        }
    }

    /// Build a Config from a model ID and auto-derived parameters.
    ///
    /// This is the "no TOML" path: given a model ID, task type, and backend hints
    /// (all derived from HF metadata in the caller), produce a ready-to-use Config.
    ///
    /// The caller (CLI layer) queries the HF API and passes in the derived values.
    /// This method applies defaults → model args → env overrides.
    #[allow(clippy::too_many_arguments)]
    pub fn from_model_args(
        model_id: &str,
        task_type: &str,
        backend: &str,
        onnx_file: Option<&str>,
        gguf_file: Option<&str>,
        device: Option<&str>,
        max_tokens: Option<usize>,
        temperature: Option<f32>,
        top_p: Option<f32>,
        num_threads: Option<usize>,
        port: Option<u16>,
        http_port: Option<u16>,
        n_gpu_layers: Option<u32>,
    ) -> Self {
        let mut config = Self::default();

        // Try to apply defaults.toml if discoverable from CWD
        let defaults_candidates = ["configs/defaults.toml", "../configs/defaults.toml"];
        for candidate in &defaults_candidates {
            let path = std::path::Path::new(candidate);
            if path.exists() {
                if let Ok(defaults_toml) = TomlConfig::from_file(path) {
                    config.apply_toml(&defaults_toml);
                }
                break;
            }
        }

        // Core model settings
        config.model_path = Some(model_id.to_string());
        config.model_source = DataSourceType::HuggingFace;
        config.task_type = TaskType::new(task_type);
        config.backend = BackendType::from(backend);

        if let Some(f) = onnx_file {
            config.onnx_file = Some(f.to_string());
        }
        if let Some(f) = gguf_file {
            config.gguf_file = Some(f.to_string());
        }

        // Auto-derive batching from task type
        config.enable_batching = !config.task_type.is_seq2seq();

        // Auto-derive service_name and task_name from model_id.
        // "onnx-community/nsfw_image_detection-ONNX" → "onnx-community.nsfw_image_detection-ONNX.v1"
        let model_slug = model_id.replace('/', ".");
        config.service_name = format!("{model_slug}-worker");
        config.task_name = format!("{model_slug}.v1");

        // Apply optional overrides
        if let Some(d) = device {
            config.device = DeviceType::from(d);
        }
        if let Some(v) = max_tokens {
            config.max_tokens = v;
        }
        if let Some(v) = temperature {
            config.temperature = v;
        }
        if let Some(v) = top_p {
            config.top_p = v;
        }
        if let Some(v) = num_threads {
            config.num_threads = v;
        }
        if let Some(v) = port {
            config.grpc_port = v;
        }
        if let Some(v) = http_port {
            config.http_port = v;
        }
        if let Some(v) = n_gpu_layers {
            config.n_gpu_layers = v;
        }

        // Env overrides are always last
        config.apply_env();

        // If device is still Auto after all layers, probe for GPU
        config.resolve_auto_device();

        config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a Config set to echo task type for validation tests.
    fn echo_config() -> Config {
        Config {
            task_type: TaskType::from("echo"),
            ..Config::default()
        }
    }

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.grpc_port, 50051);
        assert_eq!(config.health_port, 8080);
        assert!(config.enable_batching);
        // Default task type is empty string
        assert_eq!(config.task_type, TaskType::default());
    }

    #[test]
    fn test_task_type_is_echo() {
        assert!(TaskType::from("echo").is_echo());
        assert!(TaskType::from("ECHO").is_echo());
        assert!(TaskType::from("test").is_echo());
        assert!(TaskType::from("TEST").is_echo());
        assert!(!TaskType::from("chat").is_echo());
        assert!(!TaskType::from("asr").is_echo());
        assert!(!TaskType::from("custom-task").is_echo());
    }

    #[test]
    fn test_task_type_any_string() {
        // Any string can be a task type
        let task = TaskType::from("my-custom-classifier");
        assert_eq!(task.as_str(), "my-custom-classifier");
        assert!(!task.is_echo());

        let task2 = TaskType::new("another.task.v2");
        assert_eq!(task2.to_string(), "another.task.v2");
    }

    #[test]
    fn test_data_source_from_str() {
        assert_eq!(DataSourceType::from("s3"), DataSourceType::S3);
        assert_eq!(DataSourceType::from("S3"), DataSourceType::S3);
        assert_eq!(DataSourceType::from("local"), DataSourceType::Local);
        assert_eq!(
            DataSourceType::from("huggingface"),
            DataSourceType::HuggingFace
        );
        assert_eq!(DataSourceType::from("hf"), DataSourceType::HuggingFace);
    }

    #[test]
    fn test_toml_parse() {
        let toml_str = r#"
[task]
type = "chat"
name = "maiia.chat.v1"

[model]
source = "s3"
path = "models/gpt-oss-20b"

[inference]
device = "cuda"
max_tokens = 4096
temperature = 0.8

[s3]
bucket = "my-bucket"
region = "us-east-1"

[service]
grpc_port = 50052
"#;

        let toml_config: TomlConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(toml_config.task.r#type, TaskType::from("chat"));
        assert_eq!(toml_config.model.source, DataSourceType::S3);
        assert_eq!(toml_config.inference.max_tokens, 4096);
        assert_eq!(toml_config.service.grpc_port, 50052);
    }

    #[test]
    fn test_toml_parse_custom_task_type() {
        let toml_str = r#"
[task]
type = "custom-ner-model"
name = "mycompany.ner.v3"

[model]
source = "local"
path = "/models/ner"
onnx_file = "model.onnx"
"#;

        let toml_config: TomlConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(toml_config.task.r#type, TaskType::from("custom-ner-model"));
        assert_eq!(toml_config.task.name, Some("mycompany.ner.v3".to_string()));
        assert!(!toml_config.task.r#type.is_echo());
    }

    #[test]
    fn test_validate_temperature_range() {
        let mut config = echo_config();

        config.temperature = -0.1;
        assert!(config.validate().is_err());

        config.temperature = 2.1;
        assert!(config.validate().is_err());

        config.temperature = 0.0;
        assert!(config.validate().is_ok());

        config.temperature = 2.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_top_p_range() {
        let mut config = echo_config();

        config.top_p = -0.1;
        assert!(config.validate().is_err());

        config.top_p = 1.1;
        assert!(config.validate().is_err());

        config.top_p = 0.0;
        assert!(config.validate().is_ok());

        config.top_p = 1.0;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_max_tokens() {
        let mut config = echo_config();

        config.max_tokens = 0;
        assert!(config.validate().is_err());

        config.max_tokens = 1_000_001;
        assert!(config.validate().is_err());

        config.max_tokens = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_validate_num_threads() {
        let mut config = echo_config();

        config.num_threads = 0;
        assert!(config.validate().is_err());

        config.num_threads = 257;
        assert!(config.validate().is_err());

        config.num_threads = 1;
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_default_hardening_values() {
        let config = Config::default();
        assert_eq!(config.request_timeout_ms, 300_000);
        assert_eq!(config.max_payload_size_bytes, 10 * 1024 * 1024);
        assert_eq!(config.max_concurrent_requests, 64);
    }

    #[test]
    fn test_toml_parse_hardening_config() {
        let toml_str = r#"
[task]
type = "echo"

[service]
request_timeout_ms = 60000
max_payload_size_bytes = 5242880
max_concurrent_requests = 128
"#;

        let toml_config: TomlConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(toml_config.service.request_timeout_ms, 60000);
        assert_eq!(toml_config.service.max_payload_size_bytes, 5_242_880);
        assert_eq!(toml_config.service.max_concurrent_requests, 128);
    }

    #[test]
    fn test_tls_defaults_disabled() {
        let config = Config::default();
        assert!(config.tls_cert_path.is_none());
        assert!(config.tls_key_path.is_none());
        assert!(!config.tls_enabled());
    }

    #[test]
    fn test_tls_validation_cert_without_key() {
        let mut config = echo_config();
        config.tls_cert_path = Some("/tmp/cert.pem".to_string());
        config.tls_key_path = None;
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("tls_key_path is missing"), "got: {err}");
    }

    #[test]
    fn test_tls_validation_key_without_cert() {
        let mut config = echo_config();
        config.tls_cert_path = None;
        config.tls_key_path = Some("/tmp/key.pem".to_string());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("tls_cert_path is missing"), "got: {err}");
    }

    #[test]
    fn test_tls_validation_nonexistent_files() {
        let mut config = echo_config();
        config.tls_cert_path = Some("/nonexistent/cert.pem".to_string());
        config.tls_key_path = Some("/nonexistent/key.pem".to_string());
        let err = config.validate().unwrap_err().to_string();
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[test]
    fn test_tls_toml_parse() {
        let toml_str = r#"
[task]
type = "echo"

[service]
tls_cert_path = "/etc/tls/server.crt"
tls_key_path = "/etc/tls/server.key"
"#;

        let toml_config: TomlConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(
            toml_config.service.tls_cert_path,
            Some("/etc/tls/server.crt".to_string())
        );
        assert_eq!(
            toml_config.service.tls_key_path,
            Some("/etc/tls/server.key".to_string())
        );

        // Verify apply_toml propagates to Config
        let mut config = Config::default();
        config.apply_toml(&toml_config);
        assert!(config.tls_enabled());
    }

    /// Validates that ALL config files in configs/ parse correctly and pass validation.
    ///
    /// This test mirrors the real `Config::load()` behavior: for each model config,
    /// it first applies `configs/defaults.toml` (if present), then the model config,
    /// then validates. This ensures the layered config system works end-to-end.
    #[test]
    fn test_all_config_files_parse_and_validate() {
        // Recursively find all .toml files
        fn collect_toml_files(dir: &std::path::Path, out: &mut Vec<std::path::PathBuf>) {
            if let Ok(entries) = std::fs::read_dir(dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.is_dir() {
                        collect_toml_files(&path, out);
                    } else if path.extension().is_some_and(|ext| ext == "toml") {
                        out.push(path);
                    }
                }
            }
        }

        // Walk up from the crate root to find the workspace configs/ directory.
        // The crate is at crates/inference-core/, so workspace root is ../../
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("could not find workspace root");
        let configs_dir = workspace_root.join("configs");

        assert!(
            configs_dir.is_dir(),
            "configs/ directory not found at {}",
            configs_dir.display()
        );

        // Load base defaults if present
        let defaults_path = configs_dir.join("defaults.toml");
        let defaults_toml = if defaults_path.exists() {
            Some(TomlConfig::from_file(&defaults_path).expect("defaults.toml should parse"))
        } else {
            None
        };

        let mut toml_files = Vec::new();
        collect_toml_files(&configs_dir, &mut toml_files);
        toml_files.sort();

        assert!(
            !toml_files.is_empty(),
            "no .toml files found in {}",
            configs_dir.display()
        );

        let mut count = 0;
        let mut errors = Vec::new();

        for path in &toml_files {
            // Skip defaults.toml itself — it's not a model config
            if path.file_name().is_some_and(|f| f == "defaults.toml") {
                continue;
            }

            count += 1;
            let relative = path
                .strip_prefix(workspace_root)
                .unwrap_or(path)
                .display()
                .to_string();

            // Step 1: Parse as TomlConfig
            let toml_config = match TomlConfig::from_file(path) {
                Ok(tc) => tc,
                Err(e) => {
                    errors.push(format!("{relative}: TOML parse error: {e}"));
                    continue;
                }
            };

            // Step 2: Apply defaults first, then model config (mirrors Config::load())
            let mut config = Config::default();
            if let Some(ref defaults) = defaults_toml {
                config.apply_toml(defaults);
            }
            config.apply_toml(&toml_config);

            if let Err(e) = config.validate() {
                errors.push(format!("{relative}: validation error: {e}"));
            }
        }

        assert!(
            errors.is_empty(),
            "{} of {count} config files failed:\n  - {}",
            errors.len(),
            errors.join("\n  - ")
        );

        // Sanity check: we found a reasonable number of configs
        assert!(
            count >= 30,
            "expected at least 30 config files, found {count}"
        );
    }

    #[test]
    fn test_defaults_layering() {
        // Simulate: defaults set device=cpu, num_threads=4, grpc_port=50051
        // Model config overrides only task + model + service.name
        let defaults_str = r#"
[model]
source = "huggingface"

[inference]
device = "cpu"
num_threads = 4

[service]
grpc_port = 50051
health_port = 8080
enable_batching = true
max_batch_size = 32
batch_timeout_ms = 100
"#;

        let model_str = r#"
[task]
type = "feature-extraction"
name = "maiia.feature-extraction.v1"

[model]
path = "Xenova/all-MiniLM-L6-v2"
onnx_file = "onnx/model.onnx"

[service]
name = "maiia-feature-extraction-worker"
max_batch_size = 64
batch_timeout_ms = 50
"#;

        let defaults_toml: TomlConfig = toml::from_str(defaults_str).unwrap();
        let model_toml: TomlConfig = toml::from_str(model_str).unwrap();

        let mut config = Config::default();
        config.apply_toml(&defaults_toml);
        config.apply_toml(&model_toml);

        // From defaults
        assert_eq!(config.device, DeviceType::Cpu);
        assert_eq!(config.num_threads, 4);
        assert_eq!(config.grpc_port, 50051);
        assert!(config.enable_batching);

        // From model config (overrides defaults)
        assert_eq!(config.task_type, TaskType::from("feature-extraction"));
        assert_eq!(
            config.model_path,
            Some("Xenova/all-MiniLM-L6-v2".to_string())
        );
        assert_eq!(config.onnx_file, Some("onnx/model.onnx".to_string()));
        assert_eq!(config.service_name, "maiia-feature-extraction-worker");
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.batch_timeout_ms, 50);
    }

    #[test]
    fn test_find_defaults_path() {
        // Test with a path that has configs/defaults.toml as parent
        let manifest_dir = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let workspace_root = manifest_dir
            .parent()
            .and_then(|p| p.parent())
            .expect("workspace root");
        let configs_dir = workspace_root.join("configs");

        if configs_dir.join("defaults.toml").exists() {
            // A config in configs/nlp/ should find configs/defaults.toml
            let fake_config = configs_dir.join("nlp").join("some-model.toml");
            let result = Config::find_defaults_path(fake_config.to_str().unwrap());
            assert!(result.is_some(), "should find defaults.toml");
            assert!(
                result.unwrap().ends_with("defaults.toml"),
                "should point to defaults.toml"
            );
        }
    }
}
