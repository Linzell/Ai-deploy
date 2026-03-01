//! Configuration management via environment variables and TOML files.
//!
//! Configuration is loaded in the following order (later values override earlier):
//! 1. Default values
//! 2. TOML config file (if `MAIIA_AI_CONFIG_PATH` is set)
//! 3. Environment variables with `MAIIA_AI_` prefix

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::info;

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
#[derive(Debug, Clone, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum DeviceType {
    #[default]
    Cpu,
    /// Auto-detect GPU: Metal on macOS, CUDA on Linux/Windows
    Gpu,
    Cuda,
    Metal,
}

impl DeviceType {
    /// Resolve `Gpu` to platform-specific backend.
    /// Returns `Metal` on macOS, `Cuda` elsewhere.
    #[must_use]
    pub fn resolve(&self) -> Self {
        match self {
            DeviceType::Gpu => {
                if cfg!(target_os = "macos") {
                    DeviceType::Metal
                } else {
                    DeviceType::Cuda
                }
            }
            other => other.clone(),
        }
    }
}

impl From<&str> for DeviceType {
    fn from(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "gpu" => DeviceType::Gpu,
            "cuda" => DeviceType::Cuda,
            "metal" | "mps" => DeviceType::Metal,
            _ => DeviceType::Cpu,
        }
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

    /// Device: cpu, cuda, metal
    #[serde(default)]
    pub device: DeviceType,

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
            device: DeviceType::default(),
            max_tokens: default_max_tokens(),
            temperature: default_temperature(),
            top_p: default_top_p(),
            num_threads: default_num_threads(),
            max_cache_length: default_max_cache_length(),
            n_gpu_layers: 0,
            extra: HashMap::new(),
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
}

fn default_grpc_port() -> u16 {
    50051
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

impl Default for ServiceConfig {
    fn default() -> Self {
        Self {
            grpc_port: default_grpc_port(),
            health_port: default_health_port(),
            name: default_service_name(),
            enable_batching: default_true(),
            max_batch_size: default_max_batch_size(),
            batch_timeout_ms: default_batch_timeout_ms(),
            cache_dir: default_cache_dir(),
            enable_cache: default_true(),
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
    pub health_port: u16,
    pub service_name: String,
    pub enable_batching: bool,
    pub max_batch_size: usize,
    pub batch_timeout_ms: u64,
    pub cache_dir: String,
    pub enable_cache: bool,
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
            hf_token: None,
            hf_cache_dir: None,
            s3_bucket: None,
            s3_model_prefix: None,
            s3_data_prefix: None,
            s3_region: None,
            s3_endpoint: None,
            grpc_port: 50051,
            health_port: 8080,
            service_name: "maiia-ai-worker".to_string(),
            enable_batching: true,
            max_batch_size: 32,
            batch_timeout_ms: 100,
            cache_dir: "/tmp/inference-cache".to_string(),
            enable_cache: true,
        }
    }
}

impl Config {
    /// Load configuration from environment variables and optional TOML file.
    ///
    /// Load order:
    /// 1. Default values
    /// 2. TOML file (if `MAIIA_AI_CONFIG_PATH` is set)
    /// 3. Environment variables with `MAIIA_AI_` prefix
    pub fn load() -> crate::Result<Self> {
        let mut config = Self::default();

        // Try to load from TOML file first
        if let Ok(config_path) = std::env::var("MAIIA_AI_CONFIG_PATH") {
            info!("Loading config from TOML file: {}", config_path);
            let toml_config = TomlConfig::from_file(&config_path)?;
            config.apply_toml(&toml_config);
        }

        // Override with environment variables
        config.apply_env();

        Ok(config)
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
        self.device.clone_from(&toml.inference.device);
        self.max_tokens = toml.inference.max_tokens;
        self.temperature = toml.inference.temperature;
        self.top_p = toml.inference.top_p;
        self.num_threads = toml.inference.num_threads;
        self.max_cache_length = toml.inference.max_cache_length;
        self.n_gpu_layers = toml.inference.n_gpu_layers;

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
        self.health_port = toml.service.health_port;
        self.service_name.clone_from(&toml.service.name);
        self.enable_batching = toml.service.enable_batching;
        self.max_batch_size = toml.service.max_batch_size;
        self.batch_timeout_ms = toml.service.batch_timeout_ms;
        self.cache_dir.clone_from(&toml.service.cache_dir);
        self.enable_cache = toml.service.enable_cache;
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
        }
        if let Some(v) = get_env("N_GPU_LAYERS").and_then(|s| s.parse().ok()) {
            self.n_gpu_layers = v;
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
    }

    /// Validate the configuration.
    pub fn validate(&self) -> crate::Result<()> {
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

    /// Generate task name from task type if not explicitly set.
    pub fn effective_task_name(&self) -> String {
        if self.task_name != "maiia.echo.v1" || self.task_type.is_echo() {
            self.task_name.clone()
        } else {
            format!("maiia.{}.v1", self.task_type)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
