//! Candle-based text generation for modern LLMs.
//!
//! Uses `candle` (Rust-native ML framework from HuggingFace) to load and run
//! modern LLMs that don't have ONNX exports (Qwen3, Llama 3.x, Mistral, etc.).
//!
//! ## Why Candle instead of ONNX?
//!
//! Most modern LLMs (released 2024+) don't have official ONNX exports:
//! - **Qwen3**: No ONNX, only safetensors
//! - **Llama 3.x**: No official ONNX (community exports are often outdated)
//! - **Mistral v0.3+**: No ONNX
//! - **DeepSeek-R1**: No ONNX
//!
//! Candle loads `.safetensors` files directly from HuggingFace and supports:
//! - Metal acceleration on macOS
//! - CUDA acceleration on Linux
//! - Efficient autoregressive generation with KV-cache
//!
//! ## Supported Model Architectures
//!
//! - **Qwen2/Qwen3**: `candle_transformers::models::qwen2`
//! - **Llama 2/3**: `candle_transformers::models::llama`
//! - **Mistral**: `candle_transformers::models::mistral`
//! - **Phi-3**: `candle_transformers::models::phi3`
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_candle::CandleTextGenTask;
//! use inference_core::Config;
//!
//! let config = Config::load()?;
//! let task = CandleTextGenTask::from_config(&config).await?;
//!
//! // Input: {"text": "Hello, world!"}
//! let result = task.execute(r#"{"text": "Hello"}"#, "req-1").await;
//! ```

use crate::error::{TaskError, TaskResult};
use crate::utils;
use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::llama as llama_model;
use inference_core::task::{Task, TaskChunk, TaskResult as GrpcTaskResult, TaskStream};
use inference_core::{Config, KvCacheConfig};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Model architecture for text generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TextGenModelArch {
    /// Qwen2/Qwen2.5/Qwen3 family
    Qwen2,
    /// Llama 2/3 family
    Llama,
    /// Mistral family
    Mistral,
    /// Phi-3 family
    Phi3,
}

impl TextGenModelArch {
    /// Auto-detect architecture from config.json in model directory.
    pub fn detect_from_config(config_path: &Path) -> TaskResult<Self> {
        let content = std::fs::read_to_string(config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;

        let config: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        // Check model_type field
        let model_type = config
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Check architectures field
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
            "Detecting model architecture"
        );

        // Match based on model_type or architectures
        let arch_str = format!("{model_type} {architectures}").to_lowercase();

        if arch_str.contains("qwen") {
            Ok(Self::Qwen2)
        } else if arch_str.contains("llama") {
            Ok(Self::Llama)
        } else if arch_str.contains("mistral") {
            Ok(Self::Mistral)
        } else if arch_str.contains("phi") {
            Ok(Self::Phi3)
        } else {
            Err(TaskError::ModelLoad(format!(
                "Unknown model architecture: model_type={model_type}, architectures={architectures}"
            )))
        }
    }
}

/// Generation configuration for candle text generation.
#[derive(Debug, Clone)]
pub struct CandleGenConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change)
    pub temperature: f64,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f64,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Context length for repetition penalty
    pub repeat_last_n: usize,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
    /// KV cache configuration (max length, dtype preferences, etc.)
    pub kv_cache: KvCacheConfig,
}

impl Default for CandleGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: None,
            kv_cache: KvCacheConfig::default(),
        }
    }
}

impl CandleGenConfig {
    /// Create from inference config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_new_tokens: config.max_tokens,
            temperature: f64::from(config.temperature),
            top_p: f64::from(config.top_p),
            kv_cache: KvCacheConfig::from_config(config),
            ..Default::default()
        }
    }
}

/// Internal model wrapper to handle different architectures uniformly.
/// Each architecture has its own cache type, managed alongside the model.
enum CandleModel {
    Qwen2(candle_transformers::models::qwen2::ModelForCausalLM),
    Llama {
        model: llama_model::Llama,
        cache: llama_model::Cache,
        config: llama_model::Config,
    },
    Mistral(candle_transformers::models::mistral::Model),
    Phi3(candle_transformers::models::phi3::Model),
}

impl CandleModel {
    /// Forward pass through the model.
    fn forward(&mut self, input_ids: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        match self {
            Self::Qwen2(m) => m.forward(input_ids, start_pos),
            Self::Llama { model, cache, .. } => model.forward(input_ids, start_pos, cache),
            Self::Mistral(m) => m.forward(input_ids, start_pos),
            Self::Phi3(m) => m.forward(input_ids, start_pos),
        }
    }

    /// Clear the KV-cache for a new generation.
    /// For Llama, we recreate the cache since it's managed externally.
    fn clear_kv_cache(&mut self, device: &Device, dtype: DType) -> candle_core::Result<()> {
        match self {
            Self::Qwen2(m) => {
                m.clear_kv_cache();
                Ok(())
            }
            Self::Llama { cache, config, .. } => {
                // Recreate cache for new generation
                *cache = llama_model::Cache::new(true, dtype, config, device)?;
                Ok(())
            }
            Self::Mistral(m) => {
                m.clear_kv_cache();
                Ok(())
            }
            Self::Phi3(m) => {
                m.clear_kv_cache();
                Ok(())
            }
        }
    }
}

/// Candle-based text generation task.
///
/// Loads modern LLMs directly from safetensors files (no ONNX needed).
/// Supports Qwen2/3, Llama, Mistral, and Phi-3 model families.
pub struct CandleTextGenTask {
    name: String,
    model: Arc<Mutex<CandleModel>>,
    tokenizer: Tokenizer,
    device: Device,
    dtype: DType,
    gen_config: CandleGenConfig,
    eos_token_ids: Vec<u32>,
}

impl CandleTextGenTask {
    /// Create a new candle text generation task from a model directory.
    ///
    /// The model directory should contain:
    /// - `config.json` - Model configuration (architecture, vocab size, etc.)
    /// - `*.safetensors` - Model weights
    /// - `tokenizer.json` - Tokenizer configuration
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
            "Loading candle text-generation model"
        );

        // Resolve device
        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Detect architecture from config.json
        let config_path = model_dir.join("config.json");
        let arch = TextGenModelArch::detect_from_config(&config_path)?;
        info!(architecture = ?arch, "Detected model architecture");

        // Find safetensors files
        let safetensors_files = utils::find_weight_files(model_dir)?;
        if safetensors_files.is_empty() {
            return Err(TaskError::ModelNotFound(
                "No .safetensors files found".into(),
            ));
        }
        info!(
            num_files = safetensors_files.len(),
            "Found safetensors files"
        );

        // Load tokenizer
        let tokenizer = utils::load_tokenizer(model_dir)?;
        info!("Tokenizer loaded");

        // Build generation config first — we need kv_cache settings to pick dtype
        let gen_config = CandleGenConfig::from_config(config);

        // Calculate total weight size for dtype safety decisions
        let weight_size_bytes: u64 = safetensors_files
            .iter()
            .filter_map(|p| std::fs::metadata(p).ok().map(|m| m.len()))
            .sum();

        // Determine compute dtype from KV cache config + model's native torch_dtype.
        // In Candle, KV cache dtype = model compute dtype (they can't differ).
        // Using the model's native dtype (e.g., BF16 for Qwen2.5) instead of F32
        // halves memory and is faster, especially on CPU.
        let model_dtype = utils::read_model_dtype(&config_path);
        let dtype = utils::resolve_compute_dtype(
            &gen_config.kv_cache,
            model_dtype,
            &device,
            weight_size_bytes,
        );
        info!(dtype = ?dtype, weight_mb = weight_size_bytes / (1024 * 1024), "Compute dtype (controls weights + KV cache)");

        // Load model weights using safe (non-mmap) loading
        let vb = utils::load_safetensors_safe(&safetensors_files, dtype, &device)?;

        // Load model based on architecture
        let model = Self::load_model(arch, &config_path, vb, &device, dtype)?;
        info!(architecture = ?arch, "Model loaded successfully");

        // Get EOS token IDs
        let eos_token_ids = Self::get_eos_token_ids(&tokenizer, &config_path);
        info!(eos_tokens = ?eos_token_ids, "EOS token IDs");

        info!(
            max_cache_length = gen_config.kv_cache.max_length,
            flash_attention = gen_config.kv_cache.flash_attention,
            cache_dtype_k = ?gen_config.kv_cache.cache_dtype_k,
            cache_dtype_v = ?gen_config.kv_cache.cache_dtype_v,
            "KV cache config"
        );

        Ok(Self {
            name,
            model: Arc::new(Mutex::new(model)),
            tokenizer,
            device,
            dtype,
            gen_config,
            eos_token_ids,
        })
    }

    /// Load model based on detected architecture.
    fn load_model(
        arch: TextGenModelArch,
        config_path: &Path,
        vb: VarBuilder,
        device: &Device,
        dtype: DType,
    ) -> TaskResult<CandleModel> {
        let config_str = std::fs::read_to_string(config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config: {e}")))?;

        match arch {
            TextGenModelArch::Qwen2 => {
                let config: candle_transformers::models::qwen2::Config =
                    serde_json::from_str(&config_str)
                        .map_err(|e| TaskError::ModelLoad(format!("Invalid Qwen2 config: {e}")))?;
                let model = candle_transformers::models::qwen2::ModelForCausalLM::new(&config, vb)
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to load Qwen2: {e}")))?;
                Ok(CandleModel::Qwen2(model))
            }
            TextGenModelArch::Llama => {
                // Llama requires LlamaConfig -> Config conversion and external Cache
                let llama_config: llama_model::LlamaConfig = serde_json::from_str(&config_str)
                    .map_err(|e| TaskError::ModelLoad(format!("Invalid Llama config: {e}")))?;
                // Convert LlamaConfig to Config (use_flash_attn = false for compatibility)
                let config = llama_config.into_config(false);
                // Create the cache for KV-cache management
                let cache = llama_model::Cache::new(true, dtype, &config, device).map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to create Llama cache: {e}"))
                })?;
                let model = llama_model::Llama::load(vb, &config)
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to load Llama: {e}")))?;
                Ok(CandleModel::Llama {
                    model,
                    cache,
                    config,
                })
            }
            TextGenModelArch::Mistral => {
                let config: candle_transformers::models::mistral::Config =
                    serde_json::from_str(&config_str).map_err(|e| {
                        TaskError::ModelLoad(format!("Invalid Mistral config: {e}"))
                    })?;
                let model = candle_transformers::models::mistral::Model::new(&config, vb)
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to load Mistral: {e}")))?;
                Ok(CandleModel::Mistral(model))
            }
            TextGenModelArch::Phi3 => {
                let config: candle_transformers::models::phi3::Config =
                    serde_json::from_str(&config_str)
                        .map_err(|e| TaskError::ModelLoad(format!("Invalid Phi3 config: {e}")))?;
                let model = candle_transformers::models::phi3::Model::new(&config, vb)
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to load Phi3: {e}")))?;
                Ok(CandleModel::Phi3(model))
            }
        }
    }

    /// Get EOS token IDs from tokenizer and config.
    fn get_eos_token_ids(tokenizer: &Tokenizer, config_path: &Path) -> Vec<u32> {
        let mut eos_ids = Vec::new();

        // Try to get from tokenizer
        if let Some(eos) = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .get("<|endoftext|>")
        {
            eos_ids.push(*eos);
        }
        if let Some(eos) = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .get("<|im_end|>")
        {
            eos_ids.push(*eos);
        }
        if let Some(eos) = tokenizer.get_added_vocabulary().get_vocab().get("</s>") {
            eos_ids.push(*eos);
        }
        if let Some(eos) = tokenizer
            .get_added_vocabulary()
            .get_vocab()
            .get("<|eot_id|>")
        {
            eos_ids.push(*eos);
        }

        // Try to get from config.json
        if let Ok(content) = std::fs::read_to_string(config_path) {
            if let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) {
                if let Some(eos_id) = config
                    .get("eos_token_id")
                    .and_then(serde_json::Value::as_u64)
                {
                    #[allow(clippy::cast_possible_truncation)]
                    eos_ids.push(eos_id as u32);
                }
                // Some models have array of EOS tokens
                if let Some(eos_arr) = config.get("eos_token_id").and_then(|v| v.as_array()) {
                    for id in eos_arr.iter().filter_map(serde_json::Value::as_u64) {
                        #[allow(clippy::cast_possible_truncation)]
                        eos_ids.push(id as u32);
                    }
                }
            }
        }

        // Deduplicate
        eos_ids.sort_unstable();
        eos_ids.dedup();

        // Default fallback
        if eos_ids.is_empty() {
            eos_ids.push(2); // Common EOS token
        }

        eos_ids
    }

    /// Generate text from input tokens.
    async fn generate(&self, prompt: &str) -> TaskResult<(String, usize)> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = input_ids.len();

        if input_ids.is_empty() {
            return Err(TaskError::InvalidInput(
                "Empty input after tokenization".into(),
            ));
        }

        // Create tensor
        let input_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .map_err(|e| TaskError::Inference(format!("Failed to create input tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("Failed to unsqueeze: {e}")))?;

        // Create logits processor for sampling
        let seed = self.gen_config.seed.unwrap_or_else(rand::random);
        let mut logits_processor = LogitsProcessor::new(
            seed,
            Some(self.gen_config.temperature),
            Some(self.gen_config.top_p),
        );

        // Generation loop
        let mut model = self.model.lock().await;
        model
            .clear_kv_cache(&self.device, self.dtype)
            .map_err(|e| TaskError::Inference(format!("Failed to clear KV cache: {e}")))?;

        let mut generated_tokens: Vec<u32> = Vec::new();
        let mut all_tokens = input_ids.clone();
        let mut current_input = input_tensor;

        for step in 0..self.gen_config.max_new_tokens {
            // Check if we've hit the KV cache length limit.
            // Total cached positions = prompt tokens + generated tokens so far.
            let total_seq_len = prompt_len + step;
            if total_seq_len >= self.gen_config.kv_cache.max_length {
                debug!(
                    total_seq_len = total_seq_len,
                    max_cache = self.gen_config.kv_cache.max_length,
                    "Stopping generation: KV cache length limit reached"
                );
                break;
            }

            // Forward pass
            // start_pos is where this sequence begins in the KV-cache:
            // - First forward (full prompt): start_pos = 0
            // - Subsequent forwards (single token): start_pos = prompt_len + tokens_generated_so_far
            let start_pos = if step == 0 { 0 } else { prompt_len + step - 1 };
            let logits = model
                .forward(&current_input, start_pos)
                .map_err(|e| TaskError::Inference(format!("Forward pass failed: {e}")))?;

            // Get logits for last position
            let logits = logits
                .squeeze(0)
                .map_err(|e| TaskError::Inference(format!("Squeeze failed: {e}")))?;
            let logits = logits
                .get(logits.dim(0).unwrap_or(1) - 1)
                .map_err(|e| TaskError::Inference(format!("Get last logits failed: {e}")))?;

            // Apply repetition penalty
            #[allow(clippy::float_cmp)]
            let logits = if self.gen_config.repeat_penalty == 1.0 {
                logits
            } else {
                let start_at = all_tokens
                    .len()
                    .saturating_sub(self.gen_config.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.gen_config.repeat_penalty,
                    &all_tokens[start_at..],
                )
                .map_err(|e| TaskError::Inference(format!("Repeat penalty failed: {e}")))?
            };

            // Ensure logits are F32 for numerically stable sampling.
            // F16 logits from large models can overflow (max ~65504) causing NaN.
            let logits = if logits.dtype() == candle_core::DType::F32 {
                logits
            } else {
                logits
                    .to_dtype(candle_core::DType::F32)
                    .map_err(|e| TaskError::Inference(format!("Logits F32 cast failed: {e}")))?
            };

            // Detect NaN/Inf in logits on first step — gives a clear error instead of
            // cryptic "weight is negative" from the sampler. Only checked once since
            // if the first forward pass produces NaN, all subsequent ones will too.
            if step == 0 {
                if let Ok(vals) = logits.to_vec1::<f32>() {
                    if vals.iter().any(|v| v.is_nan() || v.is_infinite()) {
                        return Err(TaskError::Inference(
                            "Model produced NaN/Inf logits — numerical overflow in forward pass. \
                             Try using a GGUF quantized model with the llama backend for \
                             reliable CPU inference of large models."
                                .into(),
                        ));
                    }
                }
            }

            // Sample next token
            let next_token = logits_processor
                .sample(&logits)
                .map_err(|e| TaskError::Inference(format!("Sampling failed: {e}")))?;

            // Check for EOS
            if self.eos_token_ids.contains(&next_token) {
                debug!(step = step, token = next_token, "EOS token generated");
                break;
            }

            generated_tokens.push(next_token);
            all_tokens.push(next_token);

            // Prepare next input (just the new token, KV-cache handles context)
            current_input = Tensor::new(&[next_token], &self.device)
                .map_err(|e| TaskError::Inference(format!("Failed to create token tensor: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TaskError::Inference(format!("Failed to unsqueeze token: {e}")))?;

            if step > 0 && step % 50 == 0 {
                debug!(
                    step = step,
                    tokens = generated_tokens.len(),
                    "Generation progress"
                );
            }
        }

        // Decode generated tokens
        let generated_text = self
            .tokenizer
            .decode(&generated_tokens, true)
            .map_err(|e| TaskError::Inference(format!("Decoding failed: {e}")))?;

        Ok((generated_text, generated_tokens.len()))
    }

    /// Generate text with streaming output.
    fn generate_stream(&self, prompt: &str) -> TaskResult<impl futures::Stream<Item = TaskChunk>> {
        // Tokenize input
        let encoding = self
            .tokenizer
            .encode(prompt, true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;

        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let prompt_len = input_ids.len();

        if input_ids.is_empty() {
            return Err(TaskError::InvalidInput(
                "Empty input after tokenization".into(),
            ));
        }

        // Clone what we need for the async stream
        let model = Arc::clone(&self.model);
        let device = self.device.clone();
        let dtype = self.dtype;
        let gen_config = self.gen_config.clone();
        let eos_token_ids = self.eos_token_ids.clone();
        let tokenizer = self.tokenizer.clone();

        let stream = async_stream::stream! {
            // Create tensor
            let input_tensor = match Tensor::new(input_ids.as_slice(), &device) {
                Ok(t) => match t.unsqueeze(0) {
                    Ok(t) => t,
                    Err(e) => {
                        yield TaskChunk::error(format!("Tensor error: {e}"));
                        return;
                    }
                },
                Err(e) => {
                    yield TaskChunk::error(format!("Tensor error: {e}"));
                    return;
                }
            };

            // Create logits processor
            let seed = gen_config.seed.unwrap_or_else(rand::random);
            let mut logits_processor = LogitsProcessor::new(
                seed,
                Some(gen_config.temperature),
                Some(gen_config.top_p),
            );

            let mut model_guard = model.lock().await;
            if let Err(e) = model_guard.clear_kv_cache(&device, dtype) {
                yield TaskChunk::error(format!("Failed to clear KV cache: {e}"));
                return;
            }

            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut all_tokens = input_ids.clone();
            let mut current_input = input_tensor;

            for step in 0..gen_config.max_new_tokens {
                // Check if we've hit the KV cache length limit.
                let total_seq_len = prompt_len + step;
                if total_seq_len >= gen_config.kv_cache.max_length {
                    debug!(
                        total_seq_len = total_seq_len,
                        max_cache = gen_config.kv_cache.max_length,
                        "Stopping generation: KV cache length limit reached"
                    );
                    break;
                }

                // Forward pass
                // start_pos is where this sequence begins in the KV-cache:
                // - First forward (full prompt): start_pos = 0
                // - Subsequent forwards (single token): start_pos = prompt_len + tokens_generated_so_far
                let start_pos = if step == 0 { 0 } else { prompt_len + step - 1 };
                let logits = match model_guard.forward(&current_input, start_pos) {
                    Ok(l) => l,
                    Err(e) => {
                        yield TaskChunk::error(format!("Forward failed: {e}"));
                        return;
                    }
                };

                // Get logits for last position
                let logits = match logits.squeeze(0) {
                    Ok(l) => l,
                    Err(e) => {
                        yield TaskChunk::error(format!("Squeeze failed: {e}"));
                        return;
                    }
                };
                let dim = logits.dim(0).unwrap_or(1);
                let logits = match logits.get(dim - 1) {
                    Ok(l) => l,
                    Err(e) => {
                        yield TaskChunk::error(format!("Get logits failed: {e}"));
                        return;
                    }
                };

                // Apply repetition penalty
                #[allow(clippy::float_cmp)]
                let logits = if gen_config.repeat_penalty == 1.0 {
                    logits
                } else {
                    let start_at = all_tokens.len().saturating_sub(gen_config.repeat_last_n);
                    match candle_transformers::utils::apply_repeat_penalty(
                        &logits,
                        gen_config.repeat_penalty,
                        &all_tokens[start_at..],
                    ) {
                        Ok(l) => l,
                        Err(e) => {
                            yield TaskChunk::error(format!("Repeat penalty failed: {e}"));
                            return;
                        }
                    }
                };

                // Ensure logits are F32 for numerically stable sampling.
                let logits = match logits.dtype() {
                    candle_core::DType::F32 => logits,
                    _ => match logits.to_dtype(candle_core::DType::F32) {
                        Ok(l) => l,
                        Err(e) => {
                            yield TaskChunk::error(format!("Logits F32 cast failed: {e}"));
                            return;
                        }
                    },
                };

                // Detect NaN/Inf in logits on first step
                if step == 0 {
                    if let Ok(vals) = logits.to_vec1::<f32>() {
                        if vals.iter().any(|v| v.is_nan() || v.is_infinite()) {
                            yield TaskChunk::error(
                                "Model produced NaN/Inf logits — numerical overflow in forward pass. \
                                 Try using a GGUF quantized model with the llama backend."
                                    .to_string(),
                            );
                            return;
                        }
                    }
                }

                // Sample next token
                let next_token = match logits_processor.sample(&logits) {
                    Ok(t) => t,
                    Err(e) => {
                        yield TaskChunk::error(format!("Sampling failed: {e}"));
                        return;
                    }
                };

                // Check for EOS
                if eos_token_ids.contains(&next_token) {
                    break;
                }

                generated_tokens.push(next_token);
                all_tokens.push(next_token);

                // Decode and stream the token
                if let Ok(token_text) = tokenizer.decode(&[next_token], false) {
                    let chunk_data = serde_json::json!({
                        "token": token_text,
                        "token_id": next_token,
                    });
                    yield TaskChunk::data(chunk_data.to_string());
                }

                // Prepare next input
                current_input = match Tensor::new(&[next_token], &device) {
                    Ok(t) => match t.unsqueeze(0) {
                        Ok(t) => t,
                        Err(e) => {
                            yield TaskChunk::error(format!("Tensor error: {e}"));
                            return;
                        }
                    },
                    Err(e) => {
                        yield TaskChunk::error(format!("Tensor error: {e}"));
                        return;
                    }
                };
            }

            // Final chunk with complete output
            let full_text = tokenizer.decode(&generated_tokens, true).unwrap_or_default();
            let final_data = serde_json::json!({
                "text": full_text,
                "num_tokens": generated_tokens.len(),
            });
            yield TaskChunk::final_data(final_data.to_string());
        };

        Ok(stream)
    }
}

/// Input for candle text generation.
#[derive(Debug, Deserialize)]
pub struct CandleTextGenInput {
    /// Input text/prompt
    pub text: String,
    /// Optional: maximum tokens to generate (overrides config)
    pub max_new_tokens: Option<usize>,
    /// Optional: temperature (overrides config)
    pub temperature: Option<f64>,
    /// Optional: top_p (overrides config)
    pub top_p: Option<f64>,
    /// Optional: stream output token by token
    pub stream: Option<bool>,
}

/// Output from candle text generation.
#[derive(Debug, Serialize)]
pub struct CandleTextGenOutput {
    /// Generated text (continuation of input)
    pub text: String,
    /// Number of tokens generated
    pub num_tokens: usize,
}

#[async_trait]
impl Task for CandleTextGenTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "Candle text-gen task executing");

        // Parse input
        let input: CandleTextGenInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input: {e}")),
        };

        if input.text.is_empty() {
            return GrpcTaskResult::err("Empty input text".into());
        }

        // Generate
        match self.generate(&input.text).await {
            Ok((text, num_tokens)) => {
                let output = CandleTextGenOutput { text, num_tokens };
                match serde_json::to_string(&output) {
                    Ok(json) => GrpcTaskResult::ok(json),
                    Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
                }
            }
            Err(e) => GrpcTaskResult::err(format!("Generation failed: {e}")),
        }
    }

    async fn execute_stream(&self, payload: &str, request_id: &str) -> TaskStream {
        debug!(request_id = request_id, "Candle text-gen streaming");

        // Parse input
        let input: CandleTextGenInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => {
                return Box::pin(tokio_stream::once(TaskChunk::error(format!(
                    "Invalid input: {e}"
                ))));
            }
        };

        if input.text.is_empty() {
            return Box::pin(tokio_stream::once(TaskChunk::error(
                "Empty input text".into(),
            )));
        }

        // Generate with streaming
        match self.generate_stream(&input.text) {
            Ok(stream) => Box::pin(stream),
            Err(e) => Box::pin(tokio_stream::once(TaskChunk::error(format!(
                "Generation failed: {e}"
            )))),
        }
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn is_ready(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for CandleTextGenTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CandleTextGenTask")
            .field("name", &self.name)
            .field("device", &self.device)
            .field("eos_token_ids", &self.eos_token_ids)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gen_config_defaults() {
        let config = CandleGenConfig::default();
        assert_eq!(config.max_new_tokens, 256);
        assert!((config.temperature - 0.7).abs() < f64::EPSILON);
        assert!((config.top_p - 0.9).abs() < f64::EPSILON);
        assert_eq!(config.kv_cache.max_length, 2048);
        assert!(!config.kv_cache.flash_attention);
    }

    #[test]
    fn test_arch_detection_qwen() {
        // Test that qwen keywords are detected
        let test_cases = ["qwen2", "Qwen2ForCausalLM", "qwen3"];
        for case in test_cases {
            let lower = case.to_lowercase();
            assert!(lower.contains("qwen"), "Should detect qwen in {case}");
        }
    }

    #[test]
    fn test_arch_detection_llama() {
        let test_cases = ["llama", "LlamaForCausalLM", "llama3"];
        for case in test_cases {
            let lower = case.to_lowercase();
            assert!(lower.contains("llama"), "Should detect llama in {case}");
        }
    }
}
