//! Llama.cpp-based text generation for GGUF models.
//!
//! Uses `llama-cpp-2` (Rust bindings for llama.cpp) to load and run GGUF models
//! that cannot be run via ONNX or Candle (GPT-OSS-20B, custom architectures, etc.).
//!
//! ## Why llama.cpp?
//!
//! Some models have compatibility issues with other backends:
//! - **GPT-OSS-20B**: ONNX exports use custom `onnxruntime-genai` operators not available on macOS
//! - **Custom architectures**: Not supported by Candle's `candle-transformers`
//! - **MXFP4/exotic quantization**: Only supported in GGUF format
//!
//! llama.cpp supports:
//! - Metal acceleration on macOS
//! - CUDA acceleration on Linux/Windows
//! - Many quantization formats (Q4_K_M, Q8, F16, etc.)
//! - Most modern LLM architectures
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_llama::LlamaTextGenTask;
//! use inference_core::Config;
//!
//! // Config should have:
//! // - backend = "llama"
//! // - gguf_file = "model-Q4_K_M.gguf"
//! // - n_gpu_layers = 99  (for GPU acceleration)
//!
//! let task = LlamaTextGenTask::from_gguf_path("/path/to/model.gguf", "my-task", &config)?;
//! let result = task.execute(r#"{"text": "Hello"}"#, "req-1").await;
//! ```

use crate::error::{TaskError, TaskResult};
use async_trait::async_trait;
use inference_core::generation::{CacheDType, KvCacheConfig};
use inference_core::task::{Task, TaskChunk, TaskResult as GrpcTaskResult, TaskStream};
use inference_core::Config;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use tokio::sync::mpsc;
use tracing::{debug, info};

/// Generation configuration for llama.cpp text generation.
#[derive(Debug, Clone)]
pub struct LlamaGenConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, 0.0 = greedy)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: i32,
    /// Repetition penalty (1.0 = no penalty)
    pub repeat_penalty: f32,
    /// Number of GPU layers to offload (0 = CPU only)
    pub n_gpu_layers: u32,
    /// Context size (max sequence length)
    pub n_ctx: u32,
    /// KV cache configuration (dtype, flash attention, GPU offload).
    pub kv_cache: KvCacheConfig,
}

impl Default for LlamaGenConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            repeat_penalty: 1.1,
            n_gpu_layers: 0,
            n_ctx: 4096,
            kv_cache: KvCacheConfig::default(),
        }
    }
}

impl LlamaGenConfig {
    /// Create from inference config.
    pub fn from_config(config: &Config) -> Self {
        let kv_cache = KvCacheConfig::from_config(config);
        // Ensure n_ctx is never 0 — llama.cpp requires a positive context size.
        // Default to 4096 if kv_cache.max_length is 0.
        let n_ctx = if kv_cache.max_length == 0 {
            4096
        } else {
            kv_cache.max_length as u32
        };
        Self {
            max_new_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            n_gpu_layers: config.n_gpu_layers,
            n_ctx,
            kv_cache,
            ..Default::default()
        }
    }
}

/// Request to the llama worker thread.
enum LlamaRequest {
    /// Generate text (non-streaming)
    Generate {
        prompt: String,
        response_tx: tokio::sync::oneshot::Sender<TaskResult<(String, usize)>>,
    },
    /// Generate text with streaming
    GenerateStream {
        prompt: String,
        chunk_tx: mpsc::Sender<TaskChunk>,
    },
}

/// Signal from worker thread indicating model load result.
enum WorkerReady {
    Ok { vocab_size: i32, n_ctx_train: u32 },
    Failed(String),
}

/// Internal worker that owns all llama.cpp resources on a single thread.
/// This avoids all Send/Sync issues by keeping everything on one thread.
struct LlamaWorker {
    request_rx: std::sync::mpsc::Receiver<LlamaRequest>,
    ready_tx: std::sync::mpsc::Sender<WorkerReady>,
    gguf_path: PathBuf,
    gen_config: LlamaGenConfig,
}

/// Map our backend-agnostic `CacheDType` to llama.cpp's `KvCacheType`.
///
/// Only the types that are practical for KV cache quantization are mapped.
/// llama.cpp supports many more `ggml_type` variants but most are not useful
/// (or tested) for KV cache storage.
fn cache_dtype_to_llama(dtype: CacheDType) -> llama_cpp_2::context::params::KvCacheType {
    use llama_cpp_2::context::params::KvCacheType;
    match dtype {
        CacheDType::F32 => KvCacheType::F32,
        CacheDType::F16 => KvCacheType::F16,
        CacheDType::BF16 => KvCacheType::BF16,
        CacheDType::Q8_0 => KvCacheType::Q8_0,
        CacheDType::Q4_0 => KvCacheType::Q4_0,
        CacheDType::Q4_K => KvCacheType::Q4_K,
        CacheDType::Q5_K => KvCacheType::Q5_K,
        CacheDType::Q6_K => KvCacheType::Q6_K,
        CacheDType::Q8_K => KvCacheType::Q8_K,
        CacheDType::TQ1_0 => KvCacheType::TQ1_0,
        CacheDType::TQ2_0 => KvCacheType::TQ2_0,
        CacheDType::MXFP4 => KvCacheType::MXFP4,
    }
}

impl LlamaWorker {
    fn run(self) {
        use llama_cpp_2::context::params::LlamaContextParams;
        use llama_cpp_2::llama_backend::LlamaBackend;
        use llama_cpp_2::model::params::LlamaModelParams;
        use llama_cpp_2::model::LlamaModel;
        use std::num::NonZeroU32;

        // Initialize backend
        let backend = match LlamaBackend::init() {
            Ok(b) => b,
            Err(e) => {
                let msg = format!("Failed to initialize llama.cpp backend: {e}");
                tracing::error!("{msg}");
                let _ = self.ready_tx.send(WorkerReady::Failed(msg));
                return;
            }
        };

        // Route llama.cpp native C/C++ logs through Rust's tracing system instead
        // of stderr. This lets our EnvFilter control the level:
        // - At WARN (default): only llama.cpp errors/warnings appear (no spam)
        // - At DEBUG/INFO: full model loading details visible for debugging
        llama_cpp_2::send_logs_to_tracing(
            llama_cpp_2::LogOptions::default().with_logs_enabled(true),
        );

        // Set up model parameters
        let model_params =
            LlamaModelParams::default().with_n_gpu_layers(self.gen_config.n_gpu_layers);

        // Load the model
        let model = match LlamaModel::load_from_file(&backend, &self.gguf_path, &model_params) {
            Ok(m) => m,
            Err(e) => {
                // The llama-cpp-2 error is often "null result from llama cpp" which is
                // unhelpful. The real error was already logged by llama.cpp's native
                // logger (routed through tracing above), e.g.:
                //   "missing tensor 'blk.0.ssm_dt.bias'"
                //   "model is too large for available memory"
                let msg = format!("Failed to load GGUF model: {e}");
                tracing::error!("{msg}");
                let _ = self.ready_tx.send(WorkerReady::Failed(msg));
                return;
            }
        };

        let vocab_size = model.n_vocab();
        let n_ctx_train = model.n_ctx_train();

        info!(
            vocab_size = vocab_size,
            n_ctx_train = n_ctx_train,
            "Model loaded successfully in worker thread"
        );

        // Signal that the model is ready
        let _ = self.ready_tx.send(WorkerReady::Ok {
            vocab_size,
            n_ctx_train,
        });

        // Set up context parameters with KV cache configuration
        //
        // IMPORTANT: n_batch must be large enough to process the prompt in one go
        // (or be handled in chunks). We set it equal to n_ctx following the pattern
        // used in llama.cpp's official examples (openai_stream, tools, server).
        let n_ctx = self.gen_config.n_ctx;
        let mut ctx_params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(n_ctx);

        info!(
            n_ctx = n_ctx,
            n_batch = n_ctx,
            n_ctx_train = model.n_ctx_train(),
            "Context parameters configured"
        );

        // Apply KV cache dtype for K cache
        if let Some(dtype_k) = self.gen_config.kv_cache.cache_dtype_k {
            let kv_type = cache_dtype_to_llama(dtype_k);
            info!(dtype = %dtype_k, "Setting K cache type");
            ctx_params = ctx_params.with_type_k(kv_type);
        }

        // Apply KV cache dtype for V cache
        if let Some(dtype_v) = self.gen_config.kv_cache.cache_dtype_v {
            let kv_type = cache_dtype_to_llama(dtype_v);
            info!(dtype = %dtype_v, "Setting V cache type");
            ctx_params = ctx_params.with_type_v(kv_type);
        }

        // Apply flash attention policy
        // LLAMA_FLASH_ATTN_TYPE_ENABLED = 1 (from llama_cpp_sys_2)
        if self.gen_config.kv_cache.flash_attention {
            info!("Enabling flash attention");
            ctx_params = ctx_params.with_flash_attention_policy(1);
        }

        // Apply KV cache GPU offloading
        ctx_params = ctx_params.with_offload_kqv(self.gen_config.kv_cache.offload_to_gpu);

        // Create context
        let mut ctx = match model.new_context(&backend, ctx_params) {
            Ok(c) => c,
            Err(e) => {
                let msg = format!("Failed to create context: {e}");
                tracing::error!("{msg}");
                let _ = self.ready_tx.send(WorkerReady::Failed(msg));
                return;
            }
        };

        let eos_token = model.token_eos();

        // Process requests
        while let Ok(request) = self.request_rx.recv() {
            match request {
                LlamaRequest::Generate {
                    prompt,
                    response_tx,
                } => {
                    let result =
                        Self::generate_sync(&model, &mut ctx, &prompt, &self.gen_config, eos_token);
                    let _ = response_tx.send(result);
                }
                LlamaRequest::GenerateStream { prompt, chunk_tx } => {
                    Self::generate_stream_sync(
                        &model,
                        &mut ctx,
                        &prompt,
                        &self.gen_config,
                        eos_token,
                        &chunk_tx,
                    );
                }
            }
        }

        info!("Llama worker thread shutting down");
    }

    fn generate_sync(
        model: &llama_cpp_2::model::LlamaModel,
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        prompt: &str,
        gen_config: &LlamaGenConfig,
        eos_token: llama_cpp_2::token::LlamaToken,
    ) -> TaskResult<(String, usize)> {
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        // Create UTF-8 decoder for token decoding
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        let gen_start = std::time::Instant::now();

        // Tokenize input
        let tokens = model
            .str_to_token(prompt, AddBos::Always)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;

        let prompt_len = tokens.len();
        if tokens.is_empty() {
            return Err(TaskError::InvalidInput(
                "Empty input after tokenization".into(),
            ));
        }

        let n_ctx = ctx.n_ctx() as usize;

        // Verify KV cache can fit prompt + generated tokens
        let n_kv_req = prompt_len + gen_config.max_new_tokens;
        if n_kv_req > n_ctx {
            info!(
                prompt_len = prompt_len,
                max_new_tokens = gen_config.max_new_tokens,
                n_ctx = n_ctx,
                "Prompt + max_new_tokens exceeds n_ctx, generation will stop at context limit"
            );
        }

        debug!(prompt_len = prompt_len, n_ctx = n_ctx, "Tokenized prompt");

        // Clear the KV cache
        ctx.clear_kv_cache();

        // Create batch with capacity matching context size
        let batch_capacity = gen_config.n_ctx.max(512) as usize;
        let mut batch = LlamaBatch::new(batch_capacity, 1);

        // Add prompt tokens to batch
        let last_index = i32::try_from(prompt_len - 1)
            .map_err(|_| TaskError::Inference("Position overflow: prompt too long".to_string()))?;
        for (i, token) in (0_i32..).zip(tokens.iter()) {
            let is_last = i == last_index;
            batch
                .add(*token, i, &[0], is_last)
                .map_err(|e| TaskError::Inference(format!("Failed to add token to batch: {e}")))?;
        }

        // Decode the prompt (prefill phase)
        let prefill_start = std::time::Instant::now();
        ctx.decode(&mut batch)
            .map_err(|e| TaskError::Inference(format!("Prompt decode failed: {e}")))?;
        let prefill_ms = prefill_start.elapsed().as_millis();
        info!(
            prefill_ms = prefill_ms,
            prompt_tokens = prompt_len,
            "Prefill completed"
        );

        // Set up sampler
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(gen_config.temperature),
            LlamaSampler::top_k(gen_config.top_k),
            LlamaSampler::top_p(gen_config.top_p, 1),
            LlamaSampler::dist(rand::random()),
        ]);

        // Generation loop
        let decode_start = std::time::Instant::now();
        let mut generated_tokens = Vec::new();

        for (n_cur, _step) in (batch.n_tokens()..).zip(0..gen_config.max_new_tokens) {
            // Bail out if we've reached the context limit
            if n_cur as usize >= n_ctx {
                info!(n_cur = n_cur, n_ctx = n_ctx, "Reached context limit");
                break;
            }

            // Sample next token from the last logit position
            let new_token = sampler.sample(ctx, batch.n_tokens() - 1);

            // Accept the token to update sampler state (repetition penalty, etc.)
            sampler.accept(new_token);

            // Check for EOS
            if new_token == eos_token {
                debug!("EOS token generated");
                break;
            }

            generated_tokens.push(new_token);

            // Prepare batch for next token
            batch.clear();
            batch
                .add(new_token, n_cur, &[0], true)
                .map_err(|e| TaskError::Inference(format!("Failed to add token: {e}")))?;

            // Decode
            ctx.decode(&mut batch)
                .map_err(|e| TaskError::Inference(format!("Decode failed: {e}")))?;
        }

        let decode_ms = decode_start.elapsed().as_millis();
        let tokens_generated = generated_tokens.len();
        let tokens_per_sec = if decode_ms > 0 {
            (tokens_generated as f64 / decode_ms as f64) * 1000.0
        } else {
            0.0
        };
        info!(
            decode_ms = decode_ms,
            tokens = tokens_generated,
            tokens_per_sec = format!("{:.1}", tokens_per_sec),
            "Decode phase completed"
        );

        // Decode generated tokens to text using token_to_piece
        let mut generated_text = String::new();
        for token in &generated_tokens {
            let piece = model
                .token_to_piece(*token, &mut decoder, true, None)
                .map_err(|e| TaskError::Inference(format!("Token decode failed: {e}")))?;
            generated_text.push_str(&piece);
        }

        let total_ms = gen_start.elapsed().as_millis();
        info!(
            total_ms = total_ms,
            prefill_ms = prefill_ms,
            decode_ms = decode_ms,
            "Generation completed"
        );

        Ok((generated_text, generated_tokens.len()))
    }

    fn generate_stream_sync(
        model: &llama_cpp_2::model::LlamaModel,
        ctx: &mut llama_cpp_2::context::LlamaContext<'_>,
        prompt: &str,
        gen_config: &LlamaGenConfig,
        eos_token: llama_cpp_2::token::LlamaToken,
        chunk_tx: &mpsc::Sender<TaskChunk>,
    ) {
        use llama_cpp_2::llama_batch::LlamaBatch;
        use llama_cpp_2::model::AddBos;
        use llama_cpp_2::sampling::LlamaSampler;

        // Create UTF-8 decoder for token decoding
        let mut decoder = encoding_rs::UTF_8.new_decoder();

        // Tokenize input
        let tokens = match model.str_to_token(prompt, AddBos::Always) {
            Ok(t) => t,
            Err(e) => {
                let _ =
                    chunk_tx.blocking_send(TaskChunk::error(format!("Tokenization failed: {e}")));
                return;
            }
        };

        let prompt_len = tokens.len();
        if tokens.is_empty() {
            let _ = chunk_tx.blocking_send(TaskChunk::error(
                "Empty input after tokenization".to_string(),
            ));
            return;
        }

        let n_ctx = ctx.n_ctx() as usize;

        // Clear the KV cache
        ctx.clear_kv_cache();

        // Create batch with capacity matching context size
        let batch_capacity = gen_config.n_ctx.max(512) as usize;
        let mut batch = LlamaBatch::new(batch_capacity, 1);

        // Add prompt tokens to batch
        let Ok(last_index) = i32::try_from(prompt_len - 1) else {
            let _ = chunk_tx.blocking_send(TaskChunk::error("Position overflow".to_string()));
            return;
        };
        for (i, token) in (0_i32..).zip(tokens.iter()) {
            let is_last = i == last_index;
            if let Err(e) = batch.add(*token, i, &[0], is_last) {
                let _ = chunk_tx.blocking_send(TaskChunk::error(format!(
                    "Failed to add token to batch: {e}"
                )));
                return;
            }
        }

        // Decode the prompt
        if let Err(e) = ctx.decode(&mut batch) {
            let _ = chunk_tx.blocking_send(TaskChunk::error(format!("Prompt decode failed: {e}")));
            return;
        }

        // Set up sampler
        let mut sampler = LlamaSampler::chain_simple([
            LlamaSampler::temp(gen_config.temperature),
            LlamaSampler::top_k(gen_config.top_k),
            LlamaSampler::top_p(gen_config.top_p, 1),
            LlamaSampler::dist(rand::random()),
        ]);

        // Generation loop with streaming
        let mut generated_text = String::new();
        let mut num_tokens = 0;

        for (n_cur, _step) in (batch.n_tokens()..).zip(0..gen_config.max_new_tokens) {
            // Bail out if we've reached the context limit
            if n_cur as usize >= n_ctx {
                break;
            }

            // Sample next token from the last logit position
            let new_token = sampler.sample(ctx, batch.n_tokens() - 1);

            // Accept the token to update sampler state
            sampler.accept(new_token);

            // Check for EOS
            if new_token == eos_token {
                break;
            }

            // Decode token to text and send chunk
            let piece = model
                .token_to_piece(new_token, &mut decoder, true, None)
                .unwrap_or_default();

            generated_text.push_str(&piece);
            num_tokens += 1;

            let chunk_data = serde_json::json!({
                "token": piece,
                "token_id": new_token.0,
            });

            // Send the chunk - if the receiver is gone, stop generating
            if chunk_tx
                .blocking_send(TaskChunk::data(chunk_data.to_string()))
                .is_err()
            {
                return;
            }

            // Prepare batch for next token
            batch.clear();
            if batch.add(new_token, n_cur, &[0], true).is_err() {
                break;
            }

            // Decode
            if ctx.decode(&mut batch).is_err() {
                break;
            }
        }

        // Send final chunk
        let final_data = serde_json::json!({
            "text": generated_text,
            "num_tokens": num_tokens,
        });
        let _ = chunk_tx.blocking_send(TaskChunk::final_data(final_data.to_string()));
    }
}

/// Llama.cpp-based text generation task.
///
/// Loads GGUF models via llama.cpp for inference.
/// Supports GPU acceleration (Metal on macOS, CUDA on Linux).
///
/// Uses a dedicated worker thread for all llama.cpp operations to avoid
/// Send/Sync issues with llama.cpp types.
pub struct LlamaTextGenTask {
    name: String,
    request_tx: std::sync::mpsc::Sender<LlamaRequest>,
    gen_config: LlamaGenConfig,
    should_unload: std::sync::Arc<std::sync::Mutex<bool>>,
}

impl LlamaTextGenTask {
    /// Create a new llama text generation task from a GGUF model path.
    ///
    /// # Arguments
    /// * `gguf_path` - Path to the .gguf model file
    /// * `name` - Task name
    /// * `config` - Inference configuration
    pub fn from_gguf_path(
        gguf_path: impl Into<PathBuf>,
        name: impl Into<String>,
        config: &Config,
    ) -> TaskResult<Self> {
        let gguf_path = gguf_path.into();
        let name = name.into();
        let gen_config = LlamaGenConfig::from_config(config);

        info!(
            path = %gguf_path.display(),
            task_name = %name,
            n_gpu_layers = gen_config.n_gpu_layers,
            n_ctx = gen_config.n_ctx,
            "Loading GGUF model via llama.cpp"
        );

        // Verify the file exists
        if !gguf_path.exists() {
            return Err(TaskError::ModelLoad(format!(
                "GGUF file not found: {}",
                gguf_path.display()
            )));
        }

        // Create channel for communication with worker
        let (request_tx, request_rx) = std::sync::mpsc::channel();
        let (ready_tx, ready_rx) = std::sync::mpsc::channel();

        // Create worker
        let worker = LlamaWorker {
            request_rx,
            ready_tx,
            gguf_path: gguf_path.clone(),
            gen_config: gen_config.clone(),
        };

        // Spawn worker thread
        std::thread::Builder::new()
            .name(format!("llama-worker-{name}"))
            .spawn(move || {
                worker.run();
            })
            .map_err(|e| TaskError::ModelLoad(format!("Failed to spawn worker thread: {e}")))?;

        // Wait for the worker to signal that the model is loaded (or failed).
        // This blocks the calling thread (typically during eager startup) but
        // ensures we don't report "model loaded" until it's actually true.
        match ready_rx.recv() {
            Ok(WorkerReady::Ok {
                vocab_size,
                n_ctx_train,
            }) => {
                info!(
                    vocab_size = vocab_size,
                    n_ctx_train = n_ctx_train,
                    "Model loaded and ready"
                );
            }
            Ok(WorkerReady::Failed(msg)) => {
                return Err(TaskError::ModelLoad(msg));
            }
            Err(_) => {
                return Err(TaskError::ModelLoad(
                    "Worker thread disconnected before model was loaded".to_string(),
                ));
            }
        }

        Ok(Self {
            name,
            request_tx,
            gen_config,
            should_unload: std::sync::Arc::new(std::sync::Mutex::new(false)),
        })
    }

    /// Generate text from a prompt.
    async fn generate(&self, prompt: &str) -> TaskResult<(String, usize)> {
        let (response_tx, response_rx) = tokio::sync::oneshot::channel();

        self.request_tx
            .send(LlamaRequest::Generate {
                prompt: prompt.to_string(),
                response_tx,
            })
            .map_err(|_| TaskError::Inference("Worker thread not available".to_string()))?;

        response_rx
            .await
            .map_err(|_| TaskError::Inference("Worker thread disconnected".to_string()))?
    }

    /// Generate text with streaming output.
    fn generate_stream(&self, prompt: String) -> TaskStream {
        let (chunk_tx, mut chunk_rx) = mpsc::channel(32);

        // Send request to worker
        if self
            .request_tx
            .send(LlamaRequest::GenerateStream { prompt, chunk_tx })
            .is_err()
        {
            return Box::pin(tokio_stream::once(TaskChunk::error(
                "Worker thread not available".to_string(),
            )));
        }

        // Convert the mpsc receiver to a stream
        let stream = async_stream::stream! {
            while let Some(chunk) = chunk_rx.recv().await {
                yield chunk;
            }
        };

        Box::pin(stream)
    }
}

/// Input for llama text generation.
#[derive(Debug, Deserialize)]
pub struct LlamaTextGenInput {
    /// Input text/prompt
    pub text: String,
    /// Optional: maximum tokens to generate (overrides config)
    pub max_new_tokens: Option<usize>,
    /// Optional: temperature (overrides config)
    pub temperature: Option<f32>,
    /// Optional: top_p (overrides config)
    pub top_p: Option<f32>,
    /// Optional: stream output token by token
    pub stream: Option<bool>,
}

/// Output from llama text generation.
#[derive(Debug, Serialize)]
pub struct LlamaTextGenOutput {
    /// Generated text (continuation of input)
    pub text: String,
    /// Number of tokens generated
    pub num_tokens: usize,
}

#[async_trait]
impl Task for LlamaTextGenTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "Llama text-gen task executing");

        // Parse input
        let input: LlamaTextGenInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input: {e}")),
        };

        if input.text.is_empty() {
            return GrpcTaskResult::err("Empty input text".into());
        }

        // Generate
        match self.generate(&input.text).await {
            Ok((text, num_tokens)) => {
                let output = LlamaTextGenOutput { text, num_tokens };
                match serde_json::to_string(&output) {
                    Ok(json) => GrpcTaskResult::ok(json),
                    Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
                }
            }
            Err(e) => GrpcTaskResult::err(format!("Generation failed: {e}")),
        }
    }

    async fn execute_stream(&self, payload: String, request_id: String) -> TaskStream {
        debug!(request_id = request_id, "Llama text-gen streaming");

        // Parse input
        let input: LlamaTextGenInput = match serde_json::from_str(&payload) {
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
        self.generate_stream(input.text)
    }

    fn supports_streaming(&self) -> bool {
        true
    }

    fn is_ready(&self) -> bool {
        // Optimistically return true - the worker will report errors
        true
    }

    fn should_unload(&self) -> bool {
        *self.should_unload.lock().unwrap()
    }
}

impl std::fmt::Debug for LlamaTextGenTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaTextGenTask")
            .field("name", &self.name)
            .field("gen_config", &self.gen_config)
            .finish_non_exhaustive()
    }
}
