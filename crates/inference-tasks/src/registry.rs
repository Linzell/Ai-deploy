//! Task registry - creates tasks based on configuration.
//!
//! Routes tasks to the appropriate backend:
//! - **Candle**: Modern LLMs (Qwen, Llama, Mistral) - loads safetensors directly
//! - **Llama**: GGUF models via llama.cpp (GPT-OSS-20B, custom architectures)
//! - **ONNX**: Embeddings, classification, NER, older models with ONNX exports
//!
//! The `backend` config option controls selection:
//! - `auto` (default): Candle for text-generation, ONNX for everything else
//! - `candle`: Force Candle backend
//! - `llama`: Force llama.cpp backend (requires gguf_file config)
//! - `onnx`: Force ONNX backend

use crate::echo::EchoTask;
use crate::error::{TaskError, TaskResult};
use async_trait::async_trait;
use inference_core::task::{Task, TaskResult as CoreTaskResult};
use inference_core::{BackendType, Config, DataLoader, DataSourceType};
use inference_loader_hf::HfLoader;
use inference_loader_s3::S3Loader;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::info;

// ONNX backend types (always available)
use inference_onnx::{ClipTask, OnnxTask, PaddleOcrTask, Seq2SeqTask};

#[cfg(feature = "candle")]
use inference_candle::{
    CandleAudioClassifierTask, CandleBartTask, CandleEncoderTask, CandleImageToTextTask,
    CandleObjectDetectionTask, CandleSeq2SeqTask, CandleTextGenTask, CandleTtsTask,
};

#[cfg(feature = "llama")]
use inference_llama::LlamaTextGenTask;

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

#[cfg(feature = "postprocess")]
use inference_postprocess::Postprocessor;
use std::sync::atomic::{AtomicBool, Ordering};

/// LazyTask wrapper that defers model loading until first inference request.
///
/// This wrapper:
/// 1. Initially has no model loaded (is_ready() returns false)
/// 2. On first execute() call, loads the model and then executes
/// 3. Supports reload() and unload() methods
pub struct LazyTask {
    config: Config,
    task: Mutex<Option<Arc<dyn Task>>>,
    loaded: Arc<AtomicBool>,
    name: String,
}

impl LazyTask {
    pub fn new(config: Config) -> Self {
        let name = config.effective_task_name();
        Self {
            config,
            task: Mutex::new(None),
            loaded: Arc::new(AtomicBool::new(false)),
            name,
        }
    }
}

/// Wrapper that implements Task for LazyTask, avoiding async_trait lifetime issues
pub struct LazyTaskWrapper {
    inner: Arc<LazyTask>,
}

impl LazyTaskWrapper {
    pub fn new(task: LazyTask) -> Self {
        Self {
            inner: Arc::new(task),
        }
    }
}

#[async_trait]
impl Task for LazyTaskWrapper {
    fn name(&self) -> &str {
        &self.inner.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> CoreTaskResult {
        let task = {
            let mut guard = self.inner.task.lock().await;
            if guard.is_none() {
                info!("Loading model for task '{}'", self.inner.name);
                match TaskRegistry::create(&self.inner.config).await {
                    Ok(t) => {
                        self.inner.loaded.store(true, Ordering::Relaxed);
                        let arc: Arc<dyn inference_core::Task> = Arc::from(t);
                        *guard = Some(arc.clone());
                        arc
                    }
                    Err(e) => {
                        return CoreTaskResult::err(format!("Failed to create task: {e}"));
                    }
                }
            } else {
                guard.as_ref().unwrap().clone()
            }
        };
        task.execute(payload, request_id).await
    }

    async fn reload(&self) -> CoreTaskResult {
        let task = {
            let guard = self.inner.task.lock().await;
            guard.as_ref().cloned()
        };
        match task {
            Some(t) => t.reload().await,
            None => CoreTaskResult::ok("Model not loaded yet".to_string()),
        }
    }

    async fn unload(&self) -> CoreTaskResult {
        let mut guard = self.inner.task.lock().await;
        if let Some(task) = guard.take() {
            self.inner.loaded.store(false, Ordering::Relaxed);
            task.unload().await
        } else {
            CoreTaskResult::ok("Model already unloaded".to_string())
        }
    }

    fn is_ready(&self) -> bool {
        self.inner.loaded.load(Ordering::Relaxed)
    }
}

/// Registry for creating tasks from configuration.
pub struct TaskRegistry;

impl TaskRegistry {
    /// Create a lazy task that defers model loading until first inference request.
    ///
    /// This wraps `create()` with a `LazyTask` that:
    /// 1. Initially has no model loaded (is_ready() returns false)
    /// 2. On first execute() call, loads the model and then executes
    /// 3. Supports reload() and unload() methods
    pub async fn create_lazy(config: &Config) -> TaskResult<Box<dyn Task>> {
        let lazy_task = LazyTask::new(config.clone());
        Ok(Box::new(LazyTaskWrapper::new(lazy_task)))
    }

    /// Create a task based on the configuration.
    ///
    /// Backend routing (when `backend = "auto"`):
    /// - `task_type.is_echo()` → EchoTask (no model needed)
    /// - `task_type == "automatic-speech-recognition"` + candle feature → CandleSeq2SeqTask
    /// - `task_type == "text-to-speech"` + candle feature → CandleTtsTask
    /// - `task_type == "text-generation"` + candle feature → CandleTextGenTask
    /// - `task_type.is_seq2seq()` → Seq2SeqTask (ONNX autoregressive)
    /// - All other task types → OnnxTask (single forward pass)
    ///
    /// When `backend = "llama"`:
    /// - Uses llama.cpp to load GGUF models (requires gguf_file config)
    ///
    /// When `backend = "candle"`:
    /// - TTS (text-to-speech) → CandleTtsTask
    /// - Encoder-decoder seq2seq (ASR, translation, etc.) → CandleSeq2SeqTask
    /// - Decoder-only (text-generation) → CandleTextGenTask
    ///
    /// When `backend = "onnx"`: Force ONNX backend (Seq2SeqTask or OnnxTask)
    pub async fn create(config: &Config) -> TaskResult<Box<dyn Task>> {
        let task_name = config.effective_task_name();
        info!(
            task_type = %config.task_type,
            task_name = %task_name,
            backend = ?config.backend,
            is_seq2seq = config.task_type.is_seq2seq(),
            "Creating task"
        );

        // Echo task - no backend needed
        if config.task_type.is_echo() {
            return Ok(Box::new(EchoTask::new(task_name)));
        }

        // All other tasks require a model path
        if config.model_path.is_none() {
            return Err(TaskError::Config(format!(
                "MAIIA_AI_MODEL_PATH required for task type '{}'",
                config.task_type
            )));
        }

        // Determine effective backend
        let use_llama = Self::should_use_llama(config);
        let use_candle_tts = Self::should_use_candle_tts(config);
        let use_candle_seq2seq = Self::should_use_candle_seq2seq(config);
        let use_candle_bart = Self::should_use_candle_bart(config);
        let use_candle_audio_classifier = Self::should_use_candle_audio_classifier(config);
        let use_candle_encoder = Self::should_use_candle_encoder(config);
        let use_candle_object_detection = Self::should_use_candle_object_detection(config);
        let use_candle_image_to_text = Self::should_use_candle_image_to_text(config);
        let use_candle = Self::should_use_candle(config);
        let use_clip = Self::should_use_clip(config);
        let use_paddle_ocr = Self::should_use_paddle_ocr(config);

        if use_llama {
            Self::create_llama_task(config, task_name).await
        } else if use_candle_tts {
            Self::create_candle_tts_task(config, task_name).await
        } else if use_candle_seq2seq {
            Self::create_candle_seq2seq_task(config, task_name).await
        } else if use_candle_bart {
            Self::create_candle_bart_task(config, task_name).await
        } else if use_candle_audio_classifier {
            Self::create_candle_audio_classifier_task(config, task_name).await
        } else if use_candle_encoder {
            Self::create_candle_encoder_task(config, task_name).await
        } else if use_candle_object_detection {
            Self::create_candle_object_detection_task(config, task_name).await
        } else if use_candle_image_to_text {
            Self::create_candle_image_to_text_task(config, task_name).await
        } else if use_candle {
            Self::create_candle_task(config, task_name).await
        } else if use_clip {
            Self::create_clip_task(config, task_name).await
        } else if use_paddle_ocr {
            Self::create_paddle_ocr_task(config, task_name).await
        } else {
            Self::create_onnx_based_task(config, task_name).await
        }
    }

    /// Determine if we should use llama.cpp backend for GGUF models.
    fn should_use_llama(config: &Config) -> bool {
        let wants_llama = match config.backend {
            BackendType::Llama => true,
            // Auto + gguf_file present → inferred from model metadata
            BackendType::Auto => config.gguf_file.is_some(),
            _ => false,
        };

        if wants_llama {
            #[cfg(feature = "llama")]
            {
                return true;
            }
            #[cfg(not(feature = "llama"))]
            {
                tracing::warn!(
                    "Llama backend requested but 'llama' feature not enabled. \
                     Rebuild with: cargo build --features llama"
                );
                return false;
            }
        }

        false
    }

    /// Determine if we should use Candle backend for decoder-only (text-generation).
    fn should_use_candle(config: &Config) -> bool {
        if !config.task_type.is_decoder_only() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    if matches!(config.backend, BackendType::Candle) {
                        tracing::warn!(
                            "Candle backend requested but 'candle' feature not enabled, falling back to ONNX. \
                             Rebuild with: cargo build --features candle-metal"
                        );
                    }
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for encoder-decoder seq2seq tasks.
    ///
    /// Candle seq2seq supports:
    /// - Whisper (automatic-speech-recognition)
    /// - T5/FlanT5/MADLAD (translation, summarization)
    ///
    /// ONNX is known to have issues with KV-cache for some seq2seq models (e.g., Whisper),
    /// so we prefer Candle for these tasks when explicitly requested or on auto for ASR.
    ///
    /// Note: TTS is handled separately by `should_use_candle_tts`.
    fn should_use_candle_seq2seq(config: &Config) -> bool {
        // TTS is handled separately
        if config.task_type.is_tts() {
            return false;
        }

        // Image-to-text (BLIP) is handled by the dedicated image-to-text backend
        if config.task_type.is_image_to_text() {
            return false;
        }

        let is_enc_dec_seq2seq =
            config.task_type.is_seq2seq() && !config.task_type.is_decoder_only();

        if !is_enc_dec_seq2seq {
            return false;
        }

        match config.backend {
            BackendType::Candle => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    // Auto: prefer Candle for ASR (ONNX Whisper has KV-cache issues)
                    // and for translation/summarization (T5/Marian run natively via Candle)
                    let s = config.task_type.as_str().to_lowercase().replace('-', "_");
                    matches!(
                        s.as_str(),
                        "automatic_speech_recognition" | "translation" | "summarization"
                    )
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle BART backend for seq2seq models
    /// on encoder-only tasks (e.g. BART-MNLI for zero-shot-classification).
    ///
    /// This catches the case where a seq2seq architecture (BART, T5) is used
    /// for a task that is normally encoder-only (zero-shot-classification,
    /// text-classification, etc.). The encoder-only check would skip these
    /// because `is_seq2seq_architecture()` filters them out.
    fn should_use_candle_bart(config: &Config) -> bool {
        // Only applies to encoder-only tasks where the backend is candle
        if !config.task_type.is_encoder_only() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    // Check if the model_path hints at a BART-family model.
                    // The infer_backend() in hf_api.rs already routed seq2seq+encoder_only
                    // to candle, so if we're here with backend=candle and an encoder-only
                    // task, we need to check the model config to decide BART vs encoder.
                    //
                    // We use a heuristic: if config.json in the model dir has "d_model"
                    // (BART-style) rather than "hidden_size" (BERT-style), use BART.
                    // This is checked at task creation time, so we optimistically return
                    // true here and let the task constructor validate.
                    if let Some(ref model_path) = config.model_path {
                        // For HuggingFace models, check if model ID contains known BART archs
                        let lower = model_path.to_lowercase();
                        return lower.contains("bart")
                            || lower.contains("mnli")
                            || lower.contains("nli");
                    }
                    false
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for encoder-only tasks
    /// (BERT, RoBERTa, DistilBERT).
    ///
    /// Encoder tasks include: token-classification, text-classification,
    /// fill-mask, feature-extraction, question-answering.
    ///
    /// These are single forward pass models that Candle can run directly from
    /// safetensors, avoiding the need for ONNX exports.
    fn should_use_candle_encoder(config: &Config) -> bool {
        if !config.task_type.is_encoder_only() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    tracing::warn!(
                        "Candle encoder backend requested but 'candle' feature not enabled"
                    );
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for audio classification tasks
    /// (Wav2Vec2, HuBERT, CLAP).
    ///
    /// Audio classification models typically only have safetensors — no ONNX.
    /// Candle runs the full Wav2Vec2/HuBERT/CLAP encoder natively.
    fn should_use_candle_audio_classifier(config: &Config) -> bool {
        if !config.task_type.is_audio_classification() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    tracing::warn!(
                        "Candle audio classifier backend requested but 'candle' feature not enabled"
                    );
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for object detection tasks
    /// (DETR, Table-Transformer).
    ///
    /// Object detection models (DETR-family) typically only have safetensors — no ONNX.
    /// Candle runs the full DETR pipeline natively (ResNet + Transformer + detection heads).
    fn should_use_candle_object_detection(config: &Config) -> bool {
        if !config.task_type.is_object_detection() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    tracing::warn!(
                        "Candle object detection backend requested but 'candle' feature not enabled"
                    );
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for image-to-text tasks (BLIP captioning).
    ///
    /// BLIP models typically only have safetensors/pytorch_model.bin — ONNX exports
    /// are incomplete (missing tokenizer/config). Candle runs the full BLIP pipeline
    /// natively (ViT encoder + text decoder with autoregressive generation).
    fn should_use_candle_image_to_text(config: &Config) -> bool {
        if !config.task_type.is_image_to_text() {
            return false;
        }

        match config.backend {
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Candle | BackendType::Auto => {
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    tracing::warn!(
                        "Candle image-to-text backend requested but 'candle' feature not enabled"
                    );
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for TTS tasks.
    ///
    /// TTS always routes through the Candle path (even if the feature is off,
    /// so the user gets a clear "enable candle feature" error instead of a
    /// confusing tokenizer 404 from the ONNX fallback).
    fn should_use_candle_tts(config: &Config) -> bool {
        if !config.task_type.is_tts() {
            return false;
        }

        match config.backend {
            // Explicit ONNX/Llama: user knows what they're doing
            BackendType::Onnx | BackendType::Llama => false,
            // Candle or Auto: always route to the Candle TTS path
            // (create_candle_tts_task handles the "feature not enabled" error)
            BackendType::Candle | BackendType::Auto => true,
        }
    }

    /// Determine if we should use CLIP backend for dual-encoder zero-shot classification.
    ///
    /// CLIP is used for zero-shot-image-classification tasks which require:
    /// - Separate vision encoder (vision_model.onnx)
    /// - Separate text encoder (text_model.onnx)
    /// - Cosine similarity computation between embeddings
    #[cfg(feature = "preprocess")]
    fn should_use_clip(config: &Config) -> bool {
        let task_type = config.task_type.as_str().to_lowercase().replace('-', "_");
        matches!(task_type.as_str(), "zero_shot_image_classification")
    }

    #[cfg(not(feature = "preprocess"))]
    fn should_use_clip(_config: &Config) -> bool {
        false
    }

    /// Create a CLIP-based task (for dual-encoder zero-shot classification).
    #[cfg(feature = "preprocess")]
    async fn create_clip_task(config: &Config, task_name: String) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating ClipTask (dual-encoder)"
        );

        // Initialize preprocessor
        let preprocessor = {
            info!("Initializing preprocessor for CLIP");
            let p = Preprocessor::from_config(config).await.map_err(|e| {
                TaskError::Config(format!("Failed to initialize preprocessor: {e}"))
            })?;
            Arc::new(p)
        };

        // Get model directory with ONNX files
        let model_dir = Self::get_model_directory_for_clip(config).await?;

        let task = ClipTask::from_model_dir(&model_dir, task_name, config, preprocessor).await?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "preprocess"))]
    async fn create_clip_task(_config: &Config, _task_name: String) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "CLIP task requires 'preprocess' feature to be enabled".into(),
        ))
    }

    /// Determine if we should use PaddleOCR backend for OCR tasks.
    ///
    /// PaddleOCR is used for OCR tasks which require:
    /// - Detection model (det_model.onnx)
    /// - Recognition model (rec_model.onnx)
    /// - Character dictionary for CTC decoding
    #[cfg(feature = "preprocess")]
    fn should_use_paddle_ocr(config: &Config) -> bool {
        let task_type = config.task_type.as_str().to_lowercase().replace('-', "_");
        matches!(task_type.as_str(), "ocr")
    }

    #[cfg(not(feature = "preprocess"))]
    fn should_use_paddle_ocr(_config: &Config) -> bool {
        false
    }

    /// Create a PaddleOCR-based task (for OCR with detection + recognition).
    #[cfg(feature = "preprocess")]
    async fn create_paddle_ocr_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating PaddleOcrTask (detection + recognition)"
        );

        // Initialize preprocessor
        let preprocessor = {
            info!("Initializing preprocessor for PaddleOCR");
            let p = Preprocessor::from_config(config).await.map_err(|e| {
                TaskError::Config(format!("Failed to initialize preprocessor: {e}"))
            })?;
            Arc::new(p)
        };

        // Get model directory with ONNX files
        let model_dir = Self::get_model_directory_for_paddle_ocr(config).await?;

        let task =
            PaddleOcrTask::from_model_dir(&model_dir, task_name, config, preprocessor).await?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "preprocess"))]
    #[allow(clippy::unused_async)]
    async fn create_paddle_ocr_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "PaddleOCR task requires 'preprocess' feature to be enabled".into(),
        ))
    }

    /// Create a llama.cpp-based task (for GGUF models).
    #[cfg(feature = "llama")]
    async fn create_llama_task(config: &Config, task_name: String) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating LlamaTextGenTask"
        );

        // Get GGUF model path
        let gguf_path = Self::get_gguf_path(config).await?;

        let task = LlamaTextGenTask::from_gguf_path(&gguf_path, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "llama"))]
    #[allow(clippy::unused_async)]
    async fn create_llama_task(_config: &Config, _task_name: String) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Llama backend requested but 'llama' feature is not enabled".into(),
        ))
    }

    /// Create a Candle-based task (for modern LLMs).
    #[cfg(feature = "candle")]
    async fn create_candle_task(config: &Config, task_name: String) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleTextGenTask"
        );

        // Get model directory with safetensors
        let model_dir = Self::get_model_directory_for_candle(config).await?;

        let task = CandleTextGenTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_task(_config: &Config, _task_name: String) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based seq2seq task (for encoder-decoder models).
    ///
    /// Supports:
    /// - Whisper (ASR)
    /// - T5/FlanT5 (translation, summarization)
    #[cfg(feature = "candle")]
    async fn create_candle_seq2seq_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleSeq2SeqTask"
        );

        // Get model directory with safetensors
        let model_dir = Self::get_model_directory_for_candle(config).await?;

        let task = CandleSeq2SeqTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_seq2seq_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle seq2seq backend requested but 'candle' feature is not enabled".into(),
        ))
    }

    /// Create a Candle-based encoder task (for BERT-family models).
    ///
    /// Supports:
    /// - Token classification (NER, POS tagging)
    /// - Text classification / sentiment analysis
    /// - Fill-mask
    /// - Feature extraction (embeddings)
    /// - Question answering (extractive QA)
    #[cfg(feature = "candle")]
    async fn create_candle_encoder_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleEncoderTask"
        );

        // Get model directory with safetensors
        let model_dir = Self::get_model_directory_for_candle(config).await?;

        let task = CandleEncoderTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_encoder_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle encoder backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based audio classification task (Wav2Vec2, HuBERT, CLAP).
    #[cfg(feature = "candle")]
    async fn create_candle_audio_classifier_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleAudioClassifierTask"
        );

        let model_dir = Self::get_model_directory_for_candle(config).await?;
        let task = CandleAudioClassifierTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_audio_classifier_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle audio classifier backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based object detection task (DETR, Table-Transformer).
    #[cfg(feature = "candle")]
    async fn create_candle_object_detection_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleObjectDetectionTask"
        );

        let model_dir = Self::get_model_directory_for_candle(config).await?;
        let task = CandleObjectDetectionTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_object_detection_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle object detection backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based image-to-text task (BLIP image captioning).
    #[cfg(feature = "candle")]
    async fn create_candle_image_to_text_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleImageToTextTask"
        );

        let model_dir = Self::get_model_directory_for_candle(config).await?;
        let task = CandleImageToTextTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_image_to_text_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle image-to-text backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based BART task for seq2seq models on classification tasks.
    ///
    /// Supports:
    /// - BART-MNLI (zero-shot classification via NLI)
    #[cfg(feature = "candle")]
    async fn create_candle_bart_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleBartTask (seq2seq classification)"
        );

        // Get model directory with safetensors
        let model_dir = Self::get_model_directory_for_candle(config).await?;

        let task = CandleBartTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_bart_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Candle BART backend requested but 'candle' feature is not enabled. \
             Rebuild with: cargo build --features candle-metal  (macOS) \
             or: cargo build --features candle-cuda  (Linux/NVIDIA)"
                .into(),
        ))
    }

    /// Create a Candle-based TTS task.
    ///
    /// Supports:
    /// - Parler TTS (high-quality speech synthesis with voice descriptions)
    /// - Qwen3 TTS (custom voice with speech tokenizer)
    #[cfg(feature = "candle")]
    async fn create_candle_tts_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            "Creating CandleTtsTask"
        );

        // Get model directory with safetensors
        let model_dir = Self::get_model_directory_for_candle(config).await?;

        // Check if model is Qwen3 TTS and needs speech_tokenizer subdir files
        Self::maybe_download_speech_tokenizer(config, &model_dir).await?;

        let task = CandleTtsTask::from_model_dir(&model_dir, task_name, config)?;

        Ok(Box::new(task))
    }

    #[cfg(not(feature = "candle"))]
    #[allow(clippy::unused_async)]
    async fn create_candle_tts_task(
        _config: &Config,
        _task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        Err(TaskError::Config(
            "Text-to-speech requires the 'candle' feature. Rebuild with: cargo build --features candle".into(),
        ))
    }

    /// If the model is Qwen3 TTS (model_type == "qwen3_tts"), download the
    /// `speech_tokenizer/` subdirectory files needed for audio decoding.
    ///
    /// This is a no-op for non-HuggingFace sources or non-Qwen3 models.
    async fn maybe_download_speech_tokenizer(
        config: &Config,
        model_dir: &std::path::Path,
    ) -> TaskResult<()> {
        // Only relevant for HuggingFace downloads
        if config.model_source != DataSourceType::HuggingFace {
            return Ok(());
        }

        // Read config.json to check model_type
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to read config.json: {e}")))?;
        let config_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let model_type = config_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        if model_type != "qwen3_tts" {
            return Ok(());
        }

        info!("Qwen3 TTS model detected — downloading speech_tokenizer/ files");

        let model_path = config.model_path.as_ref().unwrap();
        let loader = HfLoader::new(
            model_path,
            config.hf_token.clone(),
            config.model_revision.clone(),
        )
        .await
        .map_err(|e| {
            TaskError::ModelLoad(format!(
                "Failed to init HF loader for speech_tokenizer: {e}"
            ))
        })?;

        // Download speech_tokenizer/model.safetensors (required for audio output)
        match loader.get("speech_tokenizer/model.safetensors").await {
            Ok(p) => info!(path = %p.display(), "Downloaded speech_tokenizer/model.safetensors"),
            Err(e) => {
                tracing::warn!(
                    "Could not download speech_tokenizer/model.safetensors: {e}. \
                     Audio output will be unavailable (codec tokens only)."
                );
                return Ok(());
            }
        }

        // Download speech_tokenizer/config.json (optional — defaults used if absent)
        match loader.get("speech_tokenizer/config.json").await {
            Ok(p) => info!(path = %p.display(), "Downloaded speech_tokenizer/config.json"),
            Err(_) => {
                info!("speech_tokenizer/config.json not found, will use defaults");
            }
        }

        Ok(())
    }

    /// Create an ONNX-based task (Seq2SeqTask or OnnxTask).
    async fn create_onnx_based_task(
        config: &Config,
        task_name: String,
    ) -> TaskResult<Box<dyn Task>> {
        // Initialize preprocessor
        #[cfg(feature = "preprocess")]
        let preprocessor = {
            info!("Initializing preprocessor");
            let p = Preprocessor::from_config(config).await.map_err(|e| {
                TaskError::Config(format!("Failed to initialize preprocessor: {e}"))
            })?;
            Arc::new(p)
        };

        // Initialize postprocessor
        #[cfg(feature = "postprocess")]
        let postprocessor = {
            info!("Initializing postprocessor");
            let tok_path = Self::get_tokenizer_path(config).await.ok();
            let p = Postprocessor::from_config(config, tok_path.as_ref()).map_err(|e| {
                TaskError::Config(format!("Failed to initialize postprocessor: {e}"))
            })?;
            Arc::new(p)
        };

        // Route to Seq2Seq or OnnxTask
        if config.task_type.is_seq2seq() {
            #[cfg(feature = "preprocess")]
            {
                Self::create_seq2seq_task(config, task_name, preprocessor).await
            }
            #[cfg(not(feature = "preprocess"))]
            {
                Err(TaskError::Config(
                    "Seq2Seq tasks require 'preprocess' feature".into(),
                ))
            }
        } else {
            #[cfg(all(feature = "preprocess", feature = "postprocess"))]
            {
                Self::create_onnx_task(config, task_name, preprocessor, postprocessor).await
            }
            #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
            {
                Self::create_onnx_task(config, task_name, preprocessor, ()).await
            }
            #[cfg(not(feature = "preprocess"))]
            {
                Err(TaskError::Config(
                    "ONNX tasks require 'preprocess' feature".into(),
                ))
            }
        }
    }

    /// Create a Seq2SeqTask for autoregressive generation tasks (ONNX backend).
    #[cfg(feature = "preprocess")]
    async fn create_seq2seq_task(
        config: &Config,
        task_name: String,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Box<dyn Task>> {
        info!(
            task_type = %config.task_type,
            is_decoder_only = config.task_type.is_decoder_only(),
            "Creating Seq2SeqTask (ONNX)"
        );

        let model_dir = Self::get_model_directory_for_onnx(config).await?;
        let task = Seq2SeqTask::from_model_dir(&model_dir, task_name, config, preprocessor)?;

        Ok(Box::new(task))
    }

    /// Create an OnnxTask for single forward pass tasks.
    #[cfg(all(feature = "preprocess", feature = "postprocess"))]
    async fn create_onnx_task(
        config: &Config,
        task_name: String,
        preprocessor: Arc<Preprocessor>,
        postprocessor: Arc<Postprocessor>,
    ) -> TaskResult<Box<dyn Task>> {
        let model_path = match config.model_source {
            DataSourceType::HuggingFace => Self::load_onnx_from_huggingface(config).await?,
            DataSourceType::S3 => Self::load_from_s3(config).await?,
            DataSourceType::Local => Self::resolve_local_path(config)?,
        };

        let task =
            OnnxTask::from_file(&model_path, task_name, config, preprocessor, postprocessor)?;

        Ok(Box::new(task))
    }

    #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
    async fn create_onnx_task(
        config: &Config,
        task_name: String,
        preprocessor: Arc<Preprocessor>,
        _postprocessor: (),
    ) -> TaskResult<Box<dyn Task>> {
        let model_path = match config.model_source {
            DataSourceType::HuggingFace => Self::load_onnx_from_huggingface(config).await?,
            DataSourceType::S3 => Self::load_from_s3(config).await?,
            DataSourceType::Local => Self::resolve_local_path(config)?,
        };

        let task = OnnxTask::from_file(&model_path, task_name, config, preprocessor)?;

        Ok(Box::new(task))
    }

    // ========================================================================
    // Model Loading Helpers
    // ========================================================================

    /// Get model directory for Candle (downloads safetensors, config.json, tokenizer.json/tekken.json).
    #[cfg(feature = "candle")]
    async fn get_model_directory_for_candle(config: &Config) -> TaskResult<PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                // Download required files for Candle
                // 1. config.json (required)
                let config_path = loader.get("config.json").await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download config.json: {e}"))
                })?;
                info!(path = %config_path.display(), "Downloaded config.json");

                // 2. Tokenizer: try tokenizer.json -> tekken.json -> vocab.txt -> vocab.json+merges.txt
                //    Not all models need a tokenizer (e.g. audio classification uses raw waveforms),
                //    so we warn instead of erroring when no tokenizer is found.
                if let Ok(path) = loader.get("tokenizer.json").await {
                    info!(path = %path.display(), "Downloaded tokenizer.json");
                } else if let Ok(path) = loader.get("tekken.json").await {
                    info!(path = %path.display(), "Downloaded tekken.json (Voxtral tokenizer)");
                } else if let Ok(path) = loader.get("vocab.txt").await {
                    info!(path = %path.display(), "Downloaded vocab.txt (will build WordPiece tokenizer)");
                } else if let Ok(path) = loader.get("vocab.json").await {
                    info!(path = %path.display(), "Downloaded vocab.json (BPE tokenizer)");
                    // BPE also needs merges.txt
                    if let Ok(mp) = loader.get("merges.txt").await {
                        info!(path = %mp.display(), "Downloaded merges.txt");
                    }
                    // Also grab tokenizer_config.json for special tokens
                    if let Ok(tc) = loader.get("tokenizer_config.json").await {
                        info!(path = %tc.display(), "Downloaded tokenizer_config.json");
                    }
                } else {
                    tracing::debug!(
                        "No tokenizer found — this is expected for audio models that don't use text tokenization"
                    );
                }

                // 3. Weight files: try safetensors -> sharded safetensors -> pytorch_model.bin
                let weight_path = if let Ok(p) = loader.get("model.safetensors").await {
                    p
                } else if let Ok(index_path) = loader.get("model.safetensors.index.json").await {
                    // Parse index to get shard filenames
                    let index_content = std::fs::read_to_string(&index_path).map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to read safetensors index: {e}"))
                    })?;
                    let index: serde_json::Value =
                        serde_json::from_str(&index_content).map_err(|e| {
                            TaskError::ModelLoad(format!("Invalid safetensors index: {e}"))
                        })?;

                    // Get unique shard filenames
                    let mut shard_files: Vec<String> = index
                        .get("weight_map")
                        .and_then(|v| v.as_object())
                        .map(|m| {
                            m.values()
                                .filter_map(|v| v.as_str().map(String::from))
                                .collect()
                        })
                        .unwrap_or_default();
                    shard_files.sort();
                    shard_files.dedup();

                    // Download each shard
                    for shard in &shard_files {
                        loader.get(shard).await.map_err(|e| {
                            TaskError::ModelLoad(format!("Failed to download {shard}: {e}"))
                        })?;
                    }
                    info!(
                        num_shards = shard_files.len(),
                        "Downloaded sharded safetensors"
                    );

                    index_path
                } else if let Ok(p) = loader.get("pytorch_model.bin").await {
                    info!(path = %p.display(), "Downloaded pytorch_model.bin (legacy format)");
                    p
                } else {
                    return Err(TaskError::ModelLoad(
                        "No weight files found (tried model.safetensors, sharded safetensors, and pytorch_model.bin)".into(),
                    ));
                };

                // Return model directory (parent of downloaded files)
                let model_dir = weight_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::S3 => {
                // S3 support for candle - download safetensors
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_prefix = model_path.trim_end_matches('/');

                // Download required files
                loader
                    .get(&format!("{s3_prefix}/config.json"))
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to get config.json: {e}")))?;

                // Try tokenizer.json first, fall back to tekken.json
                let tokenizer_result = loader.get(&format!("{s3_prefix}/tokenizer.json")).await;
                if tokenizer_result.is_err() {
                    loader
                        .get(&format!("{s3_prefix}/tekken.json"))
                        .await
                        .map_err(|e| {
                            TaskError::ModelLoad(format!(
                                "Failed to get tokenizer: no tokenizer.json or tekken.json: {e}"
                            ))
                        })?;
                }

                let safetensors_path = loader
                    .get(&format!("{s3_prefix}/model.safetensors"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to get model.safetensors: {e}"))
                    })?;

                let model_dir = safetensors_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::Local => {
                let path = PathBuf::from(model_path);
                if !path.exists() {
                    return Err(TaskError::ModelNotFound(path.display().to_string()));
                }
                // Verify required files exist
                if !path.join("config.json").exists() {
                    return Err(TaskError::ModelLoad("config.json not found".into()));
                }
                // Check for tokenizer: tokenizer.json OR tekken.json OR vocab.txt OR vocab.json
                if !path.join("tokenizer.json").exists()
                    && !path.join("tekken.json").exists()
                    && !path.join("vocab.txt").exists()
                    && !path.join("vocab.json").exists()
                {
                    return Err(TaskError::ModelLoad(
                        "No tokenizer found: neither tokenizer.json, tekken.json, vocab.txt, nor vocab.json exists".into(),
                    ));
                }
                if !path.join("model.safetensors").exists()
                    && !path.join("model.safetensors.index.json").exists()
                    && !path.join("pytorch_model.bin").exists()
                {
                    return Err(TaskError::ModelLoad(
                        "No weight files found (need model.safetensors, sharded safetensors, or pytorch_model.bin)".into(),
                    ));
                }
                Ok(path)
            }
        }
    }

    /// Get model directory for CLIP dual-encoder models.
    ///
    /// Downloads:
    /// - `onnx/vision_model.onnx` - Vision encoder
    /// - `onnx/text_model.onnx` - Text encoder
    /// - `tokenizer.json` - For text preprocessing
    /// - `preprocessor_config.json` - For image preprocessing
    #[cfg(feature = "preprocess")]
    async fn get_model_directory_for_clip(config: &Config) -> TaskResult<PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                // Download vision encoder
                let vision_path = loader.get("onnx/vision_model.onnx").await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download vision_model.onnx: {e}"))
                })?;
                info!(path = %vision_path.display(), "Downloaded vision encoder");

                // Download text encoder
                let text_path = loader.get("onnx/text_model.onnx").await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download text_model.onnx: {e}"))
                })?;
                info!(path = %text_path.display(), "Downloaded text encoder");

                // Download tokenizer for text preprocessing
                let tokenizer_path = loader.get("tokenizer.json").await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download tokenizer.json: {e}"))
                })?;
                info!(path = %tokenizer_path.display(), "Downloaded tokenizer");

                // Download preprocessor config for image preprocessing
                let preprocessor_path =
                    loader.get("preprocessor_config.json").await.map_err(|e| {
                        TaskError::ModelLoad(format!(
                            "Failed to download preprocessor_config.json: {e}"
                        ))
                    })?;
                info!(path = %preprocessor_path.display(), "Downloaded preprocessor config");

                // Return model directory (parent of onnx folder)
                let onnx_dir = vision_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get ONNX directory".into()))?;
                let model_dir = onnx_dir
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::S3 => {
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_prefix = model_path.trim_end_matches('/');

                // Download vision encoder
                let vision_path = loader
                    .get(&format!("{s3_prefix}/onnx/vision_model.onnx"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download vision_model.onnx: {e}"))
                    })?;

                // Download text encoder
                loader
                    .get(&format!("{s3_prefix}/onnx/text_model.onnx"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download text_model.onnx: {e}"))
                    })?;

                // Download tokenizer
                loader
                    .get(&format!("{s3_prefix}/tokenizer.json"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download tokenizer.json: {e}"))
                    })?;

                // Download preprocessor config
                loader
                    .get(&format!("{s3_prefix}/preprocessor_config.json"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!(
                            "Failed to download preprocessor_config.json: {e}"
                        ))
                    })?;

                // Return model directory
                let onnx_dir = vision_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get ONNX directory".into()))?;
                let model_dir = onnx_dir
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::Local => {
                let path = PathBuf::from(model_path);
                if !path.exists() {
                    return Err(TaskError::ModelNotFound(path.display().to_string()));
                }
                // Verify required files exist
                let onnx_dir = path.join("onnx");
                if !onnx_dir.join("vision_model.onnx").exists() {
                    return Err(TaskError::ModelLoad(
                        "onnx/vision_model.onnx not found".into(),
                    ));
                }
                if !onnx_dir.join("text_model.onnx").exists() {
                    return Err(TaskError::ModelLoad(
                        "onnx/text_model.onnx not found".into(),
                    ));
                }
                if !path.join("tokenizer.json").exists() {
                    return Err(TaskError::ModelLoad("tokenizer.json not found".into()));
                }
                Ok(path)
            }
        }
    }

    /// Get model directory for PaddleOCR models.
    ///
    /// Downloads all required files for PaddleOCR:
    /// - Detection model (det_model.onnx or detection/*/det.onnx)
    /// - Recognition model (rec_model.onnx or languages/*/rec.onnx)
    /// - Character dictionary (dict.txt or languages/*/dict.txt)
    #[cfg(feature = "preprocess")]
    async fn get_model_directory_for_paddle_ocr(config: &Config) -> TaskResult<PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                // Try to download detection model (try different paths)
                let det_paths = [
                    "det_model.onnx",
                    "det.onnx",
                    "detection/v5/det.onnx",
                    "detection/v4/det.onnx",
                ];
                let mut det_path = None;
                for p in &det_paths {
                    if let Ok(path) = loader.get(p).await {
                        info!(path = %path.display(), "Downloaded detection model");
                        det_path = Some(path);
                        break;
                    }
                }
                let det_path = det_path.ok_or_else(|| {
                    TaskError::ModelLoad("Failed to download detection model (det.onnx)".into())
                })?;

                // Try to download recognition model
                let rec_paths = [
                    "rec_model.onnx",
                    "rec.onnx",
                    "languages/english/rec.onnx",
                    "languages/latin/rec.onnx",
                ];
                let mut rec_downloaded = false;
                for p in &rec_paths {
                    if let Ok(path) = loader.get(p).await {
                        info!(path = %path.display(), "Downloaded recognition model");
                        rec_downloaded = true;
                        break;
                    }
                }
                if !rec_downloaded {
                    return Err(TaskError::ModelLoad(
                        "Failed to download recognition model (rec.onnx)".into(),
                    ));
                }

                // Try to download dictionary
                let dict_paths = [
                    "dict.txt",
                    "ppocr_keys_v1.txt",
                    "languages/english/dict.txt",
                    "languages/latin/dict.txt",
                ];
                let mut dict_downloaded = false;
                for p in &dict_paths {
                    if let Ok(path) = loader.get(p).await {
                        info!(path = %path.display(), "Downloaded character dictionary");
                        dict_downloaded = true;
                        break;
                    }
                }
                if !dict_downloaded {
                    return Err(TaskError::ModelLoad(
                        "Failed to download character dictionary (dict.txt)".into(),
                    ));
                }

                // Return model directory (parent of det model)
                let model_dir = det_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                // If det model is in a subdirectory, go up to the root
                let model_dir = if model_dir.ends_with("detection")
                    || model_dir
                        .file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with('v'))
                {
                    model_dir
                        .parent()
                        .and_then(|p| p.parent())
                        .unwrap_or(model_dir)
                } else {
                    model_dir
                };

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::S3 => {
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_prefix = model_path.trim_end_matches('/');

                // Download detection model
                let det_path = loader
                    .get(&format!("{s3_prefix}/det_model.onnx"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download det_model.onnx: {e}"))
                    })?;

                // Download recognition model
                loader
                    .get(&format!("{s3_prefix}/rec_model.onnx"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download rec_model.onnx: {e}"))
                    })?;

                // Download dictionary
                loader
                    .get(&format!("{s3_prefix}/dict.txt"))
                    .await
                    .map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download dict.txt: {e}"))
                    })?;

                // Return model directory
                let model_dir = det_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::Local => {
                let path = PathBuf::from(model_path);
                if !path.exists() {
                    return Err(TaskError::ModelNotFound(path.display().to_string()));
                }
                Ok(path)
            }
        }
    }

    /// Get model directory for ONNX seq2seq models.
    async fn get_model_directory_for_onnx(config: &Config) -> TaskResult<PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                // Download ONNX files - try both text encoder and vision encoder
                let encoder_result = loader.get("onnx/encoder_model.onnx").await;
                if let Ok(ref path) = encoder_result {
                    info!(path = %path.display(), "Downloaded encoder model");
                }
                // Also try vision_encoder for Florence-2 style models
                let vision_encoder_result = loader.get("onnx/vision_encoder.onnx").await;
                if let Ok(ref path) = vision_encoder_result {
                    info!(path = %path.display(), "Downloaded vision encoder model");
                }
                // Try embed_tokens for Florence-2 style models
                let embed_tokens_result = loader.get("onnx/embed_tokens.onnx").await;
                if let Ok(ref path) = embed_tokens_result {
                    info!(path = %path.display(), "Downloaded embed_tokens model");
                }

                // Determine decoder file to download
                // Priority: config.onnx_file > auto-detect standard patterns
                let decoder_path = if let Some(ref onnx_file) = config.onnx_file {
                    // User specified a specific ONNX file
                    let path = loader.get(onnx_file).await.map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download {onnx_file}: {e}"))
                    })?;

                    // For quantized models with external data files (model_q4f16.onnx_data, etc.)
                    // Try to download them - they're required for the model to load
                    if onnx_file.contains("_q4") || onnx_file.contains("_q8") {
                        // Get the base name for external data files
                        let base_name = onnx_file.trim_end_matches(".onnx");
                        // Try to download up to 10 external data shards
                        for i in 0..10 {
                            let data_file = if i == 0 {
                                format!("{base_name}.onnx_data")
                            } else {
                                format!("{base_name}.onnx_data_{i}")
                            };
                            if loader.get(&data_file).await.is_ok() {
                                info!(file = %data_file, "Downloaded external data shard");
                            } else {
                                // No more shards
                                break;
                            }
                        }
                    }

                    path
                } else {
                    // Auto-detect: try standard patterns
                    if let Ok(p) = loader.get("onnx/decoder_model_merged.onnx").await {
                        p
                    } else if let Ok(p) = loader.get("onnx/decoder_with_past_model.onnx").await {
                        p
                    } else if let Ok(p) = loader.get("onnx/decoder_model.onnx").await {
                        p
                    } else if let Ok(p) = loader.get("onnx/model.onnx").await {
                        p
                    } else if let Ok(p) = loader.get("onnx/model_q4f16.onnx").await {
                        // INT4 quantized model - download external data files
                        for i in 0..10 {
                            let data_file = if i == 0 {
                                "onnx/model_q4f16.onnx_data".to_string()
                            } else {
                                format!("onnx/model_q4f16.onnx_data_{i}")
                            };
                            if loader.get(&data_file).await.is_ok() {
                                info!(file = %data_file, "Downloaded external data shard");
                            } else {
                                break;
                            }
                        }
                        p
                    } else {
                        return Err(TaskError::ModelLoad(
                            "No ONNX decoder found (tried decoder_model_merged.onnx, model.onnx, model_q4f16.onnx)".into()
                        ));
                    }
                };

                info!(path = %decoder_path.display(), "Downloaded decoder model");

                let onnx_dir = decoder_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get ONNX directory".into()))?;
                let model_dir = onnx_dir
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::S3 => {
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_prefix = model_path.trim_end_matches('/');

                // Try both text encoder and vision encoder
                let _ = loader
                    .get(&format!("{s3_prefix}/onnx/encoder_model.onnx"))
                    .await;
                let _ = loader
                    .get(&format!("{s3_prefix}/onnx/vision_encoder.onnx"))
                    .await;
                // Try embed_tokens for Florence-2 style models
                let _ = loader
                    .get(&format!("{s3_prefix}/onnx/embed_tokens.onnx"))
                    .await;

                // Determine decoder file - use config.onnx_file if specified
                let decoder_path = if let Some(ref onnx_file) = config.onnx_file {
                    let s3_path = format!("{s3_prefix}/{onnx_file}");
                    let path = loader.get(&s3_path).await.map_err(|e| {
                        TaskError::ModelLoad(format!("Failed to download {onnx_file}: {e}"))
                    })?;

                    // Download external data files for quantized models
                    if onnx_file.contains("_q4") || onnx_file.contains("_q8") {
                        let base_name = onnx_file.trim_end_matches(".onnx");
                        for i in 0..10 {
                            let data_file = if i == 0 {
                                format!("{s3_prefix}/{base_name}.onnx_data")
                            } else {
                                format!("{s3_prefix}/{base_name}.onnx_data_{i}")
                            };
                            if loader.get(&data_file).await.is_ok() {
                                info!(file = %data_file, "Downloaded external data shard");
                            } else {
                                break;
                            }
                        }
                    }

                    path
                } else if let Ok(p) = loader
                    .get(&format!("{s3_prefix}/onnx/decoder_model_merged.onnx"))
                    .await
                {
                    p
                } else if let Ok(p) = loader
                    .get(&format!("{s3_prefix}/onnx/decoder_with_past_model.onnx"))
                    .await
                {
                    p
                } else {
                    loader
                        .get(&format!("{s3_prefix}/onnx/decoder_model.onnx"))
                        .await
                        .map_err(|e| TaskError::ModelLoad(format!("No decoder found: {e}")))?
                };

                let onnx_dir = decoder_path
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get ONNX directory".into()))?;
                let model_dir = onnx_dir
                    .parent()
                    .ok_or_else(|| TaskError::ModelLoad("Cannot get model directory".into()))?;

                Ok(model_dir.to_path_buf())
            }
            DataSourceType::Local => {
                let path = PathBuf::from(model_path);
                if !path.exists() {
                    return Err(TaskError::ModelNotFound(path.display().to_string()));
                }
                Ok(path)
            }
        }
    }

    /// Get tokenizer path for postprocessing.
    #[cfg(feature = "postprocess")]
    async fn get_tokenizer_path(config: &Config) -> TaskResult<std::path::PathBuf> {
        let model_path = config
            .model_path
            .as_ref()
            .ok_or_else(|| TaskError::Config("MODEL_PATH required".into()))?;

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                loader
                    .get("tokenizer.json")
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to get tokenizer: {e}")))
            }
            DataSourceType::S3 => {
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_path = format!("{}/tokenizer.json", model_path.trim_end_matches('/'));
                loader
                    .get(&s3_path)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to get tokenizer: {e}")))
            }
            DataSourceType::Local => {
                let path = std::path::PathBuf::from(model_path).join("tokenizer.json");
                if path.exists() {
                    Ok(path)
                } else {
                    Err(TaskError::ModelLoad(format!(
                        "tokenizer.json not found at {}",
                        path.display()
                    )))
                }
            }
        }
    }

    /// Get GGUF model path for llama.cpp backend.
    ///
    /// Downloads the GGUF file from HuggingFace or S3 if needed.
    /// The `gguf_file` config option specifies which file to download.
    #[cfg(feature = "llama")]
    async fn get_gguf_path(config: &Config) -> TaskResult<PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();
        let gguf_file = config.gguf_file.as_deref().ok_or_else(|| {
            TaskError::Config(
                "MAIIA_AI_GGUF_FILE required when backend=llama (e.g., 'model-Q4_K_M.gguf')".into(),
            )
        })?;

        info!(
            model_path = %model_path,
            gguf_file = %gguf_file,
            "Getting GGUF model path"
        );

        match config.model_source {
            DataSourceType::HuggingFace => {
                let loader = HfLoader::new(
                    model_path,
                    config.hf_token.clone(),
                    config.model_revision.clone(),
                )
                .await
                .map_err(|e| TaskError::ModelLoad(format!("Failed to init HF loader: {e}")))?;

                // Download the GGUF file
                let path = loader.get(gguf_file).await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download {gguf_file}: {e}"))
                })?;

                info!(path = %path.display(), "GGUF model downloaded");
                Ok(path)
            }
            DataSourceType::S3 => {
                let loader = S3Loader::from_config(config)
                    .await
                    .map_err(|e| TaskError::ModelLoad(format!("Failed to init S3 loader: {e}")))?;

                let s3_path = format!("{}/{}", model_path.trim_end_matches('/'), gguf_file);
                let path = loader.get(&s3_path).await.map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to download GGUF from S3: {e}"))
                })?;

                info!(path = %path.display(), "GGUF model downloaded from S3");
                Ok(path)
            }
            DataSourceType::Local => {
                let path = PathBuf::from(model_path).join(gguf_file);
                if !path.exists() {
                    return Err(TaskError::ModelNotFound(path.display().to_string()));
                }
                Ok(path)
            }
        }
    }

    /// Load ONNX model from HuggingFace Hub.
    async fn load_onnx_from_huggingface(config: &Config) -> TaskResult<std::path::PathBuf> {
        let model_id = config.model_path.as_ref().unwrap();
        let onnx_file = config.onnx_file.as_deref().unwrap_or("model.onnx");

        info!(
            model_id = %model_id,
            onnx_file = %onnx_file,
            "Downloading ONNX model from HuggingFace"
        );

        let loader = HfLoader::new(
            model_id,
            config.hf_token.clone(),
            config.model_revision.clone(),
        )
        .await
        .map_err(|e| TaskError::ModelLoad(format!("Failed to initialize HF loader: {e}")))?;

        let path = loader
            .get_onnx(onnx_file)
            .await
            .map_err(|e| TaskError::ModelLoad(format!("Failed to download model: {e}")))?;

        info!(path = %path.display(), "Model downloaded successfully");
        Ok(path)
    }

    /// Load model from S3/MinIO.
    async fn load_from_s3(config: &Config) -> TaskResult<std::path::PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();
        let onnx_file = config.onnx_file.as_deref().unwrap_or("model.onnx");

        info!(
            model_path = %model_path,
            onnx_file = %onnx_file,
            "Downloading model from S3"
        );

        let loader = S3Loader::from_config(config)
            .await
            .map_err(|e| TaskError::ModelLoad(format!("Failed to initialize S3 loader: {e}")))?;

        let s3_path = format!("{}/{}", model_path.trim_end_matches('/'), onnx_file);

        let path = loader
            .get(&s3_path)
            .await
            .map_err(|e| TaskError::ModelLoad(format!("Failed to download model from S3: {e}")))?;

        info!(path = %path.display(), "Model downloaded from S3 successfully");
        Ok(path)
    }

    /// Resolve local model path.
    fn resolve_local_path(config: &Config) -> TaskResult<std::path::PathBuf> {
        let model_path = config.model_path.as_ref().unwrap();
        let path = if let Some(ref onnx_file) = config.onnx_file {
            std::path::PathBuf::from(model_path).join(onnx_file)
        } else {
            std::path::PathBuf::from(model_path)
        };

        if !path.exists() {
            return Err(TaskError::ModelNotFound(path.display().to_string()));
        }

        Ok(path)
    }
}
