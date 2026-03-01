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

use crate::clip::ClipTask;
use crate::echo::EchoTask;
use crate::error::{TaskError, TaskResult};
use crate::onnx::OnnxTask;
use crate::paddle_ocr::PaddleOcrTask;
use crate::seq2seq::Seq2SeqTask;
use inference_core::{BackendType, Config, DataLoader, DataSourceType};
use inference_grpc::task::Task;
use inference_loader_hf::HfLoader;
use inference_loader_s3::S3Loader;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::info;

#[cfg(feature = "candle")]
use crate::candle_seq2seq::CandleSeq2SeqTask;
#[cfg(feature = "candle")]
use crate::candle_text_gen::CandleTextGenTask;
#[cfg(feature = "candle")]
use crate::candle_tts::CandleTtsTask;

#[cfg(feature = "llama")]
use crate::llama_text_gen::LlamaTextGenTask;

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

#[cfg(feature = "postprocess")]
use inference_postprocess::Postprocessor;

/// Registry for creating tasks from configuration.
pub struct TaskRegistry;

impl TaskRegistry {
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
        let use_candle = Self::should_use_candle(config);
        let use_clip = Self::should_use_clip(config);
        let use_paddle_ocr = Self::should_use_paddle_ocr(config);

        if use_llama {
            Self::create_llama_task(config, task_name).await
        } else if use_candle_tts {
            Self::create_candle_tts_task(config, task_name).await
        } else if use_candle_seq2seq {
            Self::create_candle_seq2seq_task(config, task_name).await
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
        match config.backend {
            BackendType::Llama => {
                #[cfg(feature = "llama")]
                {
                    true
                }
                #[cfg(not(feature = "llama"))]
                {
                    tracing::warn!(
                        "Llama backend requested but 'llama' feature not enabled, falling back"
                    );
                    false
                }
            }
            _ => false, // Only explicit llama backend uses llama.cpp
        }
    }

    /// Determine if we should use Candle backend for decoder-only (text-generation).
    fn should_use_candle(config: &Config) -> bool {
        match config.backend {
            BackendType::Candle => {
                #[cfg(feature = "candle")]
                {
                    // For decoder-only models (text-generation)
                    config.task_type.is_decoder_only()
                }
                #[cfg(not(feature = "candle"))]
                {
                    tracing::warn!(
                        "Candle backend requested but feature not enabled, falling back to ONNX"
                    );
                    false
                }
            }
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Auto => {
                // Auto: use Candle for text-generation if feature is enabled
                #[cfg(feature = "candle")]
                {
                    config.task_type.is_decoder_only() // text-generation
                }
                #[cfg(not(feature = "candle"))]
                {
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

        match config.backend {
            BackendType::Candle => {
                #[cfg(feature = "candle")]
                {
                    // Use Candle seq2seq for encoder-decoder models (not decoder-only, not TTS)
                    config.task_type.is_seq2seq() && !config.task_type.is_decoder_only()
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Auto => {
                // Auto: prefer Candle for ASR tasks (ONNX Whisper has KV-cache issues)
                #[cfg(feature = "candle")]
                {
                    let s = config.task_type.as_str().to_lowercase().replace('-', "_");
                    matches!(s.as_str(), "automatic_speech_recognition")
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
        }
    }

    /// Determine if we should use Candle backend for TTS tasks.
    ///
    /// Candle TTS supports:
    /// - Parler TTS (high-quality, supports voice descriptions)
    ///
    /// ONNX MMS-TTS has tokenizer compatibility issues, so we prefer Candle for TTS.
    fn should_use_candle_tts(config: &Config) -> bool {
        if !config.task_type.is_tts() {
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
                    tracing::warn!(
                        "Candle TTS backend requested but feature not enabled, falling back to ONNX"
                    );
                    false
                }
            }
            BackendType::Onnx | BackendType::Llama => false,
            BackendType::Auto => {
                // Auto: prefer Candle for TTS (ONNX MMS-TTS has tokenizer issues)
                #[cfg(feature = "candle")]
                {
                    true
                }
                #[cfg(not(feature = "candle"))]
                {
                    false
                }
            }
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
            "Candle backend requested but 'candle' feature is not enabled".into(),
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

    /// Create a Candle-based TTS task.
    ///
    /// Supports:
    /// - Parler TTS (high-quality speech synthesis with voice descriptions)
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
            "Candle TTS backend requested but 'candle' feature is not enabled".into(),
        ))
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

                // 2. tokenizer.json OR tekken.json (for Voxtral)
                // Try tokenizer.json first, fall back to tekken.json for Voxtral models
                let tokenizer_result = loader.get("tokenizer.json").await;
                if let Ok(path) = tokenizer_result {
                    info!(path = %path.display(), "Downloaded tokenizer.json");
                } else {
                    // tokenizer.json not found, try tekken.json (for Voxtral)
                    let tekken_path = loader.get("tekken.json").await.map_err(|e| {
                        TaskError::ModelLoad(format!(
                            "Failed to download tokenizer: no tokenizer.json or tekken.json found: {e}"
                        ))
                    })?;
                    info!(path = %tekken_path.display(), "Downloaded tekken.json (Voxtral tokenizer)");
                }

                // 3. safetensors files (required - try single file first, then sharded)
                let safetensors_path = if let Ok(p) = loader.get("model.safetensors").await {
                    p
                } else {
                    // Try to get sharded files by downloading the index
                    if let Ok(index_path) = loader.get("model.safetensors.index.json").await {
                        // Parse index to get shard filenames
                        let index_content = std::fs::read_to_string(&index_path).map_err(|e| {
                            TaskError::ModelLoad(format!("Failed to read safetensors index: {e}"))
                        })?;
                        let index: serde_json::Value = serde_json::from_str(&index_content)
                            .map_err(|e| {
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
                    } else {
                        return Err(TaskError::ModelLoad(
                            "No safetensors files found (tried model.safetensors and sharded)"
                                .into(),
                        ));
                    }
                };

                // Return model directory (parent of downloaded files)
                let model_dir = safetensors_path
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
                // Check for tokenizer.json OR tekken.json (for Voxtral)
                if !path.join("tokenizer.json").exists() && !path.join("tekken.json").exists() {
                    return Err(TaskError::ModelLoad(
                        "No tokenizer found: neither tokenizer.json nor tekken.json exists".into(),
                    ));
                }
                if !path.join("model.safetensors").exists()
                    && !path.join("model.safetensors.index.json").exists()
                {
                    return Err(TaskError::ModelLoad("No safetensors files found".into()));
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
