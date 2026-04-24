//! Main preprocessor that handles all input types.
//!
//! Orchestrates tokenization, image processing, and audio processing
//! based on the task type and input format.

use crate::error::{PreprocessError, PreprocessResult};
use crate::input::{
    AudioInput, DocumentInput, ImageInput, ImageSource, QAInput, RawInput, TextInput,
    TextPairInput, VisionLanguageInput,
};
use inference_core::{Config, DataLoader, DataSourceType};
use inference_loader_hf::HfLoader;
use inference_loader_s3::S3Loader;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::info;

#[cfg(feature = "text")]
use crate::tokenizer::Tokenizer;

#[cfg(feature = "image")]
use crate::image::{ImageConfig, ImageProcessor};

#[cfg(feature = "audio")]
use crate::audio::{AudioConfig, AudioProcessor};

/// Preprocessed output ready for ONNX inference.
#[derive(Debug)]
pub struct PreprocessedOutput {
    /// JSON-compatible inputs for ONNX model
    pub inputs: serde_json::Value,
    /// Offset mapping for QA tasks [(start_char, end_char), ...]
    /// Maps token positions to character positions in the original text
    pub offset_mapping: Option<Vec<(usize, usize)>>,
}

impl PreprocessedOutput {
    /// Create from JSON inputs.
    pub fn new(inputs: serde_json::Value) -> Self {
        Self {
            inputs,
            offset_mapping: None,
        }
    }

    /// Create with offset mapping (for QA tasks).
    pub fn with_offset_mapping(inputs: serde_json::Value, offsets: Vec<(usize, usize)>) -> Self {
        Self {
            inputs,
            offset_mapping: Some(offsets),
        }
    }

    /// Convert to JSON string payload.
    pub fn to_payload(&self) -> PreprocessResult<String> {
        serde_json::to_string(&serde_json::json!({
            "inputs": self.inputs
        }))
        .map_err(|e| PreprocessError::InvalidInput(format!("JSON serialization failed: {e}")))
    }
}

/// Validate a user-provided local file path to prevent directory traversal.
///
/// The path is resolved against the current working directory and checked
/// to ensure it does not escape the base directory.
fn sanitize_local_path(path: &str) -> PreprocessResult<std::path::PathBuf> {
    // Reject paths with parent-directory references
    if path.contains("..") {
        return Err(PreprocessError::InvalidInput(
            "Path contains forbidden '..' sequence".into(),
        ));
    }

    let base = std::env::current_dir().map_err(|e| {
        PreprocessError::FileLoad(format!("Failed to determine working directory: {e}"))
    })?;

    let target = base
        .join(path)
        .canonicalize()
        .map_err(|e| PreprocessError::FileLoad(format!("Invalid or inaccessible path: {e}")))?;

    if !target.starts_with(&base) {
        return Err(PreprocessError::InvalidInput(
            "Path traversal detected: path escapes allowed directory".into(),
        ));
    }

    Ok(target)
}

/// Main preprocessor that handles all input types.
pub struct Preprocessor {
    #[cfg(feature = "text")]
    tokenizer: Option<Tokenizer>,

    #[cfg(feature = "image")]
    image_processor: Option<ImageProcessor>,

    #[cfg(feature = "audio")]
    audio_processor: Option<AudioProcessor>,

    /// S3 loader for fetching files
    s3_loader: Option<Arc<RwLock<S3Loader>>>,

    /// Config reference
    config: Config,
}

impl Preprocessor {
    /// Create a preprocessor from config.
    ///
    /// This will:
    /// 1. Download tokenizer.json from HuggingFace/S3 if needed
    /// 2. Initialize appropriate processors based on task type
    pub async fn from_config(config: &Config) -> PreprocessResult<Self> {
        info!(task_type = %config.task_type, "Initializing preprocessor");

        // Initialize S3 loader if needed
        let s3_loader = if config.s3_bucket.is_some() {
            let loader = S3Loader::from_config(config)
                .await
                .map_err(|e| PreprocessError::S3(format!("Failed to init S3 loader: {e}")))?;
            Some(Arc::new(RwLock::new(loader)))
        } else {
            None
        };

        // Load tokenizer if this is a text task
        #[cfg(feature = "text")]
        let tokenizer = if Self::needs_tokenizer(&config.task_type.0) {
            Some(Self::load_tokenizer(config).await?)
        } else {
            None
        };

        #[cfg(not(feature = "text"))]
        let tokenizer: Option<()> = None;

        // Initialize image processor if needed
        #[cfg(feature = "image")]
        let image_processor = if Self::needs_image_processor(&config.task_type.0) {
            Some(Self::create_image_processor(&config.task_type.0))
        } else {
            None
        };

        #[cfg(not(feature = "image"))]
        let image_processor: Option<()> = None;

        // Initialize audio processor if needed
        #[cfg(feature = "audio")]
        let audio_processor = if Self::needs_audio_processor(&config.task_type.0) {
            Some(Self::create_audio_processor(&config.task_type.0))
        } else {
            None
        };

        #[cfg(not(feature = "audio"))]
        let audio_processor: Option<()> = None;

        Ok(Self {
            #[cfg(feature = "text")]
            tokenizer,
            #[cfg(feature = "image")]
            image_processor,
            #[cfg(feature = "audio")]
            audio_processor,
            s3_loader,
            config: config.clone(),
        })
    }

    /// Create an empty preprocessor (no-op).
    ///
    /// This is useful for testing or when preprocessing should be skipped.
    /// Calling `process()` on this will fail for raw inputs but work for
    /// passthrough tensor inputs.
    pub fn empty() -> Self {
        Self {
            #[cfg(feature = "text")]
            tokenizer: None,
            #[cfg(feature = "image")]
            image_processor: None,
            #[cfg(feature = "audio")]
            audio_processor: None,
            s3_loader: None,
            config: Config::default(),
        }
    }

    /// Process raw input and return ONNX-ready tensors.
    pub async fn process(&self, input: &RawInput) -> PreprocessResult<PreprocessedOutput> {
        match input {
            RawInput::Text(text_input) => self.process_text(text_input),
            RawInput::QuestionAnswer(qa_input) => self.process_qa(qa_input),
            RawInput::TextPair(pair_input) => self.process_text_pair(pair_input),
            RawInput::Image(image_input) => self.process_image(image_input).await,
            RawInput::Audio(audio_input) => self.process_audio(audio_input).await,
            RawInput::VisionLanguage(vl_input) => self.process_vision_language(vl_input).await,
            RawInput::Document(doc_input) => self.process_document(doc_input).await,
        }
    }

    /// Process a raw JSON payload string.
    pub async fn process_json(&self, payload: &str) -> PreprocessResult<PreprocessedOutput> {
        // Use trace level for payload content to avoid leaking PII/API keys into logs.
        tracing::trace!(payload_len = payload.len(), "Preprocessing JSON payload");

        let input: RawInput = serde_json::from_str(payload).map_err(|e| {
            tracing::error!(
                error = %e,
                payload_len = payload.len(),
                "Failed to parse payload as RawInput"
            );
            PreprocessError::InvalidInput(format!("Invalid JSON input: {e}"))
        })?;
        self.process(&input).await
    }

    // ========================================================================
    // Input Processing Methods
    // ========================================================================

    fn process_text(&self, input: &TextInput) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(feature = "text")]
        {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;

            let texts: Vec<String> = input
                .text
                .to_vec()
                .into_iter()
                .map(|t| {
                    if let Some(ref prefix) = input.prefix {
                        format!("{prefix}{t}")
                    } else {
                        t
                    }
                })
                .collect();

            let output = tokenizer.encode_batch(&texts)?;
            Ok(PreprocessedOutput::new(output.to_json_inputs()))
        }

        #[cfg(not(feature = "text"))]
        Err(PreprocessError::Config("Text feature not enabled".into()))
    }

    fn process_qa(&self, input: &QAInput) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(feature = "text")]
        {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;

            let output = tokenizer.encode_pair(&input.question, &input.context)?;

            // Extract offset mapping for the first (only) sequence
            let offsets = output
                .offset_mapping
                .as_ref()
                .and_then(|om| om.first().cloned())
                .unwrap_or_default();

            Ok(PreprocessedOutput::with_offset_mapping(
                output.to_json_inputs(),
                offsets,
            ))
        }

        #[cfg(not(feature = "text"))]
        Err(PreprocessError::Config("Text feature not enabled".into()))
    }

    fn process_text_pair(&self, input: &TextPairInput) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(feature = "text")]
        {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;

            let output = tokenizer.encode_pair(&input.text_a, &input.text_b)?;
            Ok(PreprocessedOutput::new(output.to_json_inputs()))
        }

        #[cfg(not(feature = "text"))]
        Err(PreprocessError::Config("Text feature not enabled".into()))
    }

    async fn process_image(&self, input: &ImageInput) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(feature = "image")]
        {
            let processor = self
                .image_processor
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Image processor not initialized".into()))?;

            let bytes = self
                .load_image_bytes(&input.image, input.source.as_ref())
                .await?;
            let output = processor.process_bytes(&bytes)?;
            Ok(PreprocessedOutput::new(output.to_json_inputs()))
        }

        #[cfg(not(feature = "image"))]
        Err(PreprocessError::Config("Image feature not enabled".into()))
    }

    async fn process_audio(&self, input: &AudioInput) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(feature = "audio")]
        {
            let processor = self
                .audio_processor
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Audio processor not initialized".into()))?;

            let bytes = self.load_audio_bytes(&input.audio).await?;
            let output = processor.process_bytes(&bytes)?;
            Ok(PreprocessedOutput::new(output.to_json_inputs()))
        }

        #[cfg(not(feature = "audio"))]
        Err(PreprocessError::Config("Audio feature not enabled".into()))
    }

    async fn process_vision_language(
        &self,
        input: &VisionLanguageInput,
    ) -> PreprocessResult<PreprocessedOutput> {
        #[cfg(all(feature = "text", feature = "image"))]
        {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;
            let image_processor = self
                .image_processor
                .as_ref()
                .ok_or_else(|| PreprocessError::Config("Image processor not initialized".into()))?;

            // Process text
            let text_output = tokenizer.encode(&input.text)?;

            // Process image
            let image_bytes = self.load_image_bytes(&input.image, None).await?;
            let image_output = image_processor.process_bytes(&image_bytes)?;

            // Combine outputs
            let mut inputs = serde_json::Map::new();

            // Add text inputs
            if let serde_json::Value::Object(text_map) = text_output.to_json_inputs() {
                for (k, v) in text_map {
                    inputs.insert(k, v);
                }
            }

            // Add image inputs
            if let serde_json::Value::Object(img_map) = image_output.to_json_inputs() {
                for (k, v) in img_map {
                    inputs.insert(k, v);
                }
            }

            Ok(PreprocessedOutput::new(serde_json::Value::Object(inputs)))
        }

        #[cfg(not(all(feature = "text", feature = "image")))]
        Err(PreprocessError::Config(
            "Text and Image features required".into(),
        ))
    }

    async fn process_document(
        &self,
        input: &DocumentInput,
    ) -> PreprocessResult<PreprocessedOutput> {
        // Same as vision-language for now
        self.process_vision_language(&VisionLanguageInput {
            image: input.document.clone(),
            text: input.question.clone(),
        })
        .await
    }

    // ========================================================================
    // Helper Methods
    // ========================================================================

    /// Check if task type needs a tokenizer.
    fn needs_tokenizer(task_type: &str) -> bool {
        let task = task_type.to_lowercase();

        // Exclude image-only and audio-only tasks (they use preprocessor_config.json, not tokenizer)
        if task.starts_with("image-") && !task.contains("text") {
            return false;
        }
        if task.starts_with("audio-") && !task.contains("text") {
            return false;
        }
        // Exclude pure vision tasks
        if task == "object-detection" || task == "depth-estimation" || task == "image-segmentation"
        {
            return false;
        }

        // Text tasks
        task.contains("text") ||
        task.contains("embed") ||
        task.contains("feature-extraction") ||  // More specific than "feature"
        task.contains("text-classification") ||  // Text classification, not image/audio
        task.contains("qa") ||
        task.contains("question") ||
        task.contains("summariz") ||
        task.contains("translat") ||
        task.contains("generation") ||
        task.contains("fill") ||
        task.contains("mask") ||
        task.contains("token") ||
        task.contains("ner") ||
        task.contains("sentiment") ||
        task.contains("rerank") ||
        task.contains("similar") ||
        task.contains("zero-shot") ||  // zero-shot-classification uses text
        // Multimodal with text
        task.contains("vqa") ||
        task.contains("visual") ||
        task.contains("document") ||
        task.contains("vlm") ||
        task.contains("caption") ||
        task.contains("florence")
    }

    /// Check if task type needs an image processor.
    fn needs_image_processor(task_type: &str) -> bool {
        let task = task_type.to_lowercase();
        task.contains("image")
            || task.contains("vision")
            || task.contains("vqa")
            || task.contains("visual")
            || task.contains("document")
            || task.contains("ocr")
            || task.contains("detection")
            || task.contains("segment")
            || task.contains("depth")
            || task.contains("florence")
            || task.contains("clip")
    }

    /// Check if task type needs an audio processor.
    fn needs_audio_processor(task_type: &str) -> bool {
        let task = task_type.to_lowercase();
        task.contains("audio")
            || task.contains("speech")
            || task.contains("asr")
            || task.contains("whisper")
            || task.contains("tts")
            || task.contains("voice")
    }

    /// Create image processor with appropriate config for task type.
    #[cfg(feature = "image")]
    fn create_image_processor(task_type: &str) -> ImageProcessor {
        let task = task_type.to_lowercase();

        let config = if task.contains("clip") {
            ImageConfig::clip()
        } else if task.contains("florence")
            || task.contains("vqa")
            || task.contains("visual-question")
            || task.contains("document-question")
            || task.contains("image-text-to-text")
        {
            // Florence-2 style models (768x768, ImageNet normalization)
            // Used for: image-text-to-text, VQA, document QA
            ImageConfig::florence2()
        } else if task.contains("detection") {
            // DETR-style models need pixel_mask
            ImageConfig::detr()
        } else if task.contains("trocr") || task.contains("image-to-text") {
            // TrOCR and similar OCR models use 384x384 with 0.5 normalization
            ImageConfig::trocr()
        } else if task.contains("vit") {
            ImageConfig::vit(384) // ViT-Large default
        } else {
            ImageConfig::default() // 224x224 ImageNet
        };

        ImageProcessor::with_config(config)
    }

    /// Create audio processor with appropriate config for task type.
    #[cfg(feature = "audio")]
    fn create_audio_processor(task_type: &str) -> AudioProcessor {
        let task = task_type.to_lowercase();

        let config = if task.contains("classification") || task.contains("wav2vec") {
            // Audio classification models (wav2vec2) use raw waveform
            AudioConfig::wav2vec2()
        } else {
            // ASR models (Whisper) use mel spectrogram
            AudioConfig::whisper()
        };

        AudioProcessor::with_config(config)
    }

    /// Load tokenizer from HuggingFace or S3.
    #[cfg(feature = "text")]
    async fn load_tokenizer(config: &Config) -> PreprocessResult<Tokenizer> {
        let model_path = config
            .model_path
            .as_ref()
            .ok_or_else(|| PreprocessError::Config("MODEL_PATH required for tokenizer".into()))?;

        info!(model_path = %model_path, "Loading tokenizer");

        let tokenizer_path = match config.model_source {
            DataSourceType::HuggingFace => {
                Self::download_tokenizer_hf(
                    model_path,
                    config.hf_token.as_deref(),
                    config.model_revision.as_deref(),
                )
                .await?
            }
            DataSourceType::S3 => Self::download_tokenizer_s3(config).await?,
            DataSourceType::Local => {
                let path = PathBuf::from(model_path).join("tokenizer.json");
                if !path.exists() {
                    return Err(PreprocessError::FileLoad(format!(
                        "tokenizer.json not found at {}",
                        path.display()
                    )));
                }
                path
            }
        };

        Tokenizer::from_file(&tokenizer_path)
    }

    /// Download tokenizer.json from HuggingFace.
    #[cfg(feature = "text")]
    async fn download_tokenizer_hf(
        model_id: &str,
        token: Option<&str>,
        revision: Option<&str>,
    ) -> PreprocessResult<PathBuf> {
        let loader = HfLoader::new(
            model_id,
            token.map(String::from),
            revision.map(String::from),
        )
        .await
        .map_err(|e| PreprocessError::FileLoad(format!("Failed to init HF loader: {e}")))?;

        loader
            .get("tokenizer.json")
            .await
            .map_err(|e| PreprocessError::FileLoad(format!("Failed to download tokenizer: {e}")))
    }

    /// Download tokenizer.json from S3.
    #[cfg(feature = "text")]
    async fn download_tokenizer_s3(config: &Config) -> PreprocessResult<PathBuf> {
        let loader = S3Loader::from_config(config)
            .await
            .map_err(|e| PreprocessError::S3(format!("Failed to init S3 loader: {e}")))?;

        let model_path = config.model_path.as_ref().unwrap();
        let s3_path = format!("{}/tokenizer.json", model_path.trim_end_matches('/'));

        loader
            .get(&s3_path)
            .await
            .map_err(|e| PreprocessError::S3(format!("Failed to download tokenizer: {e}")))
    }

    /// Load image bytes from various sources.
    #[cfg(feature = "image")]
    async fn load_image_bytes(
        &self,
        path: &str,
        source: Option<&ImageSource>,
    ) -> PreprocessResult<Vec<u8>> {
        let source = source
            .cloned()
            .unwrap_or_else(|| Self::detect_image_source(path));

        match source {
            ImageSource::S3 => self.load_file_bytes(path).await,
            ImageSource::Base64 => {
                let data = if path.starts_with("data:") {
                    path.split(',').nth(1).unwrap_or(path)
                } else {
                    path
                };
                base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                    .map_err(|e| PreprocessError::Image(format!("Invalid base64: {e}")))
            }
            ImageSource::Local => {
                let safe_path = sanitize_local_path(path)?;
                tokio::fs::read(&safe_path)
                    .await
                    .map_err(|e| PreprocessError::FileLoad(format!("Failed to read file: {e}")))
            }
            ImageSource::Url => {
                // For now, treat URLs as local paths or error
                // TODO: Add HTTP client for URL fetching
                Err(PreprocessError::InvalidInput(
                    "URL fetching not implemented".into(),
                ))
            }
            ImageSource::Auto => {
                // Re-detect and process
                let detected = Self::detect_image_source(path);
                if matches!(detected, ImageSource::Auto) {
                    // Default to local
                    let safe_path = sanitize_local_path(path)?;
                    tokio::fs::read(&safe_path)
                        .await
                        .map_err(|e| PreprocessError::FileLoad(format!("Failed to read file: {e}")))
                } else {
                    Box::pin(self.load_image_bytes(path, Some(&detected))).await
                }
            }
        }
    }

    /// Detect image source from path/URI.
    fn detect_image_source(path: &str) -> ImageSource {
        if path.starts_with("s3://") {
            ImageSource::S3
        } else if path.starts_with("data:") || (path.len() > 100 && !path.contains('/')) {
            ImageSource::Base64
        } else if path.starts_with("http://") || path.starts_with("https://") {
            ImageSource::Url
        } else {
            ImageSource::Local
        }
    }

    /// Load file bytes from S3 or local path.
    async fn load_file_bytes(&self, path: &str) -> PreprocessResult<Vec<u8>> {
        if path.starts_with("s3://") {
            // Parse S3 URI: s3://bucket/key
            let path = path.strip_prefix("s3://").unwrap();
            let parts: Vec<&str> = path.splitn(2, '/').collect();
            if parts.len() != 2 {
                return Err(PreprocessError::S3(
                    "Invalid S3 URI format: missing key".into(),
                ));
            }

            let (bucket, key) = (parts[0], parts[1]);
            if bucket.is_empty() {
                return Err(PreprocessError::S3(
                    "Invalid S3 URI: empty bucket name".into(),
                ));
            }
            if key.is_empty() {
                return Err(PreprocessError::S3("Invalid S3 URI: empty key".into()));
            }

            let s3_loader = self
                .s3_loader
                .as_ref()
                .ok_or_else(|| PreprocessError::S3("S3 loader not configured".into()))?;

            let loader = s3_loader.read().await;
            let local_path = loader
                .get(key)
                .await
                .map_err(|e| PreprocessError::S3(format!("Failed to download from S3: {e}")))?;

            tokio::fs::read(&local_path)
                .await
                .map_err(|e| PreprocessError::FileLoad(format!("Failed to read cached file: {e}")))
        } else {
            // Local file
            let safe_path = sanitize_local_path(path)?;
            tokio::fs::read(&safe_path)
                .await
                .map_err(|e| PreprocessError::FileLoad(format!("Failed to read file: {e}")))
        }
    }

    /// Load audio bytes from various sources (base64, S3, local file).
    #[cfg(feature = "audio")]
    async fn load_audio_bytes(&self, path: &str) -> PreprocessResult<Vec<u8>> {
        // Check for base64 data URI
        if path.starts_with("data:audio/") {
            let data = path.split(',').nth(1).unwrap_or(path);
            return base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
                .map_err(|e| PreprocessError::Audio(format!("Invalid base64 audio: {e}")));
        }

        // Check for raw base64 (no data URI prefix, long string without path separators)
        if path.len() > 100 && !path.contains('/') && !path.contains('\\') {
            return base64::Engine::decode(&base64::engine::general_purpose::STANDARD, path)
                .map_err(|e| PreprocessError::Audio(format!("Invalid base64 audio: {e}")));
        }

        // Otherwise use the generic file loader (S3 or local)
        self.load_file_bytes(path).await
    }

    /// Decode token IDs back to text.
    ///
    /// Used by seq2seq tasks to convert generated tokens back to text.
    /// Returns an error if no tokenizer is configured.
    #[cfg(feature = "text")]
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> PreprocessResult<String> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;

        tokenizer.decode(token_ids, skip_special_tokens)
    }

    /// Decode a batch of token ID sequences back to text.
    #[cfg(feature = "text")]
    pub fn decode_batch(
        &self,
        batch_token_ids: &[Vec<i64>],
        skip_special_tokens: bool,
    ) -> PreprocessResult<Vec<String>> {
        let tokenizer = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| PreprocessError::Config("Tokenizer not initialized".into()))?;

        tokenizer.decode_batch(batch_token_ids, skip_special_tokens)
    }
}

impl std::fmt::Debug for Preprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Preprocessor")
            .field("task_type", &self.config.task_type)
            .finish_non_exhaustive()
    }
}
