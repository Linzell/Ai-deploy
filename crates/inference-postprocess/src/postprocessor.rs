//! Main postprocessor that handles all output types.
//!
//! Orchestrates decoding, classification, QA, and embedding postprocessing
//! based on the task type and output format.

use crate::error::{PostprocessError, PostprocessResult};
use crate::output::{ClassificationOutput, GenericOutput, PostprocessedOutput, QAOutput};
use inference_core::Config;
use serde_json::Value;
use std::collections::HashMap;
use std::path::PathBuf;
use tracing::{debug, info};

#[cfg(feature = "text")]
use crate::decoder::TextDecoder;

#[cfg(feature = "classification")]
use crate::classification::ClassificationPostprocessor;

#[cfg(feature = "qa")]
use crate::qa::QAPostprocessor;

#[cfg(feature = "embeddings")]
use crate::embeddings::{EmbeddingPostprocessor, PoolingStrategy};

/// Task category for selecting postprocessing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskCategory {
    /// Text classification (sentiment, topic, etc.)
    TextClassification,
    /// Token classification (NER, POS)
    TokenClassification,
    /// Question answering (extractive)
    QuestionAnswering,
    /// Feature extraction (embeddings)
    FeatureExtraction,
    /// Text generation (causal LM)
    TextGeneration,
    /// Summarization
    Summarization,
    /// Translation
    Translation,
    /// Fill mask (masked LM)
    FillMask,
    /// Image classification
    ImageClassification,
    /// Object detection
    ObjectDetection,
    /// ASR (speech-to-text)
    AutomaticSpeechRecognition,
    /// Text-to-speech
    TextToSpeech,
    /// Sentence similarity
    SentenceSimilarity,
    /// Zero-shot classification
    ZeroShotClassification,
    /// Reranking
    Reranking,
    /// Generic (raw output passthrough)
    Generic,
}

impl TaskCategory {
    /// Task type pattern matchers in priority order.
    /// Each entry is (patterns_to_match, exclusions, category).
    /// Order matters: more specific patterns are checked first.
    const TASK_PATTERNS: &'static [(
        &'static [&'static str],
        &'static [&'static str],
        TaskCategory,
    )] = &[
        // QA - check first as it's very specific
        (
            &["question-answering", "qa"],
            &[],
            TaskCategory::QuestionAnswering,
        ),
        // Token classification before text classification
        (
            &["token-classification", "ner"],
            &[],
            TaskCategory::TokenClassification,
        ),
        // Zero-shot before generic classification (contains "classification")
        (&["zero-shot"], &[], TaskCategory::ZeroShotClassification),
        // Image classification before text classification
        (
            &["image-classification"],
            &[],
            TaskCategory::ImageClassification,
        ),
        // Text classification (with exclusions for image/zero-shot)
        (
            &["text-classification", "sentiment"],
            &[],
            TaskCategory::TextClassification,
        ),
        // Reranking
        (&["rerank"], &[], TaskCategory::Reranking),
        // Feature extraction / embeddings
        (
            &[
                "feature-extraction",
                "embed",
                "sentence-similarity",
                "similar",
            ],
            &[],
            TaskCategory::FeatureExtraction,
        ),
        // Text generation
        (
            &["text-generation", "causal"],
            &[],
            TaskCategory::TextGeneration,
        ),
        // Summarization
        (
            &["summarization", "summariz"],
            &[],
            TaskCategory::Summarization,
        ),
        // Translation
        (&["translation", "translat"], &[], TaskCategory::Translation),
        // Fill mask
        (&["fill-mask", "mask"], &[], TaskCategory::FillMask),
        // Object detection
        (
            &["object-detection", "detection"],
            &[],
            TaskCategory::ObjectDetection,
        ),
        // ASR
        (
            &["asr", "speech-recognition", "whisper", "automatic-speech"],
            &[],
            TaskCategory::AutomaticSpeechRecognition,
        ),
        // TTS
        (&["tts", "text-to-speech"], &[], TaskCategory::TextToSpeech),
        // Generic classification fallback (must exclude image/zero-shot variants)
        (
            &["classification"],
            &["image", "zero"],
            TaskCategory::TextClassification,
        ),
    ];

    /// Detect task category from task type string.
    pub fn from_task_type(task_type: &str) -> Self {
        let task = task_type.to_lowercase();

        for &(patterns, exclusions, category) in Self::TASK_PATTERNS {
            // Check if any pattern matches
            let matches = patterns.iter().any(|p| task.contains(p));
            // Check that no exclusions match
            let excluded = exclusions.iter().any(|e| task.contains(e));

            if matches && !excluded {
                return category;
            }
        }

        TaskCategory::Generic
    }
}

/// Context from preprocessing needed for postprocessing.
#[derive(Debug, Clone, Default)]
pub struct PreprocessContext {
    /// Original input text (for QA answer extraction)
    pub original_text: Option<String>,
    /// Original context (for QA)
    pub context: Option<String>,
    /// Token to character offset mapping
    pub offset_mapping: Option<Vec<(usize, usize)>>,
    /// Input token IDs (for some decoding operations)
    pub input_ids: Option<Vec<i64>>,
    /// Attention mask
    pub attention_mask: Option<Vec<i64>>,
    /// Candidate labels for zero-shot
    pub candidate_labels: Option<Vec<String>>,
}

/// Main postprocessor that handles all output types.
pub struct Postprocessor {
    /// Task category
    category: TaskCategory,

    #[cfg(feature = "text")]
    decoder: Option<TextDecoder>,

    #[cfg(feature = "classification")]
    classification: Option<ClassificationPostprocessor>,

    #[cfg(feature = "qa")]
    qa: Option<QAPostprocessor>,

    #[cfg(feature = "embeddings")]
    embeddings: Option<EmbeddingPostprocessor>,

    /// Label mapping for classification
    id2label: HashMap<i64, String>,

    /// Task type string (for logging)
    task_type: String,
}

impl Postprocessor {
    /// Create a postprocessor from config.
    pub fn from_config(
        config: &Config,
        tokenizer_path: Option<&PathBuf>,
    ) -> PostprocessResult<Self> {
        let task_type = config.task_type.as_str().to_string();
        let category = TaskCategory::from_task_type(&task_type);

        info!(task_type = %task_type, category = ?category, "Initializing postprocessor");

        // Load tokenizer for text tasks
        #[cfg(feature = "text")]
        let decoder = if let Some(path) = tokenizer_path {
            if Self::needs_decoder(category) {
                Some(TextDecoder::from_file(path)?)
            } else {
                None
            }
        } else {
            None
        };

        // Initialize classification postprocessor
        #[cfg(feature = "classification")]
        let classification = if matches!(
            category,
            TaskCategory::TextClassification
                | TaskCategory::ImageClassification
                | TaskCategory::ZeroShotClassification
                | TaskCategory::Reranking
        ) {
            // Default to 2 labels; will be updated if id2label provided
            Some(ClassificationPostprocessor::with_num_labels(2))
        } else {
            None
        };

        // Initialize QA postprocessor
        #[cfg(feature = "qa")]
        let qa = if matches!(category, TaskCategory::QuestionAnswering) {
            Some(QAPostprocessor::new())
        } else {
            None
        };

        // Initialize embedding postprocessor
        #[cfg(feature = "embeddings")]
        let embeddings = if matches!(
            category,
            TaskCategory::FeatureExtraction | TaskCategory::SentenceSimilarity
        ) {
            // Use mean pooling for most embedding models
            Some(
                EmbeddingPostprocessor::new()
                    .with_pooling(PoolingStrategy::Mean)
                    .with_normalize(true),
            )
        } else {
            None
        };

        Ok(Self {
            category,
            #[cfg(feature = "text")]
            decoder,
            #[cfg(feature = "classification")]
            classification,
            #[cfg(feature = "qa")]
            qa,
            #[cfg(feature = "embeddings")]
            embeddings,
            id2label: HashMap::new(),
            task_type,
        })
    }

    /// Create an empty postprocessor (raw output passthrough).
    pub fn empty() -> Self {
        Self {
            category: TaskCategory::Generic,
            #[cfg(feature = "text")]
            decoder: None,
            #[cfg(feature = "classification")]
            classification: None,
            #[cfg(feature = "qa")]
            qa: None,
            #[cfg(feature = "embeddings")]
            embeddings: None,
            id2label: HashMap::new(),
            task_type: "generic".to_string(),
        }
    }

    /// Set label mapping for classification tasks.
    #[must_use]
    pub fn with_id2label(mut self, id2label: HashMap<i64, String>) -> Self {
        #[cfg(feature = "classification")]
        {
            self.classification = Some(ClassificationPostprocessor::new(id2label.clone()));
        }
        self.id2label = id2label;
        self
    }

    /// Process raw ONNX outputs into human-readable format.
    pub fn process(
        &self,
        outputs: &HashMap<String, Value>,
        context: &PreprocessContext,
    ) -> PostprocessResult<PostprocessedOutput> {
        debug!(category = ?self.category, "Processing outputs");

        match self.category {
            TaskCategory::TextClassification
            | TaskCategory::ImageClassification
            | TaskCategory::ZeroShotClassification => self.process_classification(outputs, context),

            TaskCategory::Reranking => Self::process_reranking(outputs),

            TaskCategory::QuestionAnswering => self.process_qa(outputs, context),

            TaskCategory::FeatureExtraction | TaskCategory::SentenceSimilarity => {
                self.process_embeddings(outputs, context)
            }

            TaskCategory::TextGeneration
            | TaskCategory::Summarization
            | TaskCategory::Translation
            | TaskCategory::FillMask
            | TaskCategory::AutomaticSpeechRecognition => self.process_text_generation(outputs),

            TaskCategory::TokenClassification => {
                Ok(Self::process_token_classification(outputs, context))
            }

            TaskCategory::ObjectDetection => Ok(Self::process_object_detection(outputs)),

            TaskCategory::TextToSpeech | TaskCategory::Generic => {
                Ok(Self::process_generic(outputs))
            }
        }
    }

    /// Check if task needs text decoder.
    fn needs_decoder(category: TaskCategory) -> bool {
        matches!(
            category,
            TaskCategory::TextGeneration
                | TaskCategory::Summarization
                | TaskCategory::Translation
                | TaskCategory::FillMask
                | TaskCategory::AutomaticSpeechRecognition
                | TaskCategory::QuestionAnswering
        )
    }

    // ========================================================================
    // Task-specific processing
    // ========================================================================

    fn process_classification(
        &self,
        outputs: &HashMap<String, Value>,
        _context: &PreprocessContext,
    ) -> PostprocessResult<PostprocessedOutput> {
        #[cfg(feature = "classification")]
        {
            let processor = self.classification.as_ref().ok_or_else(|| {
                PostprocessError::Config("Classification processor not initialized".into())
            })?;

            // Get logits from output
            let logits = Self::extract_logits(outputs)?;

            // For batch output, take first item
            let logits = if logits.len() == 1 {
                logits.into_iter().next().unwrap()
            } else {
                logits.into_iter().flatten().collect()
            };

            let result = processor.process(&logits)?;
            Ok(PostprocessedOutput::Classification(result))
        }

        #[cfg(not(feature = "classification"))]
        Ok(Self::process_generic(outputs))
    }

    fn process_reranking(
        outputs: &HashMap<String, Value>,
    ) -> PostprocessResult<PostprocessedOutput> {
        // Reranking typically outputs a single score
        let logits = Self::extract_logits(outputs)?;

        // Get the first logit as the relevance score
        let score = logits
            .first()
            .and_then(|row| row.first())
            .copied()
            .ok_or_else(|| {
                PostprocessError::InvalidOutput("Reranking output has no score value".into())
            })?;

        // Convert to probability with sigmoid
        let probability = 1.0 / (1.0 + (-score).exp());

        Ok(PostprocessedOutput::Classification(
            ClassificationOutput::new("relevant", probability),
        ))
    }

    fn process_qa(
        &self,
        outputs: &HashMap<String, Value>,
        context: &PreprocessContext,
    ) -> PostprocessResult<PostprocessedOutput> {
        #[cfg(feature = "qa")]
        {
            let processor = self
                .qa
                .as_ref()
                .ok_or_else(|| PostprocessError::Config("QA processor not initialized".into()))?;

            // Extract start and end logits
            let start_logits = Self::extract_tensor_1d(outputs, "start_logits")?;
            let end_logits = Self::extract_tensor_1d(outputs, "end_logits")?;

            // If we have context and offset mapping, extract the answer text
            if let (Some(ref ctx), Some(ref offsets)) = (&context.context, &context.offset_mapping)
            {
                let result = processor.process(&start_logits, &end_logits, ctx, offsets)?;
                return Ok(PostprocessedOutput::QuestionAnswer(result));
            }

            // Otherwise, just return indices
            let (start, end, score) = processor.process_indices(&start_logits, &end_logits)?;
            Ok(PostprocessedOutput::QuestionAnswer(
                QAOutput::new("", score).with_span(start, end),
            ))
        }

        #[cfg(not(feature = "qa"))]
        Ok(Self::process_generic(outputs))
    }

    fn process_embeddings(
        &self,
        outputs: &HashMap<String, Value>,
        context: &PreprocessContext,
    ) -> PostprocessResult<PostprocessedOutput> {
        #[cfg(feature = "embeddings")]
        {
            let processor = self.embeddings.as_ref().ok_or_else(|| {
                PostprocessError::Config("Embeddings processor not initialized".into())
            })?;

            // Get hidden states - usually "last_hidden_state"
            let hidden_states = Self::extract_hidden_states(outputs)?;

            let result = processor.process(&hidden_states, context.attention_mask.as_deref())?;
            Ok(PostprocessedOutput::Embedding(result))
        }

        #[cfg(not(feature = "embeddings"))]
        Ok(Self::process_generic(outputs))
    }

    fn process_text_generation(
        &self,
        outputs: &HashMap<String, Value>,
    ) -> PostprocessResult<PostprocessedOutput> {
        #[cfg(feature = "text")]
        {
            if let Some(ref decoder) = self.decoder {
                // Try to get generated token IDs or decode from logits
                if let Some(token_ids) = outputs.get("generated_ids") {
                    let ids = Self::json_to_i64_vec(token_ids)?;
                    let result = decoder.decode(&ids)?;
                    return Ok(PostprocessedOutput::Text(result));
                }

                // Try logits
                if let Ok(logits) = Self::extract_logits(outputs) {
                    let result = decoder.decode_from_logits(&logits)?;
                    return Ok(PostprocessedOutput::Text(result));
                }
            }

            // Fallback to generic
            Ok(Self::process_generic(outputs))
        }

        #[cfg(not(feature = "text"))]
        Ok(Self::process_generic(outputs))
    }

    fn process_token_classification(
        outputs: &HashMap<String, Value>,
        _context: &PreprocessContext,
    ) -> PostprocessedOutput {
        // For now, return generic output
        // Full implementation would extract entities from token logits
        Self::process_generic(outputs)
    }

    fn process_object_detection(outputs: &HashMap<String, Value>) -> PostprocessedOutput {
        // For now, return generic output
        // Full implementation would extract bounding boxes
        Self::process_generic(outputs)
    }

    fn process_generic(outputs: &HashMap<String, Value>) -> PostprocessedOutput {
        PostprocessedOutput::Generic(GenericOutput::new(outputs.clone()))
    }

    // ========================================================================
    // Helper methods for extracting tensors
    // ========================================================================

    /// Extract logits from outputs (handles various output names).
    fn extract_logits(outputs: &HashMap<String, Value>) -> PostprocessResult<Vec<Vec<f32>>> {
        // Try common output names
        for name in ["logits", "output", "scores", "predictions"] {
            if let Some(value) = outputs.get(name) {
                return Self::json_to_f32_2d(value);
            }
        }

        Err(PostprocessError::MissingOutput(
            "No logits found in outputs".into(),
        ))
    }

    /// Extract 1D tensor by name.
    fn extract_tensor_1d(
        outputs: &HashMap<String, Value>,
        name: &str,
    ) -> PostprocessResult<Vec<f32>> {
        let value = outputs
            .get(name)
            .ok_or_else(|| PostprocessError::MissingOutput(name.to_string()))?;

        Self::json_to_f32_1d(value)
    }

    /// Extract hidden states (3D tensor: [batch, seq, hidden]).
    fn extract_hidden_states(outputs: &HashMap<String, Value>) -> PostprocessResult<Vec<Vec<f32>>> {
        // Try common names
        for name in ["last_hidden_state", "hidden_states", "embeddings"] {
            if let Some(value) = outputs.get(name) {
                // If 3D, take first batch
                if let Ok(batch) = Self::json_to_f32_3d(value) {
                    if let Some(first) = batch.into_iter().next() {
                        return Ok(first);
                    }
                }
                // If 2D, return directly
                if let Ok(states) = Self::json_to_f32_2d(value) {
                    return Ok(states);
                }
            }
        }

        Err(PostprocessError::MissingOutput(
            "No hidden states found".into(),
        ))
    }

    /// Convert JSON value to 1D f32 vector.
    fn json_to_f32_1d(value: &Value) -> PostprocessResult<Vec<f32>> {
        match value {
            Value::Array(arr) => {
                // Handle nested array (take first)
                if let Some(first) = arr.first() {
                    if first.is_array() {
                        return Self::json_to_f32_1d(first);
                    }
                }

                arr.iter()
                    .map(|v| {
                        v.as_f64()
                            .map(|f| {
                                #[allow(clippy::cast_possible_truncation)]
                                {
                                    f as f32
                                }
                            })
                            .ok_or_else(|| PostprocessError::InvalidOutput("Expected float".into()))
                    })
                    .collect()
            }
            _ => Err(PostprocessError::InvalidOutput("Expected array".into())),
        }
    }

    /// Convert JSON value to 2D f32 vector.
    fn json_to_f32_2d(value: &Value) -> PostprocessResult<Vec<Vec<f32>>> {
        match value {
            Value::Array(arr) => arr.iter().map(Self::json_to_f32_1d).collect(),
            _ => Err(PostprocessError::InvalidOutput("Expected 2D array".into())),
        }
    }

    /// Convert JSON value to 3D f32 vector.
    fn json_to_f32_3d(value: &Value) -> PostprocessResult<Vec<Vec<Vec<f32>>>> {
        match value {
            Value::Array(arr) => arr.iter().map(Self::json_to_f32_2d).collect(),
            _ => Err(PostprocessError::InvalidOutput("Expected 3D array".into())),
        }
    }

    /// Convert JSON value to i64 vector.
    fn json_to_i64_vec(value: &Value) -> PostprocessResult<Vec<i64>> {
        match value {
            Value::Array(arr) => {
                // Handle nested array
                if let Some(first) = arr.first() {
                    if first.is_array() {
                        return Self::json_to_i64_vec(first);
                    }
                }

                arr.iter()
                    .map(|v| {
                        v.as_i64().ok_or_else(|| {
                            PostprocessError::InvalidOutput("Expected integer".into())
                        })
                    })
                    .collect()
            }
            _ => Err(PostprocessError::InvalidOutput("Expected array".into())),
        }
    }
}

impl std::fmt::Debug for Postprocessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Postprocessor")
            .field("category", &self.category)
            .field("task_type", &self.task_type)
            .finish_non_exhaustive()
    }
}
