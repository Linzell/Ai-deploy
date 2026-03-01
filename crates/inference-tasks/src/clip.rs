//! CLIP-based zero-shot image classification.
//!
//! CLIP (Contrastive Language-Image Pre-training) uses dual encoders:
//! - Vision encoder: processes images → image embeddings
//! - Text encoder: processes text labels → text embeddings
//!
//! Classification is performed by computing cosine similarity between
//! the image embedding and each text label embedding.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_tasks::ClipTask;
//! use inference_core::Config;
//!
//! let task = ClipTask::from_model_dir("/path/to/clip", "clip-task", &config)?;
//! let result = task.execute(r#"{"image":"data:image/png;base64,...","candidate_labels":["cat","dog"]}"#, "req-1").await;
//! ```
//!
//! ## Input Format
//!
//! ```json
//! {"image": "s3://bucket/image.jpg", "candidate_labels": ["cat", "dog", "bird"]}
//! {"image": "data:image/jpeg;base64,...", "candidate_labels": ["indoor", "outdoor"]}
//! ```
//!
//! ## Output Format
//!
//! ```json
//! {"labels": ["cat", "dog", "bird"], "scores": [0.85, 0.10, 0.05], "label": "cat", "score": 0.85}
//! ```

use crate::error::{TaskError, TaskResult};
use crate::session::load_session_from_file;
use async_trait::async_trait;
use inference_core::Config;
use inference_grpc::task::{Task, TaskResult as GrpcTaskResult};
use ndarray::{Array2, ArrayD, Axis};
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

/// Input for CLIP zero-shot image classification.
#[derive(Debug, Deserialize)]
pub struct ClipInput {
    /// Image as base64 data URI or S3/HTTP URL
    pub image: String,
    /// Candidate labels to classify against
    pub candidate_labels: Vec<String>,
}

/// Output from CLIP classification.
#[derive(Debug, Serialize)]
pub struct ClipOutput {
    /// All labels
    pub labels: Vec<String>,
    /// Similarity scores for each label (normalized via softmax)
    pub scores: Vec<f32>,
    /// Top label
    pub label: String,
    /// Top score
    pub score: f32,
}

/// CLIP dual-encoder task for zero-shot image classification.
///
/// Loads separate vision and text encoder ONNX models.
pub struct ClipTask {
    name: String,
    vision_session: Arc<Mutex<Session>>,
    text_session: Arc<Mutex<Session>>,
    #[cfg(feature = "preprocess")]
    preprocessor: Arc<Preprocessor>,
}

impl ClipTask {
    /// Create a new CLIP task from a model directory.
    ///
    /// Expects the directory to contain:
    /// - `onnx/vision_model.onnx` or `vision_model.onnx`
    /// - `onnx/text_model.onnx` or `text_model.onnx`
    /// - `tokenizer.json`
    /// - `preprocessor_config.json`
    #[cfg(feature = "preprocess")]
    #[allow(clippy::unused_async)] // Async for API consistency with registry pattern
    pub async fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading CLIP dual-encoder model"
        );

        // Find vision model
        let vision_path =
            Self::find_model_file(model_dir, &["onnx/vision_model.onnx", "vision_model.onnx"])?;
        info!(path = %vision_path.display(), "Loading vision encoder");
        let vision_session = load_session_from_file(&vision_path, config)?;

        // Find text model
        let text_path =
            Self::find_model_file(model_dir, &["onnx/text_model.onnx", "text_model.onnx"])?;
        info!(path = %text_path.display(), "Loading text encoder");
        let text_session = load_session_from_file(&text_path, config)?;

        // Log input names for debugging
        let vision_input_names: Vec<String> = vision_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let text_input_names: Vec<String> = text_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();

        info!(
            vision_inputs = ?vision_input_names,
            text_inputs = ?text_input_names,
            "CLIP encoders loaded"
        );

        Ok(Self {
            name,
            vision_session: Arc::new(Mutex::new(vision_session)),
            text_session: Arc::new(Mutex::new(text_session)),
            preprocessor,
        })
    }

    /// Find a model file from a list of possible paths.
    fn find_model_file(model_dir: &Path, candidates: &[&str]) -> TaskResult<std::path::PathBuf> {
        for candidate in candidates {
            let path = model_dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(TaskError::ModelNotFound(format!(
            "Could not find any of {:?} in {}",
            candidates,
            model_dir.display()
        )))
    }

    /// Encode an image using the vision encoder.
    async fn encode_image(&self, pixel_values: ArrayD<f32>) -> TaskResult<Array2<f32>> {
        use ort::session::input::SessionInputValue;

        let session = Arc::clone(&self.vision_session);

        let result = tokio::task::spawn_blocking(move || {
            let mut session = session.blocking_lock();

            // Create tensor from pixel values
            let tensor = TensorRef::from_array_view(&pixel_values).map_err(|e| {
                TaskError::Inference(format!("Failed to create vision tensor: {e}"))
            })?;

            // Build input values with explicit type
            let input_values: Vec<(&str, SessionInputValue)> =
                vec![("pixel_values", tensor.into_dyn().into())];

            // Run vision encoder
            let outputs = session
                .run(input_values)
                .map_err(|e| TaskError::Inference(format!("Vision encoding failed: {e}")))?;

            // Extract image embeddings (try common output names)
            let embedding = outputs
                .get("image_embeds")
                .or_else(|| outputs.get("last_hidden_state"))
                .or_else(|| outputs.get("pooler_output"))
                .ok_or_else(|| TaskError::Inference("No image embedding output found".into()))?;

            let array = embedding.try_extract_array::<f32>().map_err(|e| {
                TaskError::Inference(format!("Failed to extract vision embedding: {e}"))
            })?;

            // Reshape to 2D [batch, embedding_dim]
            let shape = array.shape();
            let embedding_dim = shape[shape.len() - 1];
            let batch_size = shape.iter().take(shape.len() - 1).product();

            let (flat, _offset) = array.into_owned().into_raw_vec_and_offset();
            let array_2d =
                Array2::from_shape_vec((batch_size, embedding_dim), flat).map_err(|e| {
                    TaskError::Inference(format!("Failed to reshape vision embedding: {e}"))
                })?;

            Ok::<_, TaskError>(array_2d)
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))??;

        Ok(result)
    }

    /// Encode text labels using the text encoder.
    ///
    /// Note: CLIP text encoder only requires input_ids (no attention_mask).
    async fn encode_text(&self, input_ids: ArrayD<i64>) -> TaskResult<Array2<f32>> {
        use ort::session::input::SessionInputValue;

        let session = Arc::clone(&self.text_session);

        let result = tokio::task::spawn_blocking(move || {
            let mut session = session.blocking_lock();

            // Create tensor for input_ids only (CLIP doesn't use attention_mask)
            let input_ids_tensor = TensorRef::from_array_view(&input_ids).map_err(|e| {
                TaskError::Inference(format!("Failed to create input_ids tensor: {e}"))
            })?;

            // Build input values with explicit type
            let input_values: Vec<(&str, SessionInputValue)> =
                vec![("input_ids", input_ids_tensor.into_dyn().into())];

            // Run text encoder
            let outputs = session
                .run(input_values)
                .map_err(|e| TaskError::Inference(format!("Text encoding failed: {e}")))?;

            // Extract text embeddings
            let embedding = outputs
                .get("text_embeds")
                .or_else(|| outputs.get("last_hidden_state"))
                .or_else(|| outputs.get("pooler_output"))
                .ok_or_else(|| TaskError::Inference("No text embedding output found".into()))?;

            let array = embedding.try_extract_array::<f32>().map_err(|e| {
                TaskError::Inference(format!("Failed to extract text embedding: {e}"))
            })?;

            // Reshape to 2D [batch, embedding_dim]
            let shape = array.shape();
            let embedding_dim = shape[shape.len() - 1];
            let batch_size = shape.iter().take(shape.len() - 1).product();

            let (flat, _offset) = array.into_owned().into_raw_vec_and_offset();
            let array_2d =
                Array2::from_shape_vec((batch_size, embedding_dim), flat).map_err(|e| {
                    TaskError::Inference(format!("Failed to reshape text embedding: {e}"))
                })?;

            Ok::<_, TaskError>(array_2d)
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))??;

        Ok(result)
    }

    /// Compute cosine similarity between image and text embeddings.
    fn cosine_similarity(image_emb: &Array2<f32>, text_emb: &Array2<f32>) -> Vec<f32> {
        // Normalize embeddings
        let image_norm = Self::l2_normalize(image_emb);
        let text_norm = Self::l2_normalize(text_emb);

        // Compute dot product (cosine similarity since vectors are normalized)
        // image_emb: [1, D], text_emb: [N, D] -> result: [1, N]
        let similarities = image_norm.dot(&text_norm.t());

        // Return as flat vector
        let (vec, _offset) = similarities.into_raw_vec_and_offset();
        vec
    }

    /// L2 normalize embeddings along the last axis.
    fn l2_normalize(embeddings: &Array2<f32>) -> Array2<f32> {
        let norms = embeddings.map_axis(Axis(1), |row| {
            row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12)
        });

        let mut normalized = embeddings.clone();
        for (i, mut row) in normalized.rows_mut().into_iter().enumerate() {
            row.mapv_inplace(|x| x / norms[i]);
        }
        normalized
    }

    /// Apply softmax to convert similarities to probabilities.
    fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|x| x / temperature).collect();
        let max = scaled.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = scaled.iter().map(|x| (x - max).exp()).sum();
        scaled.iter().map(|x| (x - max).exp() / exp_sum).collect()
    }
}

#[async_trait]
impl Task for ClipTask {
    fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "preprocess")]
    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "CLIP task executing");

        // Parse input
        let input: ClipInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input: {e}")),
        };

        if input.candidate_labels.is_empty() {
            return GrpcTaskResult::err("candidate_labels cannot be empty".to_string());
        }

        debug!(
            num_labels = input.candidate_labels.len(),
            labels = ?input.candidate_labels,
            "Processing CLIP classification"
        );

        // 1. Preprocess image
        let image_input = format!(r#"{{"image": "{}"}}"#, input.image);
        let image_preprocessed = match self.preprocessor.process_json(&image_input).await {
            Ok(p) => p,
            Err(e) => return GrpcTaskResult::err(format!("Image preprocessing failed: {e}")),
        };

        // Extract pixel_values from preprocessed output
        let pixel_values = match image_preprocessed.inputs.get("pixel_values") {
            Some(v) => match crate::tensor_utils::json_to_array_f32(v) {
                Ok(arr) => arr,
                Err(e) => return GrpcTaskResult::err(format!("Invalid pixel_values: {e}")),
            },
            None => {
                return GrpcTaskResult::err("No pixel_values in preprocessed output".to_string())
            }
        };

        // 2. Preprocess text labels (tokenize each label)
        let mut all_input_ids: Vec<Vec<i64>> = Vec::new();
        let mut max_len = 0;

        for label in &input.candidate_labels {
            let text_input = format!(r#"{{"text": "{label}"}}"#);
            let text_preprocessed = match self.preprocessor.process_json(&text_input).await {
                Ok(p) => p,
                Err(e) => {
                    return GrpcTaskResult::err(format!(
                        "Text preprocessing failed for '{label}': {e}"
                    ))
                }
            };

            let input_ids = match text_preprocessed.inputs.get("input_ids") {
                Some(v) => match crate::tensor_utils::json_to_array_i64(v) {
                    Ok(arr) => {
                        let (vec, _offset) = arr.into_raw_vec_and_offset();
                        vec
                    }
                    Err(e) => return GrpcTaskResult::err(format!("Invalid input_ids: {e}")),
                },
                None => {
                    return GrpcTaskResult::err("No input_ids in text preprocessing".to_string())
                }
            };

            max_len = max_len.max(input_ids.len());
            all_input_ids.push(input_ids);
        }

        // Pad to same length (CLIP uses 0 for padding)
        for ids in &mut all_input_ids {
            ids.resize(max_len, 0);
        }

        // Convert to 2D array
        let num_labels = input.candidate_labels.len();
        let input_ids_flat: Vec<i64> = all_input_ids.into_iter().flatten().collect();

        let input_ids_array =
            match ArrayD::from_shape_vec(vec![num_labels, max_len], input_ids_flat) {
                Ok(arr) => arr,
                Err(e) => {
                    return GrpcTaskResult::err(format!("Failed to create input_ids array: {e}"))
                }
            };

        // 3. Encode image
        let image_embedding = match self.encode_image(pixel_values).await {
            Ok(emb) => emb,
            Err(e) => return GrpcTaskResult::err(format!("Image encoding failed: {e}")),
        };

        // 4. Encode text labels (CLIP only needs input_ids, no attention_mask)
        let text_embeddings = match self.encode_text(input_ids_array).await {
            Ok(emb) => emb,
            Err(e) => return GrpcTaskResult::err(format!("Text encoding failed: {e}")),
        };

        // 5. Compute similarities and softmax
        let similarities = Self::cosine_similarity(&image_embedding, &text_embeddings);
        let scores = Self::softmax(&similarities, 0.01); // Low temperature for sharper distribution

        // 6. Find top label
        let (top_idx, &top_score) = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        let output = ClipOutput {
            labels: input.candidate_labels.clone(),
            scores: scores.clone(),
            label: input.candidate_labels[top_idx].clone(),
            score: top_score,
        };

        match serde_json::to_string(&output) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
        }
    }

    fn is_ready(&self) -> bool {
        true
    }
}
