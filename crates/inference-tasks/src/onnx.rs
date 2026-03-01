//! Generic ONNX Runtime task.
//!
//! This is the main inference task that can run any ONNX model.
//! Accepts raw input from LangChain (text, S3 URIs, base64) and handles
//! preprocessing and postprocessing internally.
//!
//! ## Input Format (from LangChain)
//!
//! Text embedding:
//! ```json
//! {"text": "Hello world"}
//! ```
//!
//! Text with prefix (E5-style):
//! ```json
//! {"text": "What is the capital of France?", "prefix": "query: "}
//! ```
//!
//! Question answering:
//! ```json
//! {"question": "What is Paris?", "context": "Paris is the capital of France."}
//! ```
//!
//! Image classification:
//! ```json
//! {"image": "s3://bucket/image.jpg"}
//! ```
//!
//! Audio (ASR):
//! ```json
//! {"audio": "s3://bucket/speech.wav", "language": "en"}
//! ```
//!
//! Vision-language (VQA):
//! ```json
//! {"image": "s3://bucket/img.jpg", "text": "What is in this image?"}
//! ```
//!
//! ## Output Format (depends on task type)
//!
//! Text classification:
//! ```json
//! {"label": "positive", "score": 0.95}
//! ```
//!
//! Feature extraction (embeddings):
//! ```json
//! {"embedding": [0.1, 0.2, ...], "dimensions": 768}
//! ```
//!
//! Question answering:
//! ```json
//! {"answer": "Paris", "score": 0.87, "start": 10, "end": 15}
//! ```
//!
//! Text generation/ASR:
//! ```json
//! {"text": "generated text"}
//! ```
//!
//! Generic (fallback):
//! ```json
//! {"outputs": {"logits": [[...]]}}
//! ```

use crate::error::{TaskError, TaskResult};
use crate::session::{load_session_from_bytes, load_session_from_file};
use crate::tensor_utils::{
    array_f32_to_json, array_i64_to_json, contains_floats, json_to_array_f32, json_to_array_i64,
    TensorValue,
};
use async_trait::async_trait;
use inference_core::Config;
use inference_grpc::task::{Task, TaskResult as GrpcTaskResult};
use ndarray::ArrayD;
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, error, info};

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

#[cfg(feature = "postprocess")]
use inference_postprocess::postprocessor::{Postprocessor, PreprocessContext};

/// Context data extracted during preprocessing, needed for postprocessing.
#[derive(Debug, Clone, Default)]
struct PreprocessContextData {
    /// Original input text (for QA answer extraction)
    original_text: Option<String>,
    /// Original context (for QA)
    context: Option<String>,
    /// Candidate labels for zero-shot classification
    candidate_labels: Option<Vec<String>>,
    /// Offset mapping from tokens to character positions (for QA)
    offset_mapping: Option<Vec<(usize, usize)>>,
}

/// Input payload for ONNX inference.
///
/// Supports two modes:
/// 1. Raw input (preferred): {"text": "Hello"} - preprocessed internally
/// 2. Direct tensors (legacy): {"inputs": {"input_ids": [[101, ...]]}} - passed directly
#[derive(Debug, Deserialize)]
pub struct OnnxInput {
    /// Direct tensor inputs (legacy mode, bypasses preprocessing).
    /// Keys must match the model's input names.
    #[serde(default)]
    pub inputs: Option<HashMap<String, serde_json::Value>>,
}

/// Output from ONNX inference.
#[derive(Debug, Serialize)]
pub struct OnnxOutput {
    /// Named outputs as multi-dimensional arrays.
    pub outputs: HashMap<String, serde_json::Value>,
}

/// Generic ONNX task that can run any model.
///
/// Handles preprocessing and postprocessing internally - accepts raw text/image/audio input,
/// converts to tensors, runs ONNX inference, and converts outputs to human-readable format.
pub struct OnnxTask {
    name: String,
    session: Arc<Mutex<Session>>,
    input_names: Vec<String>,
    output_names: Vec<String>,
    /// Preprocessor for converting raw input to tensors
    #[cfg(feature = "preprocess")]
    preprocessor: Arc<Preprocessor>,
    /// Postprocessor for converting raw outputs to human-readable format
    #[cfg(feature = "postprocess")]
    postprocessor: Arc<Postprocessor>,
}

impl OnnxTask {
    /// Load ONNX model from a file path with preprocessor and postprocessor.
    #[cfg(all(feature = "preprocess", feature = "postprocess"))]
    pub fn from_file(
        path: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
        postprocessor: Arc<Postprocessor>,
    ) -> TaskResult<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading ONNX model");

        let session = load_session_from_file(path, config)?;
        Self::from_session(session, name, preprocessor, postprocessor)
    }

    /// Load ONNX model from a file path (preprocess only, no postprocess).
    #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
    pub fn from_file(
        path: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading ONNX model");

        let session = load_session_from_file(path, config)?;
        Self::from_session(session, name, preprocessor)
    }

    /// Load ONNX model from bytes with preprocessor and postprocessor.
    #[cfg(all(feature = "preprocess", feature = "postprocess"))]
    pub fn from_bytes(
        bytes: &[u8],
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
        postprocessor: Arc<Postprocessor>,
    ) -> TaskResult<Self> {
        info!(size = bytes.len(), "Loading ONNX model from memory");

        let session = load_session_from_bytes(bytes, config)?;
        Self::from_session(session, name, preprocessor, postprocessor)
    }

    /// Load ONNX model from bytes (preprocess only).
    #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
    pub fn from_bytes(
        bytes: &[u8],
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        info!(size = bytes.len(), "Loading ONNX model from memory");

        let session = load_session_from_bytes(bytes, config)?;
        Self::from_session(session, name, preprocessor)
    }

    /// Create from config - loads model from configured path.
    ///
    /// NOTE: This is a synchronous convenience method that creates no-op processors.
    /// For full preprocessing/postprocessing support, use `TaskRegistry::create()` instead.
    #[cfg(feature = "preprocess")]
    pub fn from_config(config: &Config) -> TaskResult<Self> {
        let model_path = config
            .model_path
            .as_ref()
            .ok_or_else(|| TaskError::Config("MAIIA_AI_MODEL_PATH required".into()))?;

        let path = if let Some(ref onnx_file) = config.onnx_file {
            std::path::PathBuf::from(model_path).join(onnx_file)
        } else {
            std::path::PathBuf::from(model_path)
        };

        // Create minimal processors without async initialization
        // Real usage should go through TaskRegistry::create()
        let preprocessor = Arc::new(Preprocessor::empty());

        #[cfg(feature = "postprocess")]
        let postprocessor = Arc::new(Postprocessor::empty());

        #[cfg(feature = "postprocess")]
        return Self::from_file(
            path,
            config.effective_task_name(),
            config,
            preprocessor,
            postprocessor,
        );

        #[cfg(not(feature = "postprocess"))]
        Self::from_file(path, config.effective_task_name(), config, preprocessor)
    }

    /// Create from an existing session (with postprocessor).
    #[cfg(all(feature = "preprocess", feature = "postprocess"))]
    #[allow(clippy::unnecessary_wraps)]
    fn from_session(
        session: Session,
        name: impl Into<String>,
        preprocessor: Arc<Preprocessor>,
        postprocessor: Arc<Postprocessor>,
    ) -> TaskResult<Self> {
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        info!(
            inputs = ?input_names,
            outputs = ?output_names,
            "ONNX model loaded"
        );

        Ok(Self {
            name: name.into(),
            session: Arc::new(Mutex::new(session)),
            input_names,
            output_names,
            preprocessor,
            postprocessor,
        })
    }

    /// Create from an existing session (preprocess only).
    #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
    #[allow(clippy::unnecessary_wraps)]
    fn from_session(
        session: Session,
        name: impl Into<String>,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        info!(
            inputs = ?input_names,
            outputs = ?output_names,
            "ONNX model loaded"
        );

        Ok(Self {
            name: name.into(),
            session: Arc::new(Mutex::new(session)),
            input_names,
            output_names,
            preprocessor,
        })
    }

    /// Get input tensor names.
    pub fn input_names(&self) -> &[String] {
        &self.input_names
    }

    /// Get output tensor names.
    pub fn output_names(&self) -> &[String] {
        &self.output_names
    }

    /// Parse JSON value to TensorValue (auto-detects type).
    fn json_to_tensor_value(value: &serde_json::Value) -> TaskResult<TensorValue> {
        if contains_floats(value) {
            Ok(TensorValue::Float32(json_to_array_f32(value)?))
        } else {
            Ok(TensorValue::Int64(json_to_array_i64(value)?))
        }
    }

    /// Run inference with mixed-type inputs.
    ///
    /// Uses `spawn_blocking` to offload CPU-bound ONNX inference to a separate thread pool,
    /// preventing blocking of the async runtime.
    async fn run_mixed(
        &self,
        inputs: Vec<(String, TensorValue)>,
    ) -> TaskResult<HashMap<String, serde_json::Value>> {
        use ort::session::input::SessionInputValue;

        // Clone what we need for the blocking task
        let session = Arc::clone(&self.session);
        let output_names = self.output_names.clone();

        // Move inputs into the blocking task
        let result = tokio::task::spawn_blocking(move || {
            // We need to create tensors that live long enough for the session run.
            // Store the arrays separately so references remain valid.
            let mut f32_arrays: Vec<(String, ArrayD<f32>)> = Vec::new();
            let mut i64_arrays: Vec<(String, ArrayD<i64>)> = Vec::new();

            for (name, value) in inputs {
                match value {
                    TensorValue::Float32(arr) => f32_arrays.push((name, arr)),
                    TensorValue::Int64(arr) => i64_arrays.push((name, arr)),
                }
            }

            // Create tensor references and SessionInputValues
            let mut input_values: Vec<(&str, SessionInputValue)> = Vec::new();

            for (name, arr) in &f32_arrays {
                let tensor = TensorRef::from_array_view(arr).map_err(|e| {
                    TaskError::Inference(format!("Failed to create f32 tensor '{name}': {e}"))
                })?;
                input_values.push((name.as_str(), tensor.into_dyn().into()));
            }

            for (name, arr) in &i64_arrays {
                let tensor = TensorRef::from_array_view(arr).map_err(|e| {
                    TaskError::Inference(format!("Failed to create i64 tensor '{name}': {e}"))
                })?;
                input_values.push((name.as_str(), tensor.into_dyn().into()));
            }

            // Run inference with blocking mutex (we're in a blocking context)
            // Use try_lock in a loop with yield, or just block since we're in spawn_blocking
            let mut session = session.blocking_lock();
            let outputs = session
                .run(input_values)
                .map_err(|e| TaskError::Inference(e.to_string()))?;

            // Extract outputs as f32 (most common output type)
            let mut result = HashMap::new();
            for name in &output_names {
                if let Some(value) = outputs.get(name.as_str()) {
                    // Try f32 first (most common), then i64
                    if let Ok(array) = value.try_extract_array::<f32>() {
                        result.insert(name.clone(), array_f32_to_json(&array.into_owned()));
                    } else if let Ok(array) = value.try_extract_array::<i64>() {
                        // Convert i64 array to JSON
                        let json = array_i64_to_json(&array.into_owned());
                        result.insert(name.clone(), json);
                    } else {
                        return Err(TaskError::Inference(format!(
                            "Unsupported output type for '{name}'"
                        )));
                    }
                }
            }

            Ok::<_, TaskError>(result)
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))??;

        Ok(result)
    }

    /// Preprocess raw input or passthrough direct tensors.
    ///
    /// This method handles two input formats:
    /// 1. **Raw input** (new): `{"text": "Hello"}` → preprocessed to tensors
    /// 2. **Direct tensors** (legacy): `{"inputs": {"input_ids": [...]}}` → passed through
    ///
    /// Returns tensor inputs ready for ONNX inference plus context for postprocessing.
    #[cfg(feature = "preprocess")]
    async fn preprocess_or_passthrough(
        &self,
        payload: &str,
    ) -> Result<(HashMap<String, serde_json::Value>, PreprocessContextData), TaskError> {
        debug!(
            payload_len = payload.len(),
            payload_preview = &payload[..payload.len().min(200)],
            "preprocess_or_passthrough received payload"
        );

        // First, try to parse as OnnxInput to check for direct tensor format
        if let Ok(direct_input) = serde_json::from_str::<OnnxInput>(payload) {
            if let Some(inputs) = direct_input.inputs {
                // Direct tensor input - passthrough (no context available)
                debug!("Using direct tensor input (legacy mode)");
                return Ok((inputs, PreprocessContextData::default()));
            }
        }

        // Try preprocessing as raw input
        debug!("Preprocessing raw input");

        // Parse input to extract context for postprocessing
        let mut context_data = Self::extract_preprocess_context(payload);

        let preprocessed = self
            .preprocessor
            .process_json(payload)
            .await
            .map_err(|e| TaskError::InvalidInput(format!("Preprocessing failed: {e}")))?;

        // Extract offset mapping from preprocessed output (for QA tasks)
        context_data.offset_mapping = preprocessed.offset_mapping;

        // Extract the inputs map from the preprocessed output
        match preprocessed.inputs {
            serde_json::Value::Object(map) => Ok((map.into_iter().collect(), context_data)),
            _ => Err(TaskError::InvalidInput(
                "Preprocessing did not return an object".into(),
            )),
        }
    }

    /// Extract context data from raw input for postprocessing.
    fn extract_preprocess_context(payload: &str) -> PreprocessContextData {
        let mut context = PreprocessContextData::default();

        if let Ok(value) = serde_json::from_str::<serde_json::Value>(payload) {
            if let Some(obj) = value.as_object() {
                // Extract text for text generation tasks
                if let Some(text) = obj.get("text").and_then(|v| v.as_str()) {
                    context.original_text = Some(text.to_string());
                }

                // Extract context for QA tasks
                if let Some(ctx) = obj.get("context").and_then(|v| v.as_str()) {
                    context.context = Some(ctx.to_string());
                }

                // Extract question for QA tasks
                if let Some(q) = obj.get("question").and_then(|v| v.as_str()) {
                    context.original_text = Some(q.to_string());
                }

                // Extract candidate labels for zero-shot classification
                if let Some(labels) = obj.get("candidate_labels").and_then(|v| v.as_array()) {
                    context.candidate_labels = Some(
                        labels
                            .iter()
                            .filter_map(|l| l.as_str().map(String::from))
                            .collect(),
                    );
                }
            }
        }

        context
    }
}

#[async_trait]
impl Task for OnnxTask {
    fn name(&self) -> &str {
        &self.name
    }

    #[cfg(all(feature = "preprocess", feature = "postprocess"))]
    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "ONNX task executing");

        // Preprocess raw input (returns tensors + context for postprocessing)
        let (tensor_inputs, context_data) = match self.preprocess_or_passthrough(payload).await {
            Ok(result) => result,
            Err(e) => {
                error!(error = %e, "Preprocessing failed");
                return GrpcTaskResult::err(format!("Preprocessing failed: {e}"));
            }
        };

        // Extract attention mask from inputs for postprocessing (needed for mean pooling)
        let attention_mask = tensor_inputs.get("attention_mask").and_then(|v| {
            json_to_array_i64(v)
                .ok()
                .map(|arr: ArrayD<i64>| arr.into_raw_vec_and_offset().0)
        });

        // Convert JSON inputs to typed tensors
        let mut tensors: Vec<(String, TensorValue)> = Vec::new();
        for name in &self.input_names {
            let Some(value) = tensor_inputs.get(name) else {
                return GrpcTaskResult::err(format!("Missing input: {name}"));
            };

            match Self::json_to_tensor_value(value) {
                Ok(tensor) => tensors.push((name.clone(), tensor)),
                Err(e) => {
                    return GrpcTaskResult::err(format!("Invalid input '{name}': {e}"));
                }
            }
        }

        // Run inference
        let outputs = match self.run_mixed(tensors).await {
            Ok(o) => o,
            Err(e) => {
                error!(error = %e, "Inference failed");
                return GrpcTaskResult::err(e.to_string());
            }
        };

        // Build postprocessing context
        let postprocess_context = PreprocessContext {
            original_text: context_data.original_text,
            context: context_data.context,
            offset_mapping: context_data.offset_mapping,
            input_ids: None,
            attention_mask,
            candidate_labels: context_data.candidate_labels,
        };

        // Apply postprocessing to convert raw outputs to human-readable format
        match self.postprocessor.process(&outputs, &postprocess_context) {
            Ok(processed) => match serde_json::to_string(&processed) {
                Ok(json) => GrpcTaskResult::ok(json),
                Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
            },
            Err(e) => {
                // Fallback to raw outputs if postprocessing fails
                debug!(error = %e, "Postprocessing failed, returning raw outputs");
                let output = OnnxOutput { outputs };
                match serde_json::to_string(&output) {
                    Ok(json) => GrpcTaskResult::ok(json),
                    Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
                }
            }
        }
    }

    #[cfg(all(feature = "preprocess", not(feature = "postprocess")))]
    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "ONNX task executing");

        // Preprocess raw input
        let (tensor_inputs, _context_data) = match self.preprocess_or_passthrough(payload).await {
            Ok(result) => result,
            Err(e) => {
                error!(error = %e, "Preprocessing failed");
                return GrpcTaskResult::err(format!("Preprocessing failed: {e}"));
            }
        };

        // Convert JSON inputs to typed tensors
        let mut tensors: Vec<(String, TensorValue)> = Vec::new();
        for name in &self.input_names {
            let Some(value) = tensor_inputs.get(name) else {
                return GrpcTaskResult::err(format!("Missing input: {name}"));
            };

            match Self::json_to_tensor_value(value) {
                Ok(tensor) => tensors.push((name.clone(), tensor)),
                Err(e) => {
                    return GrpcTaskResult::err(format!("Invalid input '{name}': {e}"));
                }
            }
        }

        // Run inference
        let outputs = match self.run_mixed(tensors).await {
            Ok(o) => o,
            Err(e) => {
                error!(error = %e, "Inference failed");
                return GrpcTaskResult::err(e.to_string());
            }
        };

        // Return raw outputs (no postprocessing)
        let output = OnnxOutput { outputs };
        match serde_json::to_string(&output) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
        }
    }

    fn is_ready(&self) -> bool {
        true
    }
}
