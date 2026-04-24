//! Model trait for inference.

use async_trait::async_trait;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

use crate::Result;

/// Input for model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelInput {
    /// Text input (for LLMs, embeddings)
    Text(String),

    /// Binary input (for audio, images)
    Binary(Bytes),

    /// Multiple text inputs (for batch processing)
    TextBatch(Vec<String>),

    /// Multiple binary inputs (for batch processing)
    BinaryBatch(Vec<Bytes>),

    /// Structured input (for complex models)
    Structured(serde_json::Value),
}

/// Output from model inference.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ModelOutput {
    /// Text output (for LLMs)
    Text(String),

    /// Binary output (for image generation)
    Binary(Bytes),

    /// Embedding output
    Embedding(Vec<f32>),

    /// Multiple text outputs
    TextBatch(Vec<String>),

    /// Multiple embeddings
    EmbeddingBatch(Vec<Vec<f32>>),

    /// Structured output (for object detection, etc.)
    Structured(serde_json::Value),
}

/// Trait for all inference models.
///
/// This trait provides a common interface for all model types,
/// allowing the gRPC service to be generic over models.
#[async_trait]
pub trait Model: Send + Sync {
    /// Load the model from the given path.
    async fn load(model_path: std::path::PathBuf) -> Result<Self>
    where
        Self: Sized;

    /// Run inference on the input.
    async fn infer(&self, input: ModelInput) -> Result<ModelOutput>;

    /// Get the model name.
    fn name(&self) -> &str;

    /// Check if the model is ready for inference.
    fn is_ready(&self) -> bool;

    /// Warm up the model (run a dummy inference).
    async fn warmup(&self) -> Result<()> {
        Ok(())
    }
}

/// Model metadata for service discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelMetadata {
    pub name: String,
    pub version: String,
    pub model_type: String,
    pub input_types: Vec<String>,
    pub output_types: Vec<String>,
}

/// Available models in the system.
pub fn get_available_models() -> Vec<ModelMetadata> {
    vec![
        ModelMetadata {
            name: "maiia.chat-completion.v1".to_string(),
            version: "1.0.0".to_string(),
            model_type: "chat-completion".to_string(),
            input_types: vec!["text".to_string()],
            output_types: vec!["text".to_string()],
        },
        ModelMetadata {
            name: "maiia.completion.v1".to_string(),
            version: "1.0.0".to_string(),
            model_type: "completion".to_string(),
            input_types: vec!["text".to_string()],
            output_types: vec!["text".to_string()],
        },
        ModelMetadata {
            name: "maiia.feature-extraction.v1".to_string(),
            version: "1.0.0".to_string(),
            model_type: "feature-extraction".to_string(),
            input_types: vec!["text".to_string()],
            output_types: vec!["embedding".to_string()],
        },
    ]
}
