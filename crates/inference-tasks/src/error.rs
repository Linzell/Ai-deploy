//! Error types for task operations.

use thiserror::Error;

/// Task-specific errors.
#[derive(Error, Debug)]
pub enum TaskError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Failed to load ONNX model: {0}")]
    OnnxLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<ort::Error> for TaskError {
    fn from(e: ort::Error) -> Self {
        TaskError::Inference(e.to_string())
    }
}

/// Result type for task operations.
pub type TaskResult<T> = std::result::Result<T, TaskError>;
