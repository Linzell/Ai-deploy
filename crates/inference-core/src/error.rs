//! Error types for the inference service.

use thiserror::Error;

/// Unified error type for all inference operations.
#[derive(Error, Debug)]
pub enum Error {
    // === Core errors ===
    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Loader error: {0}")]
    Loader(String),

    #[error("Model error: {0}")]
    Model(String),

    #[error("Inference error: {0}")]
    Inference(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(String),

    #[error("Not found: {0}")]
    NotFound(String),

    #[error("Unauthorized: {0}")]
    Unauthorized(String),

    // === Task-specific errors (merged from TaskError) ===
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Failed to load ONNX model: {0}")]
    OnnxLoad(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Alias for task-specific results (for backward compatibility during migration)
pub type TaskResult<T> = std::result::Result<T, Error>;
