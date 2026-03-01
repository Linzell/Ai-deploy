//! Error types for task operations.
//!
//! This module provides a lightweight error type for the facade crate.
//! Backend-specific errors are handled by their respective crates
//! (inference-onnx, inference-candle, inference-llama).

use thiserror::Error;

/// Task-specific errors for the facade crate.
///
/// Note: Backend-specific errors (ONNX, Candle, Llama) are handled
/// by their respective crates. This error type is for registry
/// and general task operations.
#[derive(Error, Debug)]
pub enum TaskError {
    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Failed to load model: {0}")]
    ModelLoad(String),

    #[error("Inference failed: {0}")]
    Inference(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Backend error: {0}")]
    Backend(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type for task operations.
pub type TaskResult<T> = std::result::Result<T, TaskError>;

// ============================================================================
// Backend error conversions
// ============================================================================

/// Convert inference-core errors to TaskError.
/// Note: inference_onnx::TaskError is a re-export of inference_core::Error,
/// so this covers both.
impl From<inference_core::Error> for TaskError {
    fn from(e: inference_core::Error) -> Self {
        TaskError::Backend(e.to_string())
    }
}

/// Convert inference-candle errors to TaskError (when feature enabled)
#[cfg(feature = "candle")]
impl From<inference_candle::TaskError> for TaskError {
    fn from(e: inference_candle::TaskError) -> Self {
        TaskError::Backend(e.to_string())
    }
}

/// Convert inference-llama errors to TaskError (when feature enabled)
#[cfg(feature = "llama")]
impl From<inference_llama::TaskError> for TaskError {
    fn from(e: inference_llama::TaskError) -> Self {
        TaskError::Backend(e.to_string())
    }
}
