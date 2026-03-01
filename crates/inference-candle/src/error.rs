//! Error types for Candle inference tasks.

use thiserror::Error;

/// Task-specific errors for Candle backend.
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

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),

    #[error("Candle error: {0}")]
    Candle(#[from] candle_core::Error),
}

/// Result type for Candle task operations.
pub type TaskResult<T> = std::result::Result<T, TaskError>;
