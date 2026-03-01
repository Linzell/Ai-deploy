//! Error types for postprocessing operations.

use thiserror::Error;

/// Errors that can occur during postprocessing.
#[derive(Debug, Error)]
pub enum PostprocessError {
    /// Invalid output from model
    #[error("Invalid output: {0}")]
    InvalidOutput(String),

    /// Missing required output tensor
    #[error("Missing output: {0}")]
    MissingOutput(String),

    /// Decoding error
    #[error("Decoding error: {0}")]
    Decoding(String),

    /// Configuration error
    #[error("Configuration error: {0}")]
    Config(String),

    /// JSON serialization error
    #[error("JSON error: {0}")]
    Json(String),

    /// Shape mismatch
    #[error("Shape mismatch: {0}")]
    Shape(String),
}

/// Result type for postprocessing operations.
pub type PostprocessResult<T> = Result<T, PostprocessError>;
