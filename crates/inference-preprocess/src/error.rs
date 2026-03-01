//! Error types for preprocessing.

use thiserror::Error;

/// Preprocessing error types.
#[derive(Debug, Error)]
pub enum PreprocessError {
    #[error("Tokenizer error: {0}")]
    Tokenizer(String),

    #[error("Image processing error: {0}")]
    Image(String),

    #[error("Audio processing error: {0}")]
    Audio(String),

    #[error("Failed to load file: {0}")]
    FileLoad(String),

    #[error("Invalid input: {0}")]
    InvalidInput(String),

    #[error("S3 error: {0}")]
    S3(String),

    #[error("Configuration error: {0}")]
    Config(String),

    #[error("Unsupported input type for this preprocessor")]
    UnsupportedInput,
}

/// Result type for preprocessing operations.
pub type PreprocessResult<T> = Result<T, PreprocessError>;

impl From<inference_core::Error> for PreprocessError {
    fn from(e: inference_core::Error) -> Self {
        PreprocessError::Config(e.to_string())
    }
}
