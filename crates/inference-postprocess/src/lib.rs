//! # Inference Postprocessing
//!
//! This crate handles all postprocessing for the inference service:
//! - **Text**: Decoding token IDs back to text
//! - **Classification**: Softmax, top-k labels, confidence scores
//! - **QA**: Span extraction, answer decoding
//! - **Embeddings**: Pooling strategies, normalization
//!
//! ## Design
//!
//! The postprocessing accepts raw ONNX outputs (logits, hidden states)
//! and converts them to human-readable formats (text, labels, scores).
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_postprocess::{Postprocessor, PostprocessedOutput};
//!
//! let postprocessor = Postprocessor::from_config(&config).await?;
//!
//! // Process ONNX outputs
//! let raw_outputs = /* from ONNX inference */;
//! let result = postprocessor.process(&raw_outputs, &preprocessor_context)?;
//! // Returns: {"label": "positive", "score": 0.95}
//! ```

pub mod error;
pub mod output;
pub mod postprocessor;

#[cfg(feature = "text")]
pub mod decoder;

#[cfg(feature = "classification")]
pub mod classification;

#[cfg(feature = "qa")]
pub mod qa;

#[cfg(feature = "embeddings")]
pub mod embeddings;

pub use error::{PostprocessError, PostprocessResult};
pub use output::PostprocessedOutput;
pub use postprocessor::Postprocessor;
