//! # Inference Preprocessing
//!
//! This crate handles all preprocessing for the inference service:
//! - **Text**: Tokenization using HuggingFace tokenizers
//! - **Image**: Loading, resizing, normalizing for vision models
//! - **Audio**: WAV loading, mel spectrogram for Whisper-style ASR
//!
//! ## Design
//!
//! The preprocessing accepts raw inputs (text strings, S3 URIs, base64 images)
//! and outputs tensors ready for ONNX inference.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_preprocess::{Preprocessor, RawInput};
//!
//! let preprocessor = Preprocessor::from_config(&config).await?;
//!
//! // Text input
//! let input = RawInput::text("What is the capital of France?");
//! let tensors = preprocessor.process(&input).await?;
//!
//! // Image from S3
//! let input = RawInput::image_s3("s3://bucket/image.jpg");
//! let tensors = preprocessor.process(&input).await?;
//! ```

pub mod error;
pub mod input;
pub mod preprocessor;

#[cfg(feature = "text")]
pub mod tokenizer;

#[cfg(feature = "tekken")]
pub mod tekken_tokenizer;

#[cfg(feature = "image")]
pub mod image;

#[cfg(feature = "audio")]
pub mod audio;

pub use error::{PreprocessError, PreprocessResult};
pub use input::RawInput;
pub use preprocessor::Preprocessor;
