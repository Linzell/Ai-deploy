//! llama.cpp-based inference backend for GGUF models.
//!
//! This crate provides text generation using llama.cpp via the `llama-cpp-2` bindings.
//! It supports GGUF model files with various quantization formats (Q4_K_M, Q8, F16, etc.)
//! and hardware acceleration via Metal (macOS) or CUDA (Linux/Windows).
//!
//! ## When to use llama.cpp?
//!
//! - Models only available in GGUF format
//! - Custom architectures not supported by Candle
//! - Models requiring exotic quantization (MXFP4, etc.)
//! - When ONNX exports use unsupported operators
//!
//! ## Features
//!
//! - `metal` - Enable Metal acceleration on macOS
//! - `cuda` - Enable CUDA acceleration on Linux/Windows
//!
//! ## Example
//!
//! ```rust,ignore
//! use inference_llama::LlamaTextGenTask;
//! use inference_core::Config;
//!
//! let config = Config::default();
//! let task = LlamaTextGenTask::from_gguf_path(
//!     "/path/to/model.gguf",
//!     "my-llama-task",
//!     &config,
//! )?;
//!
//! let result = task.execute(r#"{"text": "Hello, world!"}"#, "req-1").await;
//! ```

pub mod error;
pub mod text_gen;

// Re-export main types
pub use error::{TaskError, TaskResult};
pub use text_gen::{LlamaGenConfig, LlamaTextGenInput, LlamaTextGenOutput, LlamaTextGenTask};
