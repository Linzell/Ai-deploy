//! # Inference Tasks
//!
//! Facade crate that provides task implementations for the inference service,
//! re-exporting from specialized backend crates.
//!
//! ## Architecture
//!
//! This crate acts as a facade, providing a unified API while delegating
//! to specialized backend crates:
//!
//! - **`inference-onnx`**: ONNX Runtime tasks (embeddings, classification, NER, OCR)
//! - **`inference-candle`**: Candle-based tasks (text-gen, seq2seq, TTS)
//! - **`inference-llama`**: llama.cpp tasks (GGUF models)
//!
//! ## Backend Selection
//!
//! Configuration determines which backend and model to use - no code changes required.
//! The `backend` config option controls selection:
//!
//! - `auto` (default): Candle for text-generation, ONNX for everything else
//! - `candle`: Force Candle backend
//! - `llama`: Force llama.cpp backend (requires gguf_file config)
//! - `onnx`: Force ONNX backend
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_tasks::TaskRegistry;
//! use inference_core::Config;
//!
//! // Load from config - backend selected automatically
//! let config = Config::load()?;
//! let task = TaskRegistry::create(&config).await?;
//!
//! // Execute task
//! let result = task.execute(r#"{"text": "Hello world"}"#, "req-1").await;
//! ```
//!
//! ## Input Formats
//!
//! Text generation (Candle/Llama backend):
//! ```json
//! {"text": "Once upon a time"}
//! ```
//!
//! Text embedding (ONNX backend):
//! ```json
//! {"text": "What is the capital of France?"}
//! ```

pub mod error;
pub mod registry;

// Echo task (test task, no dependencies)
mod echo;
pub use echo::EchoTask;

// Re-export error types
pub use error::{TaskError, TaskResult};

// Re-export registry
pub use registry::TaskRegistry;

// ============================================================================
// ONNX Backend (always available via inference-onnx)
// ============================================================================

/// ONNX task types re-exported from `inference-onnx`.
pub mod onnx {
    pub use inference_onnx::*;
}

// Re-export main ONNX types at crate root for convenience
pub use inference_onnx::{
    ClipInput, ClipOutput, ClipTask, GenerationConfig, ModelArchitecture, OnnxTask, PaddleOcrInput,
    PaddleOcrOutput, PaddleOcrTask, Seq2SeqInput, Seq2SeqOutput, Seq2SeqTask,
};

// ============================================================================
// Candle Backend (optional, enabled with "candle" feature)
// ============================================================================

/// Candle task types re-exported from `inference-candle`.
#[cfg(feature = "candle")]
pub mod candle {
    pub use inference_candle::*;
}

// Re-export main Candle types at crate root when feature is enabled
#[cfg(feature = "candle")]
pub use inference_candle::{
    CandleGenConfig, CandleSeq2SeqInput, CandleSeq2SeqOutput, CandleSeq2SeqTask, CandleTextGenTask,
    CandleTtsInput, CandleTtsOutput, CandleTtsTask, Seq2SeqArch, Seq2SeqGenConfig,
    TextGenModelArch, TtsGenConfig,
};

// ============================================================================
// Llama.cpp Backend (optional, enabled with "llama" feature)
// ============================================================================

/// Llama.cpp task types re-exported from `inference-llama`.
#[cfg(feature = "llama")]
pub mod llama {
    pub use inference_llama::*;
}

// Re-export main Llama types at crate root when feature is enabled
#[cfg(feature = "llama")]
pub use inference_llama::{
    LlamaGenConfig, LlamaTextGenInput, LlamaTextGenOutput, LlamaTextGenTask,
};

// ============================================================================
// Preprocessing (optional, enabled with "preprocess" feature)
// ============================================================================

// Re-export preprocessing types for convenience
#[cfg(feature = "preprocess")]
pub use inference_preprocess::{PreprocessError, PreprocessResult, Preprocessor, RawInput};
