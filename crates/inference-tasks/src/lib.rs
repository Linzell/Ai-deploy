//! # Inference Tasks
//!
//! Task implementations for the inference service, supporting both ONNX and Candle backends.
//!
//! ## Design Philosophy
//!
//! This crate provides multiple backends for running ML models:
//!
//! - **ONNX (`OnnxTask`)**: For simpler tasks (embeddings, classification, NER) where ONNX
//!   models are available and efficient.
//! - **Candle (`CandleTextGenTask`)**: For modern LLMs (Qwen3, Llama 3.x, Mistral) that don't
//!   have ONNX exports. Loads safetensors directly from HuggingFace.
//!
//! Configuration determines which backend and model to use - no code changes required.
//!
//! ## Why Two Backends?
//!
//! Most modern LLMs (2024+) don't have ONNX exports:
//! - **Qwen3, Llama 3.x, Mistral v0.3+**: Only safetensors available
//! - **DeepSeek-R1, MiniMax**: No ONNX
//!
//! ONNX is still preferred for:
//! - Embeddings (BGE-M3, all-MiniLM)
//! - Classification
//! - NER
//! - Older/simpler models with good ONNX support
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
//! // Task accepts raw input and handles preprocessing internally
//! // Input: {"text": "Hello world"}
//! ```
//!
//! ## Input Formats (from LangChain)
//!
//! Text generation (Candle backend):
//! ```json
//! {"text": "Once upon a time"}
//! ```
//!
//! Text embedding (ONNX backend):
//! ```json
//! {"text": "What is the capital of France?"}
//! ```
//!
//! Question answering:
//! ```json
//! {"question": "What is Paris?", "context": "Paris is the capital of France."}
//! ```
//!
//! Image classification:
//! ```json
//! {"image": "s3://bucket/image.jpg"}
//! ```

pub mod error;
pub mod onnx;
pub mod registry;
pub mod seq2seq;
pub mod session;
pub mod tensor_utils;

// CLIP dual-encoder task for zero-shot image classification
pub mod clip;

// PaddleOCR multi-model task for OCR
pub mod paddle_ocr;

// Shared utilities for Candle-based tasks
#[cfg(feature = "candle")]
pub mod candle_utils;

// Candle-based text generation (for modern LLMs without ONNX exports)
#[cfg(feature = "candle")]
pub mod candle_text_gen;

// Candle-based seq2seq (Whisper ASR, T5/FlanT5 text-to-text)
#[cfg(feature = "candle")]
pub mod candle_seq2seq;

// Candle-based TTS (Parler TTS)
#[cfg(feature = "candle")]
pub mod candle_tts;

// Llama.cpp-based text generation (GGUF models)
#[cfg(feature = "llama")]
pub mod llama_text_gen;

pub use error::{TaskError, TaskResult};
pub use onnx::OnnxTask;
pub use registry::TaskRegistry;
pub use seq2seq::{GenerationConfig, ModelArchitecture, Seq2SeqInput, Seq2SeqOutput, Seq2SeqTask};

// Re-export candle text generation when feature is enabled
#[cfg(feature = "candle")]
pub use candle_text_gen::{CandleGenConfig, CandleTextGenTask, TextGenModelArch};

// Re-export candle seq2seq when feature is enabled
#[cfg(feature = "candle")]
pub use candle_seq2seq::{
    CandleSeq2SeqInput, CandleSeq2SeqOutput, CandleSeq2SeqTask, Seq2SeqArch, Seq2SeqGenConfig,
};

// Re-export candle TTS when feature is enabled
#[cfg(feature = "candle")]
pub use candle_tts::{CandleTtsInput, CandleTtsOutput, CandleTtsTask, TtsGenConfig};

// Re-export llama text generation when feature is enabled
#[cfg(feature = "llama")]
pub use llama_text_gen::{LlamaGenConfig, LlamaTextGenInput, LlamaTextGenOutput, LlamaTextGenTask};

// Re-export echo task (no dependencies)
mod echo;
pub use echo::EchoTask;

// Re-export CLIP task
pub use clip::{ClipInput, ClipOutput, ClipTask};

// Re-export PaddleOCR task
pub use paddle_ocr::{PaddleOcrInput, PaddleOcrOutput, PaddleOcrTask};

// Re-export preprocessing types for convenience
pub use inference_preprocess::{PreprocessError, PreprocessResult, Preprocessor, RawInput};
