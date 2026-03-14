//! # Inference Candle
//!
//! Candle-based inference backend for modern LLMs, seq2seq models, and TTS.
//!
//! This crate provides native Rust inference using HuggingFace's Candle framework:
//! - [`CandleTextGenTask`] - Text generation (Qwen, Llama, Mistral, Phi)
//! - [`CandleSeq2SeqTask`] - Encoder-decoder models (Whisper, Voxtral, T5)
//! - [`CandleTtsTask`] - Text-to-speech (Parler TTS)
//!
//! ## Features
//!
//! - `metal` - Metal/CoreML support for macOS
//! - `cuda` - CUDA support for Linux/Windows
//!
//! ## Why Candle?
//!
//! Most modern LLMs (Qwen3, Llama 3.x, Mistral v0.3+) don't have official ONNX exports.
//! Candle loads safetensors directly and supports Metal/CUDA acceleration.

mod error;
mod seq2seq;
mod text_gen;
mod tts;
mod utils;

pub use error::{TaskError, TaskResult};
pub use seq2seq::{
    CandleSeq2SeqInput, CandleSeq2SeqOutput, CandleSeq2SeqTask, Seq2SeqArch, Seq2SeqGenConfig,
    Seq2SeqTokenizer,
};
pub use text_gen::{
    CandleGenConfig, CandleTextGenInput, CandleTextGenOutput, CandleTextGenTask, TextGenModelArch,
};
pub use tts::{CandleTtsInput, CandleTtsOutput, CandleTtsTask, TtsGenConfig};
pub use utils::{find_weight_files, is_gguf, is_gpu_compiled, load_safetensors_safe, resolve_device};

// Re-export from inference-core for convenience
pub use inference_core::task::{Task, TaskChunk, TaskResult as GrpcTaskResult, TaskStream};
