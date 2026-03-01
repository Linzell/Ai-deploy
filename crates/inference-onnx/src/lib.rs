//! # Inference ONNX
//!
//! ONNX Runtime backend for inference tasks.
//!
//! This crate provides ONNX-based inference tasks:
//! - [`OnnxTask`] - Generic ONNX model inference (single forward pass)
//! - [`Seq2SeqTask`] - Autoregressive generation (encoder-decoder and decoder-only)
//! - [`ClipTask`] - CLIP dual-encoder for zero-shot classification
//! - [`PaddleOcrTask`] - PaddleOCR detection + recognition pipeline
//!
//! ## Features
//!
//! - `preprocess` - Enable preprocessing (tokenization, image, audio)
//! - `postprocess` - Enable postprocessing (decode tokens, format outputs)
//! - `coreml` - Metal/CoreML support for macOS
//! - `cuda` - CUDA support for Linux/Windows

mod clip;
mod error;
mod onnx;
mod paddle_ocr;
mod seq2seq;
mod session;
mod tensor_utils;

pub use clip::{ClipInput, ClipOutput, ClipTask};
pub use error::{TaskError, TaskResult};
pub use onnx::{OnnxInput, OnnxOutput, OnnxTask};
pub use paddle_ocr::{PaddleOcrInput, PaddleOcrOutput, PaddleOcrTask, TextBox};
pub use seq2seq::{Seq2SeqInput, Seq2SeqOutput, Seq2SeqTask};
pub use session::{
    load_session_for_seq2seq, load_session_for_seq2seq_from_bytes, load_session_from_bytes,
    load_session_from_file,
};
pub use tensor_utils::{
    array_f32_to_json, array_i64_to_json, contains_floats, infer_shape, json_to_array2_i64,
    json_to_array_f32, json_to_array_i64, json_to_tensor_value, TensorValue,
};

// Re-export from inference-core for convenience
pub use inference_core::generation::{GenerationConfig, ModelArchitecture};
pub use inference_core::task::{Task, TaskChunk, TaskResult as GrpcTaskResult, TaskStream};
