//! Qwen3 TTS model implementation.
//!
//! Module structure:
//! - `config` — Serde config structs (from HF config.json)
//! - `rope` — MRoPE (multimodal rotary embeddings) and standard RoPE
//! - `layers` — Shared building blocks (Attention, MLP, DecoderLayer, ResizeMLP)
//! - `talker` — Main transformer (text → CB0 tokens)
//! - `code_predictor` — Sub-model for CB1-CB15 generation
//! - `speech_tokenizer` — Neural audio codec decoder (codes → waveform)
//! - `generate` — Two-stage generation loop and public API

// VarBuilder is passed by value throughout candle (it's a lightweight ref-counted wrapper).
// Standard ML names like q, k, v, b are conventional and clear.
// gate_proj/up_proj/down_proj are standard transformer MLP names.

#[allow(clippy::needless_pass_by_value)]
pub mod code_predictor;
pub mod config;
#[allow(clippy::needless_pass_by_value)]
pub mod generate;
#[allow(
    clippy::needless_pass_by_value,
    clippy::many_single_char_names,
    clippy::struct_field_names,
    clippy::type_complexity
)]
pub mod layers;
pub mod rope;
#[allow(clippy::needless_pass_by_value)]
pub mod speech_tokenizer;
#[allow(clippy::needless_pass_by_value)]
pub mod talker;

pub use config::Qwen3TtsConfig;
pub use generate::{GenerateOutput, Qwen3TtsModel};
