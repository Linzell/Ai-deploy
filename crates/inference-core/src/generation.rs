//! Generation configuration for autoregressive text generation.
//!
//! This module provides shared generation parameters used across all backends
//! (ONNX seq2seq, Candle, llama.cpp).

use crate::Config;
use serde::{Deserialize, Serialize};

/// Generation configuration for autoregressive decoding.
///
/// Used by all text generation backends to control sampling behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationConfig {
    /// Maximum number of tokens to generate
    pub max_new_tokens: usize,
    /// Temperature for sampling (1.0 = no change, <1.0 = sharper, >1.0 = smoother)
    pub temperature: f32,
    /// Top-p (nucleus) sampling threshold (0.0-1.0)
    pub top_p: f32,
    /// Top-k sampling (0 = disabled)
    pub top_k: usize,
    /// Repetition penalty (1.0 = no penalty, >1.0 = penalize repeats)
    pub repetition_penalty: f32,
    /// Context length for repetition penalty
    pub repeat_last_n: usize,
    /// End-of-sequence token IDs
    pub eos_token_ids: Vec<i64>,
    /// Pad token ID
    pub pad_token_id: i64,
    /// Decoder start token ID (for encoder-decoder models)
    pub decoder_start_token_id: i64,
    /// Random seed for reproducibility (None = random)
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 256,
            temperature: 1.0,
            top_p: 0.9,
            top_k: 0,
            repetition_penalty: 1.0,
            repeat_last_n: 64,
            eos_token_ids: vec![2], // Common EOS token
            pad_token_id: 0,
            decoder_start_token_id: 0,
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Create from inference config.
    pub fn from_config(config: &Config) -> Self {
        Self {
            max_new_tokens: config.max_tokens,
            temperature: config.temperature,
            top_p: config.top_p,
            ..Default::default()
        }
    }

    /// Builder: set max_new_tokens
    #[must_use]
    pub fn with_max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_new_tokens = max_tokens;
        self
    }

    /// Builder: set temperature
    #[must_use]
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Builder: set top_p
    #[must_use]
    pub fn with_top_p(mut self, top_p: f32) -> Self {
        self.top_p = top_p;
        self
    }

    /// Builder: set top_k
    #[must_use]
    pub fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Builder: set repetition_penalty
    #[must_use]
    pub fn with_repetition_penalty(mut self, penalty: f32) -> Self {
        self.repetition_penalty = penalty;
        self
    }

    /// Builder: set EOS token IDs
    #[must_use]
    pub fn with_eos_tokens(mut self, eos_token_ids: Vec<i64>) -> Self {
        self.eos_token_ids = eos_token_ids;
        self
    }

    /// Builder: set seed
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

/// Model architecture type for generation models.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    /// Decoder-only (GPT2, Llama, Qwen, Mistral, etc.)
    DecoderOnly,
    /// Encoder-decoder (T5, BART, Whisper, Florence-2, etc.)
    EncoderDecoder,
}

impl ModelArchitecture {
    /// Check if this is a decoder-only architecture
    pub fn is_decoder_only(&self) -> bool {
        matches!(self, Self::DecoderOnly)
    }

    /// Check if this is an encoder-decoder architecture
    pub fn is_encoder_decoder(&self) -> bool {
        matches!(self, Self::EncoderDecoder)
    }
}
