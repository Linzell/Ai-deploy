//! Text decoding for generation tasks.
//!
//! Handles token decoding for text generation, translation, summarization, and ASR.

use crate::error::{PostprocessError, PostprocessResult};
use crate::output::TextOutput;
use std::sync::Arc;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::debug;

/// Text decoder using HuggingFace tokenizers.
pub struct TextDecoder {
    tokenizer: Arc<HfTokenizer>,
    /// Skip special tokens when decoding
    skip_special_tokens: bool,
}

impl TextDecoder {
    /// Create a new text decoder from a tokenizer.
    pub fn new(tokenizer: HfTokenizer) -> Self {
        Self {
            tokenizer: Arc::new(tokenizer),
            skip_special_tokens: true,
        }
    }

    /// Load decoder from tokenizer.json file.
    pub fn from_file(path: &std::path::Path) -> PostprocessResult<Self> {
        let tokenizer = HfTokenizer::from_file(path)
            .map_err(|e| PostprocessError::Config(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self::new(tokenizer))
    }

    /// Load decoder from bytes.
    pub fn from_bytes(bytes: &[u8]) -> PostprocessResult<Self> {
        let tokenizer = HfTokenizer::from_bytes(bytes)
            .map_err(|e| PostprocessError::Config(format!("Failed to load tokenizer: {e}")))?;
        Ok(Self::new(tokenizer))
    }

    /// Set whether to skip special tokens.
    #[must_use]
    pub fn with_skip_special_tokens(mut self, skip: bool) -> Self {
        self.skip_special_tokens = skip;
        self
    }

    /// Decode token IDs to text.
    pub fn decode(&self, token_ids: &[i64]) -> PostprocessResult<TextOutput> {
        // Convert i64 to u32 for tokenizer
        let ids: Vec<u32> = token_ids
            .iter()
            .filter_map(|&id| {
                if id >= 0 {
                    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                    Some(id as u32)
                } else {
                    None
                }
            })
            .collect();

        debug!(num_tokens = ids.len(), "Decoding tokens");

        let text = self
            .tokenizer
            .decode(&ids, self.skip_special_tokens)
            .map_err(|e| PostprocessError::Decoding(format!("Decode failed: {e}")))?;

        Ok(TextOutput::new(text))
    }

    /// Decode batch of token IDs.
    pub fn decode_batch(&self, batch_token_ids: &[Vec<i64>]) -> PostprocessResult<Vec<TextOutput>> {
        batch_token_ids.iter().map(|ids| self.decode(ids)).collect()
    }

    /// Decode from logits (argmax over vocabulary).
    ///
    /// Expects logits shape: [seq_len, vocab_size]
    pub fn decode_from_logits(&self, logits: &[Vec<f32>]) -> PostprocessResult<TextOutput> {
        let token_ids: Vec<i64> = logits
            .iter()
            .map(|token_logits| {
                token_logits
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map_or(0, |(idx, _)| {
                        #[allow(clippy::cast_possible_wrap)]
                        {
                            idx as i64
                        }
                    })
            })
            .collect();

        self.decode(&token_ids)
    }

    /// Get the underlying tokenizer for advanced operations.
    pub fn tokenizer(&self) -> &HfTokenizer {
        &self.tokenizer
    }
}

impl std::fmt::Debug for TextDecoder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TextDecoder")
            .field("skip_special_tokens", &self.skip_special_tokens)
            .finish_non_exhaustive()
    }
}
