//! Tekken tokenizer support for Mistral models.
//!
//! This module provides support for Mistral's Tekken tokenizer format (`tekken.json`),
//! used by models like Voxtral. The Tekken format is different from HuggingFace's
//! standard `tokenizer.json` format.

use crate::error::{PreprocessError, PreprocessResult};
use std::path::Path;
use tekken::{SpecialTokenPolicy, Tekkenizer};
use tracing::{debug, info};

/// Wrapper around the Tekken tokenizer for Mistral models.
pub struct TekkenTokenizer {
    inner: Tekkenizer,
}

impl TekkenTokenizer {
    /// Load tokenizer from a tekken.json file.
    pub fn from_file(path: impl AsRef<Path>) -> PreprocessResult<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading Tekken tokenizer from file");

        let tokenizer = Tekkenizer::from_file(path).map_err(|e| {
            PreprocessError::Tokenizer(format!("Failed to load Tekken tokenizer: {e}"))
        })?;

        Ok(Self { inner: tokenizer })
    }

    /// Encode text to token IDs.
    ///
    /// # Arguments
    /// * `text` - The text to encode
    /// * `add_bos` - Whether to add beginning-of-sequence token
    /// * `add_eos` - Whether to add end-of-sequence token
    pub fn encode(&self, text: &str, add_bos: bool, add_eos: bool) -> PreprocessResult<Vec<i64>> {
        debug!(
            text_len = text.len(),
            add_bos, add_eos, "Encoding text with Tekken"
        );

        let tokens = self
            .inner
            .encode(text, add_bos, add_eos)
            .map_err(|e| PreprocessError::Tokenizer(format!("Tekken encoding failed: {e}")))?;

        // Convert u32 tokens to i64
        Ok(tokens.into_iter().map(|t| i64::from(t)).collect())
    }

    /// Decode token IDs back to text.
    ///
    /// # Arguments
    /// * `token_ids` - The token IDs to decode
    /// * `skip_special_tokens` - Whether to skip special tokens in output
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> PreprocessResult<String> {
        // Convert i64 to u32
        let tokens: Vec<u32> = token_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();

        let policy = if skip_special_tokens {
            SpecialTokenPolicy::Ignore
        } else {
            SpecialTokenPolicy::Keep
        };

        self.inner
            .decode(&tokens, policy)
            .map_err(|e| PreprocessError::Tokenizer(format!("Tekken decoding failed: {e}")))
    }

    /// Check if this tokenizer has audio support (for Voxtral).
    pub fn has_audio_support(&self) -> bool {
        self.inner.has_audio_support()
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

impl std::fmt::Debug for TekkenTokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TekkenTokenizer")
            .field("vocab_size", &self.vocab_size())
            .field("has_audio_support", &self.has_audio_support())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    // Tests would require actual tekken.json files
}
