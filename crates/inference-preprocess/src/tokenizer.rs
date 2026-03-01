//! Tokenizer using HuggingFace tokenizers crate.
//!
//! This provides the same tokenization as Python's `transformers` library,
//! using the Rust implementation of HuggingFace tokenizers.

use crate::error::{PreprocessError, PreprocessResult};
use std::path::Path;
use std::sync::Arc;
use tokenizers::Tokenizer as HfTokenizer;
use tracing::{debug, info, warn};

/// Processed tokens ready for ONNX inference.
#[derive(Debug, Clone)]
pub struct TokenizedOutput {
    /// Token IDs [batch_size, seq_len]
    pub input_ids: Vec<Vec<i64>>,
    /// Attention mask [batch_size, seq_len]
    pub attention_mask: Vec<Vec<i64>>,
    /// Token type IDs (for BERT-style models) [batch_size, seq_len]
    pub token_type_ids: Option<Vec<Vec<i64>>>,
    /// Offset mapping for each token [(start_char, end_char), ...]
    /// Used for QA tasks to map token positions back to character positions
    pub offset_mapping: Option<Vec<Vec<(usize, usize)>>>,
}

impl TokenizedOutput {
    /// Convert to JSON-compatible HashMap for ONNX input.
    pub fn to_json_inputs(&self) -> serde_json::Value {
        let mut inputs = serde_json::Map::new();
        inputs.insert("input_ids".to_string(), serde_json::json!(self.input_ids));
        inputs.insert(
            "attention_mask".to_string(),
            serde_json::json!(self.attention_mask),
        );
        if let Some(ref token_type_ids) = self.token_type_ids {
            inputs.insert(
                "token_type_ids".to_string(),
                serde_json::json!(token_type_ids),
            );
        }
        serde_json::Value::Object(inputs)
    }
}

/// Tokenizer wrapper for HuggingFace tokenizers.
pub struct Tokenizer {
    inner: Arc<HfTokenizer>,
    /// Whether to include token_type_ids (BERT-style models)
    include_token_type_ids: bool,
    /// Maximum sequence length (None = model default)
    max_length: Option<usize>,
}

impl Tokenizer {
    /// Load tokenizer from a local file (tokenizer.json).
    ///
    /// Handles both standard HuggingFace tokenizers and simple vocab-only
    /// tokenizers (used by TTS models like MMS-TTS, SpeechT5).
    pub fn from_file(path: impl AsRef<Path>) -> PreprocessResult<Self> {
        let path = path.as_ref();
        info!(path = %path.display(), "Loading tokenizer from file");

        // Try loading directly first
        match HfTokenizer::from_file(path) {
            Ok(tokenizer) => Ok(Self {
                inner: Arc::new(tokenizer),
                include_token_type_ids: true,
                max_length: None,
            }),
            Err(e) => {
                // Check if it's a model type error - try to fix the tokenizer
                let err_str = e.to_string();
                if err_str.contains("ModelUntagged") || err_str.contains("model") {
                    warn!(
                        error = %err_str,
                        "Standard tokenizer loading failed, trying to patch vocab-only format"
                    );
                    Self::from_file_with_vocab_patch(path)
                } else {
                    Err(PreprocessError::Tokenizer(format!(
                        "Failed to load tokenizer: {e}"
                    )))
                }
            }
        }
    }

    /// Load tokenizer by patching vocab-only format to use WordLevel model.
    ///
    /// TTS models often have tokenizers with just a vocab mapping and no model type.
    /// This function patches the JSON to add a WordLevel model type.
    fn from_file_with_vocab_patch(path: &Path) -> PreprocessResult<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| PreprocessError::FileLoad(format!("Failed to read tokenizer: {e}")))?;

        let mut json: serde_json::Value = serde_json::from_str(&content)
            .map_err(|e| PreprocessError::Tokenizer(format!("Invalid tokenizer JSON: {e}")))?;

        // Check if model exists but has no type
        if let Some(model) = json.get_mut("model") {
            if model.get("type").is_none() {
                // Add WordLevel type for simple vocab tokenizers
                if let Some(obj) = model.as_object_mut() {
                    obj.insert("type".to_string(), serde_json::json!("WordLevel"));
                    // WordLevel requires unk_token
                    if obj.get("unk_token").is_none() {
                        obj.insert("unk_token".to_string(), serde_json::json!("<unk>"));
                    }
                    info!("Patched tokenizer to use WordLevel model");
                }
            }
        }

        let patched = serde_json::to_vec(&json).map_err(|e| {
            PreprocessError::Tokenizer(format!("Failed to serialize patched tokenizer: {e}"))
        })?;

        Self::from_bytes(&patched)
    }

    /// Load tokenizer from bytes (tokenizer.json content).
    pub fn from_bytes(bytes: &[u8]) -> PreprocessResult<Self> {
        info!(size = bytes.len(), "Loading tokenizer from bytes");

        let tokenizer = HfTokenizer::from_bytes(bytes)
            .map_err(|e| PreprocessError::Tokenizer(format!("Failed to load tokenizer: {e}")))?;

        Ok(Self {
            inner: Arc::new(tokenizer),
            include_token_type_ids: true,
            max_length: None,
        })
    }

    /// Set whether to include token_type_ids in output.
    #[must_use]
    pub fn with_token_type_ids(mut self, include: bool) -> Self {
        self.include_token_type_ids = include;
        self
    }

    /// Set maximum sequence length for truncation.
    #[must_use]
    pub fn with_max_length(mut self, max_length: usize) -> Self {
        self.max_length = Some(max_length);
        self
    }

    /// Tokenize a single text.
    pub fn encode(&self, text: &str) -> PreprocessResult<TokenizedOutput> {
        self.encode_batch(&[text.to_string()])
    }

    /// Tokenize a batch of texts.
    pub fn encode_batch(&self, texts: &[String]) -> PreprocessResult<TokenizedOutput> {
        debug!(batch_size = texts.len(), "Tokenizing batch");

        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| PreprocessError::Tokenizer(format!("Tokenization failed: {e}")))?;

        Ok(self.extract_encodings(&encodings))
    }

    /// Tokenize a text pair (for QA, similarity, etc.).
    ///
    /// Creates a sequence like: [CLS] text_a [SEP] text_b [SEP]
    pub fn encode_pair(&self, text_a: &str, text_b: &str) -> PreprocessResult<TokenizedOutput> {
        self.encode_pair_batch(&[(text_a.to_string(), text_b.to_string())])
    }

    /// Tokenize a batch of text pairs.
    pub fn encode_pair_batch(
        &self,
        pairs: &[(String, String)],
    ) -> PreprocessResult<TokenizedOutput> {
        debug!(batch_size = pairs.len(), "Tokenizing text pairs");

        let encodings = self
            .inner
            .encode_batch(pairs.to_vec(), true)
            .map_err(|e| PreprocessError::Tokenizer(format!("Tokenization failed: {e}")))?;

        Ok(self.extract_encodings(&encodings))
    }

    /// Extract tokenized output from HuggingFace encodings.
    ///
    /// Common logic for both single texts and text pairs.
    fn extract_encodings(&self, encodings: &[tokenizers::Encoding]) -> TokenizedOutput {
        let mut input_ids = Vec::with_capacity(encodings.len());
        let mut attention_mask = Vec::with_capacity(encodings.len());
        let mut token_type_ids = if self.include_token_type_ids {
            Some(Vec::with_capacity(encodings.len()))
        } else {
            None
        };
        let mut offset_mapping = Vec::with_capacity(encodings.len());

        for encoding in encodings {
            // Get IDs and convert to i64
            let ids: Vec<i64> = encoding.get_ids().iter().map(|&id| i64::from(id)).collect();
            let mask: Vec<i64> = encoding
                .get_attention_mask()
                .iter()
                .map(|&m| i64::from(m))
                .collect();
            let offsets: Vec<(usize, usize)> = encoding.get_offsets().to_vec();

            // Truncate if max_length is set
            let (ids, mask, offsets) = if let Some(max_len) = self.max_length {
                (
                    ids.into_iter().take(max_len).collect(),
                    mask.into_iter().take(max_len).collect(),
                    offsets.into_iter().take(max_len).collect(),
                )
            } else {
                (ids, mask, offsets)
            };

            input_ids.push(ids);
            attention_mask.push(mask);
            offset_mapping.push(offsets);

            if let Some(ref mut ttype) = token_type_ids {
                let types: Vec<i64> = encoding
                    .get_type_ids()
                    .iter()
                    .map(|&t| i64::from(t))
                    .collect();
                let types = if let Some(max_len) = self.max_length {
                    types.into_iter().take(max_len).collect()
                } else {
                    types
                };
                ttype.push(types);
            }
        }

        TokenizedOutput {
            input_ids,
            attention_mask,
            token_type_ids,
            offset_mapping: Some(offset_mapping),
        }
    }

    /// Decode token IDs back to text.
    ///
    /// Used for seq2seq tasks to convert generated tokens back to text.
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> PreprocessResult<String> {
        // Convert i64 to u32 for HuggingFace tokenizer
        // Token IDs should always be non-negative and fit in u32
        let ids: Vec<u32> = token_ids
            .iter()
            .filter_map(|&id| u32::try_from(id).ok())
            .collect();

        self.inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| PreprocessError::Tokenizer(format!("Failed to decode tokens: {e}")))
    }

    /// Decode a batch of token ID sequences back to text.
    pub fn decode_batch(
        &self,
        batch_token_ids: &[Vec<i64>],
        skip_special_tokens: bool,
    ) -> PreprocessResult<Vec<String>> {
        batch_token_ids
            .iter()
            .map(|ids| self.decode(ids, skip_special_tokens))
            .collect()
    }

    /// Get access to the underlying HuggingFace tokenizer for advanced operations.
    pub fn inner(&self) -> &HfTokenizer {
        &self.inner
    }
}

impl std::fmt::Debug for Tokenizer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tokenizer")
            .field("include_token_type_ids", &self.include_token_type_ids)
            .field("max_length", &self.max_length)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    // Tests would require actual tokenizer.json files
}
