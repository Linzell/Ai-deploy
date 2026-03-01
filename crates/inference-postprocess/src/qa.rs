//! Question answering postprocessing.
//!
//! Handles span extraction for extractive QA models like BERT/RoBERTa.

use crate::error::{PostprocessError, PostprocessResult};
use crate::output::QAOutput;

/// QA postprocessor for extractive question answering.
pub struct QAPostprocessor {
    /// Number of top answers to consider
    top_k: usize,
    /// Maximum answer length in tokens
    max_answer_length: usize,
    /// Minimum score threshold
    score_threshold: f32,
}

impl Default for QAPostprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl QAPostprocessor {
    /// Create a new QA postprocessor.
    pub fn new() -> Self {
        Self {
            top_k: 5,
            max_answer_length: 30,
            score_threshold: 0.0,
        }
    }

    /// Set maximum answer length.
    #[must_use]
    pub fn with_max_answer_length(mut self, length: usize) -> Self {
        self.max_answer_length = length;
        self
    }

    /// Set score threshold.
    #[must_use]
    pub fn with_score_threshold(mut self, threshold: f32) -> Self {
        self.score_threshold = threshold;
        self
    }

    /// Process start and end logits into QA output.
    ///
    /// # Arguments
    /// * `start_logits` - Logits for start positions [seq_len]
    /// * `end_logits` - Logits for end positions [seq_len]
    /// * `context` - Original context text
    /// * `offset_mapping` - Token to character offset mapping [(start, end), ...]
    pub fn process(
        &self,
        start_logits: &[f32],
        end_logits: &[f32],
        context: &str,
        offset_mapping: &[(usize, usize)],
    ) -> PostprocessResult<QAOutput> {
        if start_logits.len() != end_logits.len() {
            return Err(PostprocessError::Shape(
                "Start and end logits must have same length".into(),
            ));
        }

        if start_logits.is_empty() {
            return Err(PostprocessError::InvalidOutput("Empty logits".into()));
        }

        // Find best start/end combination
        let (best_start, best_end, score) = self.find_best_span(start_logits, end_logits);

        // Extract answer text using offset mapping
        let answer = if best_start < offset_mapping.len() && best_end < offset_mapping.len() {
            let char_start = offset_mapping[best_start].0;
            let char_end = offset_mapping[best_end].1;

            if char_start <= char_end && char_end <= context.len() {
                context[char_start..char_end].to_string()
            } else {
                String::new()
            }
        } else {
            String::new()
        };

        Ok(QAOutput::new(answer, score).with_span(best_start, best_end))
    }

    /// Process without offset mapping (returns token indices only).
    pub fn process_indices(
        &self,
        start_logits: &[f32],
        end_logits: &[f32],
    ) -> PostprocessResult<(usize, usize, f32)> {
        Ok(self.find_best_span(start_logits, end_logits))
    }

    /// Find the best start/end span.
    fn find_best_span(&self, start_logits: &[f32], end_logits: &[f32]) -> (usize, usize, f32) {
        // Apply softmax to get probabilities
        let start_probs = crate::classification::softmax(start_logits);
        let end_probs = crate::classification::softmax(end_logits);

        // Find top-k start positions
        let mut start_indices: Vec<(usize, f32)> =
            start_probs.iter().copied().enumerate().collect();
        start_indices
            .sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        start_indices.truncate(self.top_k);

        // Find top-k end positions
        let mut end_indices: Vec<(usize, f32)> = end_probs.iter().copied().enumerate().collect();
        end_indices.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        end_indices.truncate(self.top_k);

        // Find best valid span
        let mut best_score = f32::NEG_INFINITY;
        let mut best_start = 0;
        let mut best_end = 0;

        for &(start_idx, start_prob) in &start_indices {
            for &(end_idx, end_prob) in &end_indices {
                // Skip invalid spans (end before start, or too long)
                if end_idx < start_idx {
                    continue;
                }
                if end_idx - start_idx + 1 > self.max_answer_length {
                    continue;
                }

                let score = start_prob * end_prob;
                if score > best_score {
                    best_score = score;
                    best_start = start_idx;
                    best_end = end_idx;
                }
            }
        }

        // Handle no valid span found
        if best_score == f32::NEG_INFINITY {
            // Return first position as fallback (often [CLS] token for "no answer")
            return (0, 0, 0.0);
        }

        (best_start, best_end, best_score)
    }

    /// Decode answer from input_ids using token indices.
    ///
    /// This is a simpler version when we have the original tokens.
    #[cfg(feature = "text")]
    pub fn decode_answer(
        &self,
        input_ids: &[i64],
        start_idx: usize,
        end_idx: usize,
        tokenizer: &tokenizers::Tokenizer,
    ) -> PostprocessResult<String> {
        if start_idx > end_idx || end_idx >= input_ids.len() {
            return Ok(String::new());
        }

        let token_ids: Vec<u32> = input_ids[start_idx..=end_idx]
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

        tokenizer
            .decode(&token_ids, true)
            .map_err(|e| PostprocessError::Decoding(format!("Failed to decode: {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_best_span() {
        let processor = QAPostprocessor::new();

        // Simple case: clear start at position 2, end at position 4
        let start_logits = vec![-1.0, -1.0, 5.0, -1.0, -1.0, -1.0];
        let end_logits = vec![-1.0, -1.0, -1.0, -1.0, 5.0, -1.0];

        let (start, end, _score) = processor.find_best_span(&start_logits, &end_logits);
        assert_eq!(start, 2);
        assert_eq!(end, 4);
    }

    #[test]
    fn test_invalid_span_rejected() {
        let processor = QAPostprocessor::new().with_max_answer_length(2);

        // Start at 0, end at 5 - too long
        let start_logits = vec![5.0, -1.0, -1.0, -1.0, -1.0, -1.0];
        let end_logits = vec![-1.0, -1.0, -1.0, -1.0, -1.0, 5.0];

        let (start, end, _) = processor.find_best_span(&start_logits, &end_logits);
        // Should pick a shorter valid span
        assert!(end - start < 2);
    }
}
