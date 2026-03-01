//! Classification postprocessing.
//!
//! Handles softmax computation, label mapping, and top-k extraction
//! for text and image classification tasks.

use crate::error::{PostprocessError, PostprocessResult};
use crate::output::{ClassificationOutput, LabelScore, MultiLabelOutput};
use std::collections::HashMap;

/// Classification postprocessor.
pub struct ClassificationPostprocessor {
    /// Label mapping (id -> label name)
    id2label: HashMap<i64, String>,
    /// Number of top predictions to return
    top_k: usize,
    /// Threshold for multi-label classification
    threshold: f32,
    /// Whether this is multi-label classification
    multi_label: bool,
}

impl ClassificationPostprocessor {
    /// Create a new classification postprocessor.
    pub fn new(id2label: HashMap<i64, String>) -> Self {
        Self {
            id2label,
            top_k: 5,
            threshold: 0.5,
            multi_label: false,
        }
    }

    /// Look up label name by index, with fallback to "LABEL_{index}".
    ///
    /// # Clippy allowances
    /// - `cast_possible_wrap`: Index values from classification outputs are always
    ///   non-negative and small (typically < 1000 labels), safe to cast to i64.
    #[allow(clippy::cast_possible_wrap)]
    fn get_label(&self, index: usize) -> String {
        self.id2label
            .get(&(index as i64))
            .cloned()
            .unwrap_or_else(|| format!("LABEL_{index}"))
    }

    /// Create with default label mapping (LABEL_0, LABEL_1, etc.)
    pub fn with_num_labels(num_labels: usize) -> Self {
        let id2label: HashMap<i64, String> = (0..num_labels)
            .map(|i| {
                #[allow(clippy::cast_possible_wrap)]
                (i as i64, format!("LABEL_{i}"))
            })
            .collect();
        Self::new(id2label)
    }

    /// Set top-k predictions to return.
    #[must_use]
    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Set multi-label mode with threshold.
    #[must_use]
    pub fn with_multi_label(mut self, threshold: f32) -> Self {
        self.multi_label = true;
        self.threshold = threshold;
        self
    }

    /// Process logits into classification output.
    ///
    /// Expects logits shape: [batch_size, num_classes] or [num_classes]
    pub fn process(&self, logits: &[f32]) -> PostprocessResult<ClassificationOutput> {
        if logits.is_empty() {
            return Err(PostprocessError::InvalidOutput("Empty logits".into()));
        }

        // Apply softmax
        let probabilities = softmax(logits);

        // Find top prediction
        let (max_idx, max_prob) = probabilities
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .ok_or_else(|| PostprocessError::InvalidOutput("Failed to find max".into()))?;

        let label = self.get_label(max_idx);

        // Build all scores if we have labels
        let scores: HashMap<String, f32> = probabilities
            .iter()
            .enumerate()
            .map(|(i, &p)| (self.get_label(i), p))
            .collect();

        Ok(ClassificationOutput::new(label, *max_prob).with_all_scores(scores))
    }

    /// Process logits into multi-label output.
    ///
    /// Returns all labels above threshold.
    pub fn process_multi_label(&self, logits: &[f32]) -> PostprocessResult<MultiLabelOutput> {
        if logits.is_empty() {
            return Err(PostprocessError::InvalidOutput("Empty logits".into()));
        }

        // Apply sigmoid for multi-label (independent probabilities)
        let probabilities: Vec<f32> = logits.iter().map(|&x| sigmoid(x)).collect();

        // Get all labels above threshold
        let mut labels: Vec<LabelScore> = probabilities
            .iter()
            .enumerate()
            .filter(|(_, &p)| p >= self.threshold)
            .map(|(i, &p)| LabelScore::new(self.get_label(i), p))
            .collect();

        // Sort by score descending
        labels.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to top_k
        labels.truncate(self.top_k);

        Ok(MultiLabelOutput { labels })
    }

    /// Process batch of logits.
    ///
    /// Expects shape: [batch_size, num_classes]
    pub fn process_batch(
        &self,
        logits: &[Vec<f32>],
    ) -> PostprocessResult<Vec<ClassificationOutput>> {
        logits.iter().map(|l| self.process(l)).collect()
    }

    /// Get top-k predictions with scores.
    pub fn top_k(&self, logits: &[f32], k: usize) -> PostprocessResult<Vec<LabelScore>> {
        if logits.is_empty() {
            return Err(PostprocessError::InvalidOutput("Empty logits".into()));
        }

        let probabilities = softmax(logits);

        // Create (index, probability) pairs and sort
        let mut indexed: Vec<(usize, f32)> = probabilities.iter().copied().enumerate().collect();
        indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k and convert to LabelScore
        let results: Vec<LabelScore> = indexed
            .into_iter()
            .take(k)
            .map(|(i, p)| LabelScore::new(self.get_label(i), p))
            .collect();

        Ok(results)
    }
}

/// Compute softmax over a slice.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    // Find max for numerical stability
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    // Compute exp(x - max) for each element
    let exp_values: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();

    // Sum of all exp values
    let sum: f32 = exp_values.iter().sum();

    // Normalize
    exp_values.into_iter().map(|e| e / sum).collect()
}

/// Compute sigmoid for a single value.
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = softmax(&logits);

        // Sum should be 1
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Probabilities should be in ascending order (like logits)
        assert!(probs[0] < probs[1]);
        assert!(probs[1] < probs[2]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.99);
        assert!(sigmoid(-10.0) < 0.01);
    }

    #[test]
    fn test_classification() {
        let mut id2label = HashMap::new();
        id2label.insert(0, "negative".to_string());
        id2label.insert(1, "positive".to_string());

        let processor = ClassificationPostprocessor::new(id2label);
        let logits = vec![-1.0, 2.0]; // Should predict positive

        let result = processor.process(&logits).unwrap();
        assert_eq!(result.label, "positive");
        assert!(result.score > 0.9);
    }
}
