//! Embedding postprocessing.
//!
//! Handles pooling strategies and normalization for feature extraction models.

use crate::error::{PostprocessError, PostprocessResult};
use crate::output::EmbeddingOutput;

/// Pooling strategy for embeddings.
#[derive(Debug, Clone, Copy, Default)]
pub enum PoolingStrategy {
    /// Use [CLS] token embedding (position 0)
    #[default]
    Cls,
    /// Mean pooling over all tokens
    Mean,
    /// Max pooling over all tokens
    Max,
    /// Mean of last N hidden states
    LastNMean(usize),
}

/// Embedding postprocessor.
pub struct EmbeddingPostprocessor {
    /// Pooling strategy
    pooling: PoolingStrategy,
    /// Whether to L2 normalize the output
    normalize: bool,
    /// Optional: truncate to this many dimensions (Matryoshka)
    truncate_dim: Option<usize>,
}

impl Default for EmbeddingPostprocessor {
    fn default() -> Self {
        Self::new()
    }
}

impl EmbeddingPostprocessor {
    /// Create a new embedding postprocessor.
    pub fn new() -> Self {
        Self {
            pooling: PoolingStrategy::default(),
            normalize: true,
            truncate_dim: None,
        }
    }

    /// Set pooling strategy.
    #[must_use]
    pub fn with_pooling(mut self, pooling: PoolingStrategy) -> Self {
        self.pooling = pooling;
        self
    }

    /// Set whether to normalize embeddings.
    #[must_use]
    pub fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set dimension truncation (for Matryoshka embeddings).
    #[must_use]
    pub fn with_truncate_dim(mut self, dim: usize) -> Self {
        self.truncate_dim = Some(dim);
        self
    }

    /// Process last hidden state into embedding.
    ///
    /// # Arguments
    /// * `hidden_states` - Shape: [seq_len, hidden_dim]
    /// * `attention_mask` - Optional attention mask [seq_len] for mean pooling
    pub fn process(
        &self,
        hidden_states: &[Vec<f32>],
        attention_mask: Option<&[i64]>,
    ) -> PostprocessResult<EmbeddingOutput> {
        if hidden_states.is_empty() {
            return Err(PostprocessError::InvalidOutput(
                "Empty hidden states".into(),
            ));
        }

        let mut embedding = match self.pooling {
            PoolingStrategy::Cls => Self::cls_pooling(hidden_states)?,
            PoolingStrategy::Mean => Self::mean_pooling(hidden_states, attention_mask)?,
            PoolingStrategy::Max => Self::max_pooling(hidden_states)?,
            PoolingStrategy::LastNMean(n) => Self::last_n_mean_pooling(hidden_states, n)?,
        };

        // Normalize in-place (avoids allocation since we own the vector)
        if self.normalize {
            l2_normalize_inplace(&mut embedding);
        }

        // Truncate if requested
        if let Some(dim) = self.truncate_dim {
            if dim < embedding.len() {
                embedding.truncate(dim);
                // Re-normalize after truncation (in-place)
                if self.normalize {
                    l2_normalize_inplace(&mut embedding);
                }
            }
        }

        Ok(EmbeddingOutput::new(embedding))
    }

    /// Process batch of hidden states.
    pub fn process_batch(
        &self,
        batch_hidden_states: &[Vec<Vec<f32>>],
        batch_attention_mask: Option<&[Vec<i64>]>,
    ) -> PostprocessResult<Vec<EmbeddingOutput>> {
        batch_hidden_states
            .iter()
            .enumerate()
            .map(|(i, hs)| {
                let mask = batch_attention_mask.map(|m| m[i].as_slice());
                self.process(hs, mask)
            })
            .collect()
    }

    /// CLS token pooling (first token).
    fn cls_pooling(hidden_states: &[Vec<f32>]) -> PostprocessResult<Vec<f32>> {
        hidden_states
            .first()
            .cloned()
            .ok_or_else(|| PostprocessError::InvalidOutput("No hidden states".into()))
    }

    /// Mean pooling over all tokens.
    fn mean_pooling(
        hidden_states: &[Vec<f32>],
        attention_mask: Option<&[i64]>,
    ) -> PostprocessResult<Vec<f32>> {
        let hidden_dim = hidden_states
            .first()
            .map(Vec::len)
            .ok_or_else(|| PostprocessError::InvalidOutput("No hidden states".into()))?;

        let mut sum = vec![0.0f32; hidden_dim];
        let mut count = 0.0f32;

        for (i, hidden) in hidden_states.iter().enumerate() {
            let mask_value = attention_mask.map_or(1, |m| if i < m.len() { m[i] } else { 0 });

            if mask_value > 0 {
                #[allow(clippy::cast_precision_loss)]
                let weight = mask_value as f32;
                for (j, &val) in hidden.iter().enumerate() {
                    sum[j] += val * weight;
                }
                count += weight;
            }
        }

        // Avoid division by zero
        if count > 0.0 {
            for val in &mut sum {
                *val /= count;
            }
        }

        Ok(sum)
    }

    /// Max pooling over all tokens.
    fn max_pooling(hidden_states: &[Vec<f32>]) -> PostprocessResult<Vec<f32>> {
        let hidden_dim = hidden_states
            .first()
            .map(Vec::len)
            .ok_or_else(|| PostprocessError::InvalidOutput("No hidden states".into()))?;

        let mut max_vals = vec![f32::NEG_INFINITY; hidden_dim];

        for hidden in hidden_states {
            for (j, &val) in hidden.iter().enumerate() {
                if val > max_vals[j] {
                    max_vals[j] = val;
                }
            }
        }

        Ok(max_vals)
    }

    /// Mean of last N hidden states.
    fn last_n_mean_pooling(hidden_states: &[Vec<f32>], n: usize) -> PostprocessResult<Vec<f32>> {
        let seq_len = hidden_states.len();
        let start = seq_len.saturating_sub(n);
        let last_n = &hidden_states[start..];

        Self::mean_pooling(last_n, None)
    }
}

/// L2 normalize a vector (returns new allocation).
pub fn l2_normalize(vec: &[f32]) -> Vec<f32> {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm > 0.0 {
        vec.iter().map(|&x| x / norm).collect()
    } else {
        vec.to_vec()
    }
}

/// L2 normalize a vector in-place (no allocation).
pub fn l2_normalize_inplace(vec: &mut [f32]) {
    let norm: f32 = vec.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm > 0.0 {
        for x in vec.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|&x| x * x).sum::<f32>().sqrt();

    if norm_a > 0.0 && norm_b > 0.0 {
        dot / (norm_a * norm_b)
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_l2_normalize() {
        let vec = vec![3.0, 4.0];
        let normalized = l2_normalize(&vec);

        assert!((normalized[0] - 0.6).abs() < 1e-6);
        assert!((normalized[1] - 0.8).abs() < 1e-6);

        // Norm should be 1
        let norm: f32 = normalized.iter().map(|&x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let c = vec![1.0, 0.0];

        // Orthogonal vectors
        assert!((cosine_similarity(&a, &b) - 0.0).abs() < 1e-6);

        // Same vector
        assert!((cosine_similarity(&a, &c) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_pooling() {
        let processor = EmbeddingPostprocessor::new()
            .with_pooling(PoolingStrategy::Mean)
            .with_normalize(false);

        let hidden_states = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result = processor.process(&hidden_states, None).unwrap();
        assert_eq!(result.embedding, vec![2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_cls_pooling() {
        let processor = EmbeddingPostprocessor::new()
            .with_pooling(PoolingStrategy::Cls)
            .with_normalize(false);

        let hidden_states = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];

        let result = processor.process(&hidden_states, None).unwrap();
        assert_eq!(result.embedding, vec![1.0, 2.0, 3.0]);
    }
}
