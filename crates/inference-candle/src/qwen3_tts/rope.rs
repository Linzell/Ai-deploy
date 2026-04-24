//! Rotary position embeddings for Qwen3 TTS.
//!
//! Two variants:
//! - **MRoPE** (multimodal RoPE) for the talker — splits head_dim into 3 sections,
//!   each receiving independent position IDs (temporal, height, width).
//!   Uses interleaved layout (pairs of [cos, -sin; sin, cos] applied element-wise).
//! - **Standard RoPE** for the code predictor — single position dimension.

use candle_core::{DType, Device, IndexOp, Result, Tensor};

/// Precomputed rotary embedding frequencies.
///
/// Shared between MRoPE (talker) and standard RoPE (code predictor).
/// The difference is only in how position IDs are constructed and how
/// cos/sin are sliced across the head dimension.
#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    /// Inverse frequency tensor, shape `[dim/2]`.
    inv_freq: Tensor,
    /// Maximum sequence length for which we cache cos/sin.
    max_seq_len: usize,
    /// Cached cos values, shape `[max_seq_len, dim/2]`.
    cos_cache: Tensor,
    /// Cached sin values, shape `[max_seq_len, dim/2]`.
    sin_cache: Tensor,
}

impl RotaryEmbedding {
    /// Create a new rotary embedding.
    ///
    /// `dim` is the number of dimensions to rotate (typically `head_dim`).
    /// `theta` is the base frequency (typically 1_000_000 for Qwen3 TTS).
    pub fn new(dim: usize, theta: f64, max_seq_len: usize, device: &Device) -> Result<Self> {
        let half_dim = dim / 2;
        let inv_freq: Vec<f32> = (0..half_dim)
            .map(|i| 1.0 / theta.powf(i as f64 * 2.0 / dim as f64) as f32)
            .collect();
        let inv_freq = Tensor::new(inv_freq, device)?;

        // Precompute cos/sin for positions [0, max_seq_len)
        let positions: Vec<f32> = (0..max_seq_len).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?; // [seq, 1]
        let inv_freq_row = inv_freq.unsqueeze(0)?; // [1, dim/2]
        let freqs = positions.matmul(&inv_freq_row)?; // [seq, dim/2]

        let cos_cache = freqs.cos()?;
        let sin_cache = freqs.sin()?;

        Ok(Self {
            inv_freq,
            max_seq_len,
            cos_cache,
            sin_cache,
        })
    }

    /// Extend the cache if needed for longer sequences.
    fn ensure_len(&mut self, seq_len: usize) -> Result<()> {
        if seq_len <= self.max_seq_len {
            return Ok(());
        }
        let device = self.inv_freq.device();
        let new_max = seq_len.next_power_of_two();
        let positions: Vec<f32> = (0..new_max).map(|p| p as f32).collect();
        let positions = Tensor::new(positions, device)?.unsqueeze(1)?;
        let inv_freq_row = self.inv_freq.unsqueeze(0)?;
        let freqs = positions.matmul(&inv_freq_row)?;
        self.cos_cache = freqs.cos()?;
        self.sin_cache = freqs.sin()?;
        self.max_seq_len = new_max;
        Ok(())
    }

    /// Get cos/sin slices for standard RoPE.
    ///
    /// Returns `(cos, sin)` each of shape `[seq_len, dim/2]`.
    pub fn get_cos_sin(&mut self, seq_len: usize, offset: usize) -> Result<(Tensor, Tensor)> {
        self.ensure_len(offset + seq_len)?;
        let cos = self.cos_cache.i(offset..offset + seq_len)?;
        let sin = self.sin_cache.i(offset..offset + seq_len)?;
        Ok((cos, sin))
    }
}

/// Apply standard rotary embedding (interleaved layout) to Q and K.
///
/// `x` has shape `[batch, num_heads, seq_len, head_dim]`.
/// `cos`, `sin` have shape `[seq_len, head_dim/2]`.
///
/// Uses the interleaved approach: pairs `(x[2i], x[2i+1])` are rotated.
pub fn apply_rotary_emb(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
    let (_b, _h, seq_len, head_dim) = x.dims4()?;
    let half = head_dim / 2;

    // Split into even/odd pairs
    let x_reshape = x.reshape(((), seq_len, half, 2))?;
    let x0 = x_reshape.i((.., .., .., 0))?; // [b*h, seq, half]
    let x1 = x_reshape.i((.., .., .., 1))?;

    // Broadcast cos/sin to match: [1, 1, seq, half] for the 4D view
    let cos = cos.unsqueeze(0)?; // [1, seq, half]
    let sin = sin.unsqueeze(0)?;

    // Rotate: (x0 * cos - x1 * sin, x0 * sin + x1 * cos)
    let o0 = (x0.broadcast_mul(&cos)? - x1.broadcast_mul(&sin)?)?;
    let o1 = (x0.broadcast_mul(&sin)? + x1.broadcast_mul(&cos)?)?;

    // Interleave back
    let o0 = o0.unsqueeze(candle_core::D::Minus1)?; // [b*h, seq, half, 1]
    let o1 = o1.unsqueeze(candle_core::D::Minus1)?;
    let out = Tensor::cat(&[&o0, &o1], candle_core::D::Minus1)?; // [b*h, seq, half, 2]
    out.reshape(x.shape())
}

/// Apply MRoPE (multimodal rotary position embeddings) to Q and K.
///
/// MRoPE splits head_dim into `mrope_section` chunks. Each section gets
/// its own position_ids. For TTS, `mrope_section = [24, 20, 20]` means:
/// - Dims 0..48: temporal positions
/// - Dims 48..88: "height" positions (often same as temporal for audio)
/// - Dims 88..128: "width" positions (often same as temporal for audio)
///
/// `x` shape: `[batch, num_heads, seq_len, head_dim]`
/// `position_ids` shape: `[3, seq_len]` — one row per modality.
/// Each section's cos/sin is computed from its own positions using the
/// corresponding slice of `inv_freq`.
pub fn apply_mrope(
    x: &Tensor,
    rotary: &mut RotaryEmbedding,
    position_ids: &Tensor,
    mrope_section: &[usize],
) -> Result<Tensor> {
    let (_b, _h, seq_len, _head_dim) = x.dims4()?;
    let _device = x.device();
    let dtype = x.dtype();

    // We need to build per-section cos/sin from each modality's position_ids
    // and the corresponding slice of inv_freq.
    rotary.ensure_len(seq_len + 4096)?; // ensure cache is large enough

    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    let mut freq_offset = 0;

    for (section_idx, &section_size) in mrope_section.iter().enumerate() {
        // Get position IDs for this modality: shape [seq_len]
        let pos_ids = position_ids.i(section_idx)?; // [seq_len]
        let pos_ids = pos_ids.to_dtype(DType::F32)?;

        // Slice inv_freq for this section: [section_size]
        let section_inv_freq = rotary.inv_freq.i(freq_offset..freq_offset + section_size)?;
        freq_offset += section_size;

        // Compute freqs: [seq_len, section_size]
        let pos_col = pos_ids.unsqueeze(1)?; // [seq_len, 1]
        let freq_row = section_inv_freq.unsqueeze(0)?; // [1, section_size]
        let freqs = pos_col.matmul(&freq_row)?;

        cos_parts.push(freqs.cos()?.to_dtype(dtype)?);
        sin_parts.push(freqs.sin()?.to_dtype(dtype)?);
    }

    // Concatenate sections: [seq_len, head_dim/2]
    let cos_full = Tensor::cat(&cos_parts, 1)?;
    let sin_full = Tensor::cat(&sin_parts, 1)?;

    apply_rotary_emb(x, &cos_full, &sin_full)
}

/// Build simple sequential position IDs for MRoPE.
///
/// For TTS, all 3 modalities typically use the same sequential positions.
/// Returns shape `[3, seq_len]` on the given device.
pub fn make_mrope_position_ids(seq_len: usize, offset: usize, device: &Device) -> Result<Tensor> {
    let positions: Vec<u32> = (0..seq_len).map(|i| (i + offset) as u32).collect();
    let row = Tensor::new(positions, device)?;
    // Stack 3 copies (one per modality)
    Tensor::stack(&[&row, &row, &row], 0)
}

/// Pre-compute MRoPE cos/sin for all sections, concatenated into a single
/// `[seq_len, head_dim/2]` tensor pair.
///
/// This avoids needing a `&mut RotaryEmbedding` inside per-layer closures.
pub fn precompute_mrope_cos_sin(
    rotary: &mut RotaryEmbedding,
    position_ids: &Tensor,
    mrope_section: &[usize],
    dtype: DType,
) -> Result<(Tensor, Tensor)> {
    rotary.ensure_len(8192)?; // ensure cache is large enough

    let mut cos_parts = Vec::new();
    let mut sin_parts = Vec::new();
    let mut freq_offset = 0;

    for (section_idx, &section_size) in mrope_section.iter().enumerate() {
        let pos_ids = position_ids.i(section_idx)?;
        let pos_ids = pos_ids.to_dtype(DType::F32)?;

        let section_inv_freq = rotary.inv_freq.i(freq_offset..freq_offset + section_size)?;
        freq_offset += section_size;

        let pos_col = pos_ids.unsqueeze(1)?;
        let freq_row = section_inv_freq.unsqueeze(0)?;
        let freqs = pos_col.matmul(&freq_row)?;

        cos_parts.push(freqs.cos()?.to_dtype(dtype)?);
        sin_parts.push(freqs.sin()?.to_dtype(dtype)?);
    }

    let cos_full = Tensor::cat(&cos_parts, 1)?;
    let sin_full = Tensor::cat(&sin_parts, 1)?;
    Ok((cos_full, sin_full))
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn test_rotary_embedding_creation() {
        let rope = RotaryEmbedding::new(128, 1_000_000.0, 256, &Device::Cpu).unwrap();
        assert_eq!(rope.cos_cache.dims(), &[256, 64]);
        assert_eq!(rope.sin_cache.dims(), &[256, 64]);
    }

    #[test]
    fn test_mrope_position_ids_shape() {
        let pos = make_mrope_position_ids(10, 0, &Device::Cpu).unwrap();
        assert_eq!(pos.dims(), &[3, 10]);
    }
}
