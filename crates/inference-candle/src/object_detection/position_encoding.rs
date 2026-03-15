//! Sine positional encoding for 2D feature maps.
//!
//! This generates fixed (non-learnable) sinusoidal position embeddings
//! for spatial positions in a 2D feature map, as used in DETR.
//!
//! The output is of shape `[batch, d_model, H, W]` which gets flattened
//! to `[batch, H*W, d_model]` for the transformer.

use candle_core::{DType, Device, Result, Tensor};

/// Generate sine positional embeddings for a 2D feature map.
///
/// # Arguments
/// - `batch_size` — batch size
/// - `height`, `width` — spatial dimensions of the feature map
/// - `num_pos_features` — d_model / 2 (half for x, half for y)
/// - `temperature` — frequency scaling (default 10000)
/// - `device` — target device
/// - `dtype` — target dtype
///
/// # Returns
/// Tensor of shape `[batch, H*W, d_model]`
pub fn sine_position_embedding(
    batch_size: usize,
    height: usize,
    width: usize,
    num_pos_features: usize,
    temperature: f64,
    device: &Device,
    dtype: DType,
) -> Result<Tensor> {
    // Create y and x coordinate grids, normalized to [0, scale]
    // mask is all zeros (no padding), so cumsum gives 1..H and 1..W
    let scale = 2.0 * std::f64::consts::PI;

    // y_embed: [H, W] where each row i has value (i+1)
    // x_embed: [H, W] where each col j has value (j+1)
    let mut y_vals = vec![0f32; height * width];
    let mut x_vals = vec![0f32; height * width];
    for i in 0..height {
        for j in 0..width {
            y_vals[i * width + j] = (i + 1) as f32;
            x_vals[i * width + j] = (j + 1) as f32;
        }
    }

    let y_embed = Tensor::from_vec(y_vals, (height, width), device)?.to_dtype(dtype)?;
    let x_embed = Tensor::from_vec(x_vals, (height, width), device)?.to_dtype(dtype)?;

    // Normalize to [0, 1] and scale
    let eps = 1e-6;
    let y_max = height as f64 + eps;
    let x_max = width as f64 + eps;
    let y_embed = ((y_embed / y_max)? * scale)?;
    let x_embed = ((x_embed / x_max)? * scale)?;

    // dim_t: [num_pos_features]
    // dim_t[i] = temperature^(2 * floor(i/2) / num_pos_features)
    let dim_t: Vec<f32> = (0..num_pos_features)
        .map(|i| {
            let exp = 2.0 * (i / 2) as f64 / num_pos_features as f64;
            temperature.powf(exp) as f32
        })
        .collect();
    let dim_t = Tensor::from_vec(dim_t, num_pos_features, device)?.to_dtype(dtype)?;

    // pos_x: [H, W, num_pos_features] = x_embed[:,:,None] / dim_t
    let x_expanded = x_embed.unsqueeze(2)?; // [H, W, 1]
    let dim_t_expanded = dim_t.unsqueeze(0)?.unsqueeze(0)?; // [1, 1, D]
    let pos_x = x_expanded.broadcast_div(&dim_t_expanded)?; // [H, W, D]

    // pos_y: [H, W, num_pos_features]
    let y_expanded = y_embed.unsqueeze(2)?;
    let pos_y = y_expanded.broadcast_div(&dim_t_expanded)?;

    // Apply sin to even indices, cos to odd indices
    // In HF DETR: pos_x[:,:,0::2].sin(), pos_x[:,:,1::2].cos() then stack+flatten
    // Candle doesn't support step indexing, so we use index_select
    let even_indices: Vec<u32> = (0..num_pos_features)
        .filter(|i| i % 2 == 0)
        .map(|i| i as u32)
        .collect();
    let odd_indices: Vec<u32> = (0..num_pos_features)
        .filter(|i| i % 2 == 1)
        .map(|i| i as u32)
        .collect();

    let even_idx = Tensor::from_vec(even_indices.clone(), even_indices.len(), device)?;
    let odd_idx = Tensor::from_vec(odd_indices.clone(), odd_indices.len(), device)?;

    // pos_x sin/cos interleaving
    let pos_x_sin = pos_x.index_select(&even_idx, 2)?.sin()?; // [H, W, D/2]
    let pos_x_cos = pos_x.index_select(&odd_idx, 2)?.cos()?; // [H, W, D/2]

    // pos_y sin/cos interleaving
    let pos_y_sin = pos_y.index_select(&even_idx, 2)?.sin()?;
    let pos_y_cos = pos_y.index_select(&odd_idx, 2)?.cos()?;

    // Interleave: [H, W, D] where positions alternate sin/cos
    // Stack along dim=3 and flatten → [H, W, D]
    let half = even_indices.len();
    let pos_x_interleaved =
        interleave_sin_cos(&pos_x_sin, &pos_x_cos, half, height, width, device, dtype)?;
    let pos_y_interleaved =
        interleave_sin_cos(&pos_y_sin, &pos_y_cos, half, height, width, device, dtype)?;

    // Concatenate y and x: [H, W, 2*num_pos_features] = [H, W, d_model]
    let pos = Tensor::cat(&[&pos_y_interleaved, &pos_x_interleaved], 2)?; // [H, W, d_model]

    // Reshape to [1, H*W, d_model] and expand to batch
    let d_model = num_pos_features * 2;
    let pos = pos.reshape((1, height * width, d_model))?;

    if batch_size > 1 {
        pos.broadcast_as((batch_size, height * width, d_model))?
            .contiguous()
    } else {
        Ok(pos)
    }
}

/// Interleave sin and cos tensors along the last dimension.
///
/// sin: [H, W, D/2], cos: [H, W, D/2] → out: [H, W, D]
/// where out[:,:,0] = sin[:,:,0], out[:,:,1] = cos[:,:,0], out[:,:,2] = sin[:,:,1], ...
fn interleave_sin_cos(
    sin: &Tensor,
    cos: &Tensor,
    half: usize,
    height: usize,
    width: usize,
    _device: &Device,
    _dtype: DType,
) -> Result<Tensor> {
    // Stack sin and cos along a new dim=3: [H, W, D/2, 2]
    let sin_expanded = sin.unsqueeze(3)?; // [H, W, D/2, 1]
    let cos_expanded = cos.unsqueeze(3)?;
    let stacked = Tensor::cat(&[&sin_expanded, &cos_expanded], 3)?; // [H, W, D/2, 2]

    // Reshape to [H, W, D]
    stacked.reshape((height, width, half * 2))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sine_position_embedding_shape() {
        let pos =
            sine_position_embedding(2, 25, 25, 128, 10000.0, &Device::Cpu, DType::F32).unwrap();
        assert_eq!(pos.dims(), &[2, 625, 256]); // batch=2, 25*25=625, d_model=256
    }

    #[test]
    fn test_sine_position_embedding_bounded() {
        let pos = sine_position_embedding(1, 4, 4, 4, 10000.0, &Device::Cpu, DType::F32).unwrap();
        let vals = pos.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        // All values should be in [-1, 1] (sin/cos range)
        for v in &vals {
            assert!(
                *v >= -1.0 - 1e-6 && *v <= 1.0 + 1e-6,
                "Value out of range: {v}"
            );
        }
    }
}
