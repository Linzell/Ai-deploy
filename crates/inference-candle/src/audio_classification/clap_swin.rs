//! Swin Transformer building blocks for CLAP's HTS-AT audio encoder.
//!
//! Implements:
//! - Window partition / reverse
//! - Windowed multi-head self-attention with relative position bias
//! - Swin Transformer block (with optional shifted windows)
//! - Patch merging (spatial downsampling between stages)
//! - Swin stage (sequence of blocks + optional downsampling)
//!
//! Weight names follow the HuggingFace CLAP checkpoint layout:
//! `audio_model.audio_encoder.layers.{stage}.blocks.{block}.*`

// ML code uses standard mathematical variable names (b, h, w, c, q, k, v)
// and usize→i64 casts are safe for realistic tensor dimensions.
#![allow(clippy::many_single_char_names, clippy::cast_possible_wrap)]

use candle_core::{IndexOp, Tensor};
use candle_nn::VarBuilder;

use crate::{TaskError, TaskResult};

// ---------------------------------------------------------------------------
// Window helpers
// ---------------------------------------------------------------------------

/// Partition a 4D feature map `[B, H, W, C]` into windows of `[B*nW, ws*ws, C]`.
pub fn window_partition(x: &Tensor, window_size: usize) -> candle_core::Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    let nh = h / window_size;
    let nw = w / window_size;
    let ws = window_size;
    // [B, nh, ws, nw, ws, C]
    let x = x.reshape((b, nh, ws, nw, ws, c))?;
    // [B, nh, nw, ws, ws, C]
    let x = x.permute((0, 1, 3, 2, 4, 5))?;
    // [B*nh*nw, ws*ws, C]
    x.reshape((b * nh * nw, ws * ws, c))
}

/// Reverse window partition: `[B*nW, ws*ws, C]` → `[B, H, W, C]`.
pub fn window_reverse(
    windows: &Tensor,
    window_size: usize,
    h: usize,
    w: usize,
    batch: usize,
) -> candle_core::Result<Tensor> {
    let ws = window_size;
    let nh = h / ws;
    let nw = w / ws;
    let c = windows.dim(2)?;
    // [B, nh, nw, ws, ws, C]
    let x = windows.reshape((batch, nh, nw, ws, ws, c))?;
    // [B, nh, ws, nw, ws, C]
    let x = x.permute((0, 1, 3, 2, 4, 5))?;
    // [B, H, W, C]
    x.reshape((batch, h, w, c))
}

// ---------------------------------------------------------------------------
// Compute relative position index (done once per window size)
// ---------------------------------------------------------------------------

/// Build the relative position index table for a given window size.
///
/// Returns a 1D tensor of shape `[ws*ws * ws*ws]` containing indices
/// into the relative_position_bias_table. These are the same indices
/// HuggingFace precomputes and stores as a buffer.
pub fn build_relative_position_index(window_size: usize) -> Vec<i64> {
    let ws = window_size as i64;
    let n = (ws * ws) as usize;
    let mut index = vec![0i64; n * n];

    for row in 0..ws {
        for col in 0..ws {
            let i = (row * ws + col) as usize;
            for row2 in 0..ws {
                for col2 in 0..ws {
                    let j = (row2 * ws + col2) as usize;
                    let dy = row - row2 + ws - 1;
                    let dx = col - col2 + ws - 1;
                    index[i * n + j] = dy * (2 * ws - 1) + dx;
                }
            }
        }
    }
    index
}

// ---------------------------------------------------------------------------
// Attention mask for shifted windows
// ---------------------------------------------------------------------------

/// Create the attention mask for shifted window self-attention.
///
/// Returns a tensor of shape `[nW, ws*ws, ws*ws]` where masked positions
/// are set to -100.0 and valid positions are 0.0.
pub fn compute_shift_mask(
    h: usize,
    w: usize,
    window_size: usize,
    shift_size: usize,
) -> candle_core::Result<Tensor> {
    let ws = window_size;
    // Build region label map [H, W]
    let mut img_mask = vec![0i64; h * w];
    let mut cnt = 0i64;

    // The regions are defined by slicing h/w into 3 bands each:
    // [0..h-ws], [h-ws..h-shift], [h-shift..h]  ×  same for w
    let h_slices = [(0, h - ws), (h - ws, h - shift_size), (h - shift_size, h)];
    let w_slices = [(0, w - ws), (w - ws, w - shift_size), (w - shift_size, w)];

    for &(hs, he) in &h_slices {
        for &(ws_start, ws_end) in &w_slices {
            for r in hs..he {
                for c in ws_start..ws_end {
                    img_mask[r * w + c] = cnt;
                }
            }
            cnt += 1;
        }
    }

    // Partition into windows
    let nh = h / ws;
    let nw = w / ws;
    let n_windows = nh * nw;
    let ww = ws * ws;

    let mut mask_windows = vec![0i64; n_windows * ww];
    for wi in 0..nh {
        for wj in 0..nw {
            let win_idx = wi * nw + wj;
            for di in 0..ws {
                for dj in 0..ws {
                    let r = wi * ws + di;
                    let c = wj * ws + dj;
                    mask_windows[win_idx * ww + di * ws + dj] = img_mask[r * w + c];
                }
            }
        }
    }

    // Build attention mask: [nW, ww, ww]
    // mask[nw][i][j] = -100 if region[i] != region[j], else 0
    let mut attn_mask = vec![0.0f32; n_windows * ww * ww];
    for nw_idx in 0..n_windows {
        for i in 0..ww {
            for j in 0..ww {
                let ri = mask_windows[nw_idx * ww + i];
                let rj = mask_windows[nw_idx * ww + j];
                if ri != rj {
                    attn_mask[nw_idx * ww * ww + i * ww + j] = -100.0;
                }
            }
        }
    }

    Tensor::from_vec(attn_mask, (n_windows, ww, ww), &candle_core::Device::Cpu)
}

// ---------------------------------------------------------------------------
// Swin Self-Attention
// ---------------------------------------------------------------------------

/// Windowed multi-head self-attention with relative position bias.
struct SwinAttention {
    qkv_weight: Tensor,
    qkv_bias: Option<Tensor>,
    proj_weight: Tensor,
    proj_bias: Tensor,
    relative_position_bias_table: Tensor, // [(2*ws-1)*(2*ws-1), num_heads]
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SwinAttention {
    fn load(vb: &VarBuilder, dim: usize, num_heads: usize, qkv_bias: bool) -> TaskResult<Self> {
        let head_dim = dim / num_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();
        let vb_self = vb.pp("attention").pp("self");

        // CLAP stores separate Q, K, V weights (not fused QKV)
        let q_w = vb_self
            .get((dim, dim), "query.weight")
            .map_err(|e| TaskError::ModelLoad(format!("query.weight: {e}")))?;
        let k_w = vb_self
            .get((dim, dim), "key.weight")
            .map_err(|e| TaskError::ModelLoad(format!("key.weight: {e}")))?;
        let v_w = vb_self
            .get((dim, dim), "value.weight")
            .map_err(|e| TaskError::ModelLoad(format!("value.weight: {e}")))?;
        let qkv_weight = Tensor::cat(&[&q_w, &k_w, &v_w], 0)
            .map_err(|e| TaskError::ModelLoad(format!("cat qkv weights: {e}")))?;

        let qkv_bias = if qkv_bias {
            let q_b = vb_self
                .get(dim, "query.bias")
                .map_err(|e| TaskError::ModelLoad(format!("query.bias: {e}")))?;
            let k_b = vb_self
                .get(dim, "key.bias")
                .map_err(|e| TaskError::ModelLoad(format!("key.bias: {e}")))?;
            let v_b = vb_self
                .get(dim, "value.bias")
                .map_err(|e| TaskError::ModelLoad(format!("value.bias: {e}")))?;
            Some(
                Tensor::cat(&[&q_b, &k_b, &v_b], 0)
                    .map_err(|e| TaskError::ModelLoad(format!("cat qkv biases: {e}")))?,
            )
        } else {
            None
        };

        let vb_out = vb.pp("attention").pp("output");
        let proj_weight = vb_out
            .get((dim, dim), "dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("output.dense.weight: {e}")))?;
        let proj_bias = vb_out
            .get(dim, "dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("output.dense.bias: {e}")))?;

        let rp_table_size = (2 * 8 - 1) * (2 * 8 - 1); // We read actual shape from weights
        let relative_position_bias_table = vb_self
            .get((rp_table_size, num_heads), "relative_position_bias_table")
            .map_err(|e| TaskError::ModelLoad(format!("relative_position_bias_table: {e}")))?;

        Ok(Self {
            qkv_weight,
            qkv_bias,
            proj_weight,
            proj_bias,
            relative_position_bias_table,
            num_heads,
            head_dim,
            scale,
        })
    }

    /// Forward: `[B*nW, ws*ws, C]` → `[B*nW, ws*ws, C]`
    ///
    /// `relative_position_index`: `[ws*ws, ws*ws]` (i64 indices)
    /// `attn_mask`: optional `[nW, ws*ws, ws*ws]`
    fn forward(
        &self,
        x: &Tensor,
        relative_position_index: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (bnw, n, _c) = x.dims3()?;
        let device = x.device();

        // QKV projection: [B*nW, N, C] @ [3C, C]^T → [B*nW, N, 3C]
        let qkv = x
            .contiguous()?
            .broadcast_matmul(&self.qkv_weight.t()?.contiguous()?)?;
        let qkv = if let Some(bias) = &self.qkv_bias {
            qkv.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)?
        } else {
            qkv
        };

        // Reshape to [B*nW, N, 3, num_heads, head_dim] → [3, B*nW, num_heads, N, head_dim]
        let qkv = qkv.reshape((bnw, n, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        // Attention scores: [B*nW, num_heads, N, N]
        let q_scaled = (q.contiguous()? * self.scale)?;
        let attn = q_scaled.matmul(&k.t()?.contiguous()?)?;

        // Add relative position bias: [N, N, num_heads]
        // Index into table: relative_position_index is [N*N] of i64
        let rpi_flat = relative_position_index.flatten_all()?;
        let bias = self
            .relative_position_bias_table
            .index_select(&rpi_flat, 0)?;
        // [N*N, num_heads] → [N, N, num_heads] → [num_heads, N, N]
        let bias = bias
            .reshape((n, n, self.num_heads))?
            .permute((2, 0, 1))?
            .contiguous()?;
        // Broadcast add: [B*nW, num_heads, N, N] + [1, num_heads, N, N]
        let attn = attn.broadcast_add(&bias.unsqueeze(0)?)?;

        // Apply shifted-window attention mask if present
        let attn = if let Some(mask) = attn_mask {
            // mask: [nW, N, N] → need to reshape attn for broadcasting
            let n_windows = mask.dim(0)?;
            let batch = bnw / n_windows;
            // [batch, nW, num_heads, N, N]
            let attn = attn.reshape((batch, n_windows, self.num_heads, n, n))?;
            // mask: [1, nW, 1, N, N]
            let mask = mask
                .to_dtype(attn.dtype())?
                .to_device(device)?
                .unsqueeze(0)?
                .unsqueeze(2)?;
            let attn = attn.broadcast_add(&mask)?;
            attn.reshape((bnw, self.num_heads, n, n))?
        } else {
            attn
        };

        // Softmax over last dim
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Weighted values: [B*nW, num_heads, N, head_dim]
        let out = attn.matmul(&v.contiguous()?)?;
        // [B*nW, N, num_heads, head_dim] → [B*nW, N, C]
        let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((
            bnw,
            n,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        out.contiguous()?
            .broadcast_matmul(&self.proj_weight.t()?.contiguous()?)?
            .broadcast_add(&self.proj_bias.unsqueeze(0)?.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Swin MLP
// ---------------------------------------------------------------------------

struct SwinMlp {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
}

impl SwinMlp {
    fn load(vb: &VarBuilder, in_dim: usize, mlp_ratio: f64) -> TaskResult<Self> {
        let hidden = (in_dim as f64 * mlp_ratio) as usize;
        let fc1_weight = vb
            .get((hidden, in_dim), "intermediate.dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("intermediate.dense.weight: {e}")))?;
        let fc1_bias = vb
            .get(hidden, "intermediate.dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("intermediate.dense.bias: {e}")))?;
        let fc2_weight = vb
            .get((in_dim, hidden), "output.dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("output.dense.weight: {e}")))?;
        let fc2_bias = vb
            .get(in_dim, "output.dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("output.dense.bias: {e}")))?;
        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x
            .contiguous()?
            .broadcast_matmul(&self.fc1_weight.t()?.contiguous()?)?
            .broadcast_add(&self.fc1_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let x = x.gelu_erf()?;
        x.contiguous()?
            .broadcast_matmul(&self.fc2_weight.t()?.contiguous()?)?
            .broadcast_add(&self.fc2_bias.unsqueeze(0)?.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// LayerNorm helper (loads weight/bias from VarBuilder)
// ---------------------------------------------------------------------------

struct LnParams {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LnParams {
    fn load(vb: &VarBuilder, dim: usize, name: &str, eps: f64) -> TaskResult<Self> {
        let weight = vb
            .get(dim, &format!("{name}.weight"))
            .map_err(|e| TaskError::ModelLoad(format!("{name}.weight: {e}")))?;
        let bias = vb
            .get(dim, &format!("{name}.bias"))
            .map_err(|e| TaskError::ModelLoad(format!("{name}.bias: {e}")))?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x.contiguous()?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = (&x_centered * &x_centered)?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let norm = x_centered.broadcast_div(&std)?;
        norm.broadcast_mul(&self.weight.unsqueeze(0)?.unsqueeze(0)?)?
            .broadcast_add(&self.bias.unsqueeze(0)?.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Swin Transformer Block
// ---------------------------------------------------------------------------

/// A single Swin Transformer block.
///
/// Uses windowed self-attention (optionally shifted) + MLP + residual.
pub struct SwinBlock {
    ln_before: LnParams,
    attn: SwinAttention,
    ln_after: LnParams,
    mlp: SwinMlp,
    shift_size: usize,
    window_size: usize,
}

impl SwinBlock {
    pub fn load(
        vb: &VarBuilder,
        dim: usize,
        num_heads: usize,
        window_size: usize,
        shift_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
    ) -> TaskResult<Self> {
        let ln_before = LnParams::load(vb, dim, "layernorm_before", 1e-5)?;
        let attn = SwinAttention::load(vb, dim, num_heads, qkv_bias)?;
        let ln_after = LnParams::load(vb, dim, "layernorm_after", 1e-5)?;
        let mlp = SwinMlp::load(vb, dim, mlp_ratio)?;
        Ok(Self {
            ln_before,
            attn,
            ln_after,
            mlp,
            shift_size,
            window_size,
        })
    }

    /// Forward: `[B, H*W, C]` → `[B, H*W, C]`
    pub fn forward(
        &self,
        x: &Tensor,
        h: usize,
        w: usize,
        relative_position_index: &Tensor,
        attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b, _hw, c) = x.dims3()?;
        let ws = self.window_size;

        let shortcut = x.clone();
        let x = self.ln_before.forward(x)?;

        // Reshape to spatial: [B, H, W, C]
        let x = x.reshape((b, h, w, c))?;

        // Cyclic shift
        let x = if self.shift_size > 0 {
            // Roll along H and W dims
            roll_2d(&x, -(self.shift_size as i64), -(self.shift_size as i64))?
        } else {
            x
        };

        // Partition into windows: [B*nW, ws*ws, C]
        let x_windows = window_partition(&x, ws)?;

        // Windowed attention with mask for shifted windows
        let mask = if self.shift_size > 0 { attn_mask } else { None };
        let attn_out = self
            .attn
            .forward(&x_windows, relative_position_index, mask)?;

        // Reverse windows: [B, H, W, C]
        let x = window_reverse(&attn_out, ws, h, w, b)?;

        // Reverse cyclic shift
        let x = if self.shift_size > 0 {
            roll_2d(&x, self.shift_size as i64, self.shift_size as i64)?
        } else {
            x
        };

        // Back to [B, H*W, C]
        let x = x.reshape((b, h * w, c))?;

        // Residual + MLP
        let x = (shortcut + x)?;
        let residual = x.clone();
        let x = self.ln_after.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        residual + x
    }
}

// ---------------------------------------------------------------------------
// Cyclic shift (roll) for 2D spatial tensor
// ---------------------------------------------------------------------------

/// Roll a `[B, H, W, C]` tensor along H (dim=1) and W (dim=2) by `shift_h` and `shift_w`.
fn roll_2d(x: &Tensor, shift_h: i64, shift_w: i64) -> candle_core::Result<Tensor> {
    let x = roll_along(x, shift_h, 1)?;
    roll_along(&x, shift_w, 2)
}

/// Roll a tensor along a single dimension by `shift` positions.
fn roll_along(x: &Tensor, shift: i64, dim: usize) -> candle_core::Result<Tensor> {
    let size = x.dim(dim)? as i64;
    if size == 0 || shift % size == 0 {
        return Ok(x.clone());
    }
    let shift = ((shift % size) + size) % size; // normalize to positive
    let split_point = (size - shift) as usize;
    let a = x.narrow(dim, 0, split_point)?;
    let b = x.narrow(dim, split_point, size as usize - split_point)?;
    Tensor::cat(&[&b, &a], dim)
}

// ---------------------------------------------------------------------------
// Patch Merging (spatial downsampling between stages)
// ---------------------------------------------------------------------------

/// PatchMerging: takes 2×2 spatial neighbors, concatenates → Linear(4C, 2C).
///
/// Input: `[B, H*W, C]` → Output: `[B, (H/2)*(W/2), 2C]`
pub struct PatchMerging {
    norm: LnParams,
    reduction_weight: Tensor,
}

impl PatchMerging {
    pub fn load(vb: &VarBuilder, dim: usize) -> TaskResult<Self> {
        let norm = LnParams::load(vb, 4 * dim, "norm", 1e-5)?;
        let reduction_weight = vb
            .get((2 * dim, 4 * dim), "reduction.weight")
            .map_err(|e| TaskError::ModelLoad(format!("reduction.weight: {e}")))?;
        Ok(Self {
            norm,
            reduction_weight,
        })
    }

    /// Forward: `[B, H*W, C]` with spatial dims `(H, W)` → `[B, H/2 * W/2, 2C]`
    pub fn forward(&self, x: &Tensor, h: usize, w: usize) -> candle_core::Result<Tensor> {
        let (b, _hw, c) = x.dims3()?;
        let x = x.reshape((b, h, w, c))?;

        // Gather 2×2 patches using narrow (select even/odd rows and columns)
        // x0 = x[:, 0::2, 0::2, :] (even rows, even cols)
        // x1 = x[:, 1::2, 0::2, :] (odd rows, even cols)
        // x2 = x[:, 0::2, 1::2, :] (even rows, odd cols)
        // x3 = x[:, 1::2, 1::2, :] (odd rows, odd cols)
        let new_h = h / 2;
        let new_w = w / 2;

        // Build indices for even/odd selection
        let device = x.device();
        let even_rows: Vec<u32> = (0..h as u32).step_by(2).collect();
        let odd_rows: Vec<u32> = (1..h as u32).step_by(2).collect();
        let even_cols: Vec<u32> = (0..w as u32).step_by(2).collect();
        let odd_cols: Vec<u32> = (1..w as u32).step_by(2).collect();

        let even_rows_t = Tensor::from_vec(even_rows, (new_h,), device)?;
        let odd_rows_t = Tensor::from_vec(odd_rows, (new_h,), device)?;
        let even_cols_t = Tensor::from_vec(even_cols, (new_w,), device)?;
        let odd_cols_t = Tensor::from_vec(odd_cols, (new_w,), device)?;

        // index_select along dim 1 (rows), then dim 2 (cols)
        let x0 = x
            .index_select(&even_rows_t, 1)?
            .index_select(&even_cols_t, 2)?;
        let x1 = x
            .index_select(&odd_rows_t, 1)?
            .index_select(&even_cols_t, 2)?;
        let x2 = x
            .index_select(&even_rows_t, 1)?
            .index_select(&odd_cols_t, 2)?;
        let x3 = x
            .index_select(&odd_rows_t, 1)?
            .index_select(&odd_cols_t, 2)?;

        // Concat along channel dim: [B, H/2, W/2, 4C]
        let x = Tensor::cat(&[&x0, &x1, &x2, &x3], 3)?;
        // Flatten spatial: [B, H/2 * W/2, 4C]
        let x = x.reshape((b, new_h * new_w, 4 * c))?;

        // LayerNorm
        let x = self.norm.forward(&x)?;

        // Linear reduction: [B, H/2*W/2, 4C] @ [2C, 4C]^T → [B, H/2*W/2, 2C]
        x.contiguous()?
            .broadcast_matmul(&self.reduction_weight.t()?.contiguous()?)
    }
}

// ---------------------------------------------------------------------------
// Swin Stage (sequence of blocks + optional downsampling)
// ---------------------------------------------------------------------------

/// One stage of the Swin Transformer.
///
/// Contains N blocks with alternating regular/shifted window attention,
/// followed by optional PatchMerging for downsampling.
pub struct SwinStage {
    blocks: Vec<SwinBlock>,
    downsample: Option<PatchMerging>,
    _window_size: usize,
    relative_position_index: Tensor, // [ws*ws, ws*ws]
    attn_mask: Option<Tensor>,       // [nW, ws*ws, ws*ws] for shifted window blocks
}

impl SwinStage {
    /// Load a Swin stage.
    ///
    /// `stage_idx`: 0-based stage index
    /// `dim`: channel dimension at this stage
    /// `depth`: number of blocks
    /// `num_heads`: attention heads at this stage
    /// `h, w`: spatial resolution at this stage
    /// `is_last`: if true, no PatchMerging appended
    #[allow(clippy::too_many_arguments)]
    pub fn load(
        vb: &VarBuilder,
        dim: usize,
        depth: usize,
        num_heads: usize,
        window_size: usize,
        mlp_ratio: f64,
        qkv_bias: bool,
        h: usize,
        w: usize,
        is_last: bool,
    ) -> TaskResult<Self> {
        let shift_size = window_size / 2;

        let mut blocks = Vec::with_capacity(depth);
        for i in 0..depth {
            let block_shift = if i % 2 == 0 { 0 } else { shift_size };
            let block = SwinBlock::load(
                &vb.pp("blocks").pp(i.to_string()),
                dim,
                num_heads,
                window_size,
                block_shift,
                mlp_ratio,
                qkv_bias,
            )?;
            blocks.push(block);
        }

        let downsample = if is_last {
            None
        } else {
            Some(PatchMerging::load(&vb.pp("downsample"), dim)?)
        };

        // Build relative position index
        let rpi_vec = build_relative_position_index(window_size);
        let ws2 = window_size * window_size;
        let relative_position_index =
            Tensor::from_vec(rpi_vec, (ws2, ws2), &candle_core::Device::Cpu)
                .map_err(|e| TaskError::ModelLoad(format!("rpi tensor: {e}")))?;

        // Build attention mask for shifted windows (if any block uses shift)
        let attn_mask = if depth > 1 && h >= window_size && w >= window_size {
            // Need mask for shifted blocks
            Some(
                compute_shift_mask(h, w, window_size, shift_size)
                    .map_err(|e| TaskError::ModelLoad(format!("shift mask: {e}")))?,
            )
        } else {
            None
        };

        Ok(Self {
            blocks,
            downsample,
            _window_size: window_size,
            relative_position_index,
            attn_mask,
        })
    }

    /// Forward pass through all blocks + optional downsampling.
    ///
    /// Returns `(output, new_h, new_w)`.
    pub fn forward(
        &self,
        x: &Tensor,
        h: usize,
        w: usize,
    ) -> candle_core::Result<(Tensor, usize, usize)> {
        // Move index/mask to device if needed
        let device = x.device();
        let rpi = self.relative_position_index.to_device(device)?;
        let mask = self
            .attn_mask
            .as_ref()
            .map(|m| m.to_dtype(x.dtype()).and_then(|m| m.to_device(device)))
            .transpose()?;

        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, h, w, &rpi, mask.as_ref())?;
        }

        if let Some(ds) = &self.downsample {
            let x = ds.forward(&x, h, w)?;
            Ok((x, h / 2, w / 2))
        } else {
            Ok((x, h, w))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_relative_position_index_shape() {
        let idx = build_relative_position_index(8);
        assert_eq!(idx.len(), 64 * 64);
        // All indices should be in [0, (2*8-1)*(2*8-1))
        let max_idx = (2 * 8 - 1) * (2 * 8 - 1);
        assert!(idx.iter().all(|&v| v >= 0 && v < max_idx as i64));
    }

    #[test]
    fn test_window_partition_reverse_roundtrip() {
        let device = candle_core::Device::Cpu;
        let b = 1;
        let h = 16;
        let w = 16;
        let c = 32;
        let ws = 8;
        let data: Vec<f32> = (0..b * h * w * c).map(|i| i as f32).collect();
        let x = Tensor::from_vec(data, (b, h, w, c), &device).unwrap();
        let windows = window_partition(&x, ws).unwrap();
        let expected_nw = (h / ws) * (w / ws);
        assert_eq!(windows.dims(), &[b * expected_nw, ws * ws, c]);
        let recovered = window_reverse(&windows, ws, h, w, b).unwrap();
        assert_eq!(recovered.dims(), &[b, h, w, c]);
        // Values should match
        let orig: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let rec: Vec<f32> = recovered.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(orig, rec);
    }

    #[test]
    fn test_shift_mask_shape() {
        let mask = compute_shift_mask(16, 64, 8, 4).unwrap();
        let nw = (16 / 8) * (64 / 8);
        assert_eq!(mask.dims(), &[nw, 64, 64]);
    }

    #[test]
    fn test_roll_along_identity() {
        let device = candle_core::Device::Cpu;
        let x = Tensor::arange(0f32, 12.0, &device)
            .unwrap()
            .reshape((3, 4))
            .unwrap();
        let rolled = roll_along(&x, 0, 1).unwrap();
        let a: Vec<f32> = x.flatten_all().unwrap().to_vec1().unwrap();
        let b: Vec<f32> = rolled.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_roll_along_positive() {
        let device = candle_core::Device::Cpu;
        // [0, 1, 2, 3] rolled by 1 along dim 0 → [3, 0, 1, 2]
        let x = Tensor::arange(0f32, 4.0, &device)
            .unwrap()
            .reshape((4,))
            .unwrap();
        // For 1D roll along dim 0
        let rolled = roll_along(&x.unsqueeze(1).unwrap(), 1, 0).unwrap();
        let vals: Vec<f32> = rolled.flatten_all().unwrap().to_vec1().unwrap();
        assert_eq!(vals, vec![3.0, 0.0, 1.0, 2.0]);
    }
}
