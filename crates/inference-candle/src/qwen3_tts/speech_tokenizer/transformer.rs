//! Pre-transformer for the speech tokenizer decoder.
//!
//! 8-layer transformer with:
//! - RMSNorm (pre-norm)
//! - Multi-head attention (16 heads, head_dim=64, hidden=512), no bias, with RoPE
//! - SwiGLU MLP (hidden=512, intermediate=1024)
//! - Layer-scale on both attention and MLP outputs
//!
//! Wrapping projections:
//!   input_proj:  Linear(1024 → 512, bias)
//!   output_proj: Linear(512 → 1024, bias)

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

// ---------------------------------------------------------------------------
// Helper: linear projection that handles 3D input with 2D weight
// ---------------------------------------------------------------------------

/// Compute `x @ weight.T`, collapsing batch dims so candle's matmul works.
///
/// `x`: `[..., in_dim]`, `weight`: `[out_dim, in_dim]`.
/// Returns `[..., out_dim]`.
fn linear_fwd(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let in_dim = *dims.last().unwrap();
    let batch: usize = dims.iter().rev().skip(1).product();
    let flat = x.reshape((batch, in_dim))?;
    let wt = weight.t()?;
    let out = flat.matmul(&wt)?;
    let out_dim = out.dim(1)?;
    let mut shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    shape.push(out_dim);
    out.reshape(shape)
}

// ---------------------------------------------------------------------------
// RmsNorm (simple inline, no dependency on candle_transformers)
// ---------------------------------------------------------------------------

struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(candle_core::DType::F32)?;
        let variance = (&x * &x)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(dtype)?.broadcast_mul(&self.weight)
    }
}

// ---------------------------------------------------------------------------
// LayerScale
// ---------------------------------------------------------------------------

struct LayerScale {
    scale: Tensor,
}

impl LayerScale {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let scale = vb.get(dim, "scale")?;
        Ok(Self { scale })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)
    }
}

// ---------------------------------------------------------------------------
// Rotary embeddings (standard, not MRoPE)
// ---------------------------------------------------------------------------

fn precompute_rope(
    max_len: usize,
    head_dim: usize,
    theta: f64,
    device: &candle_core::Device,
) -> Result<(Tensor, Tensor)> {
    let half = head_dim / 2;
    let inv_freq: Vec<f32> = (0..half)
        .map(|i| 1.0 / (theta as f32).powf(i as f32 * 2.0 / head_dim as f32))
        .collect();
    let inv_freq = Tensor::new(inv_freq, device)?; // [half]
    let positions: Vec<f32> = (0..max_len).map(|p| p as f32).collect();
    let positions = Tensor::new(positions, device)?.unsqueeze(1)?; // [max_len, 1]
    let freqs = positions.broadcast_mul(&inv_freq.unsqueeze(0)?)?; // [max_len, half]
    let cos = freqs.cos()?;
    let sin = freqs.sin()?;
    Ok((cos, sin))
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor, seq_offset: usize) -> Result<Tensor> {
    // x: [B, num_heads, T, head_dim]
    let seq_len = x.dim(2)?;
    let head_dim = x.dim(3)?;
    let half = head_dim / 2;

    let cos = cos.narrow(0, seq_offset, seq_len)?; // [T, half]
    let sin = sin.narrow(0, seq_offset, seq_len)?;

    // Split x into first half and second half
    let x1 = x.narrow(3, 0, half)?;
    let x2 = x.narrow(3, half, half)?;

    // Reshape cos/sin for broadcasting: [1, 1, T, half]
    let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?.unsqueeze(0)?;

    // RoPE: [x1*cos - x2*sin, x1*sin + x2*cos]
    let r1 = (x1.broadcast_mul(&cos)? - x2.broadcast_mul(&sin)?)?;
    let r2 = (x1.broadcast_mul(&sin)? + x2.broadcast_mul(&cos)?)?;
    Tensor::cat(&[r1, r2], 3)
}

// ---------------------------------------------------------------------------
// Attention (MHA, no bias, with RoPE + layer_scale)
// ---------------------------------------------------------------------------

struct Attention {
    q_proj: Tensor,
    k_proj: Tensor,
    v_proj: Tensor,
    o_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl Attention {
    fn new(hidden: usize, num_heads: usize, head_dim: usize, vb: VarBuilder) -> Result<Self> {
        let qkv_dim = num_heads * head_dim;
        let q_proj = vb.get((qkv_dim, hidden), "q_proj.weight")?;
        let k_proj = vb.get((qkv_dim, hidden), "k_proj.weight")?;
        let v_proj = vb.get((qkv_dim, hidden), "v_proj.weight")?;
        let o_proj = vb.get((hidden, qkv_dim), "o_proj.weight")?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
        })
    }

    #[allow(clippy::many_single_char_names)]
    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, seq_offset: usize) -> Result<Tensor> {
        let (b, t, _h) = x.dims3()?;

        let q = linear_fwd(x, &self.q_proj)?;
        let k = linear_fwd(x, &self.k_proj)?;
        let v = linear_fwd(x, &self.v_proj)?;

        // Reshape to [B, num_heads, T, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Apply RoPE (cat inside produces contiguous Q, K)
        let q = apply_rope(&q, cos, sin, seq_offset)?;
        let k = apply_rope(&k, cos, sin, seq_offset)?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = (q.matmul(&k_t)? / scale)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // [B, num_heads, T, head_dim] → [B, T, num_heads * head_dim]
        let out = out.transpose(1, 2)?.reshape((b, t, ()))?;
        linear_fwd(&out, &self.o_proj)
    }
}

// ---------------------------------------------------------------------------
// SwiGLU MLP
// ---------------------------------------------------------------------------

#[allow(clippy::struct_field_names)]
struct Mlp {
    gate_proj: Tensor,
    up_proj: Tensor,
    down_proj: Tensor,
}

impl Mlp {
    fn new(hidden: usize, intermediate: usize, vb: VarBuilder) -> Result<Self> {
        let gate_proj = vb.get((intermediate, hidden), "gate_proj.weight")?;
        let up_proj = vb.get((intermediate, hidden), "up_proj.weight")?;
        let down_proj = vb.get((hidden, intermediate), "down_proj.weight")?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = linear_fwd(x, &self.gate_proj)?.silu()?;
        let up = linear_fwd(x, &self.up_proj)?;
        let hidden = (gate * up)?;
        linear_fwd(&hidden, &self.down_proj)
    }
}

// ---------------------------------------------------------------------------
// Transformer layer
// ---------------------------------------------------------------------------

struct TransformerLayer {
    input_layernorm: RmsNorm,
    attn: Attention,
    attn_layer_scale: LayerScale,
    post_attention_layernorm: RmsNorm,
    mlp: Mlp,
    mlp_layer_scale: LayerScale,
}

impl TransformerLayer {
    fn new(
        hidden: usize,
        num_heads: usize,
        head_dim: usize,
        intermediate: usize,
        eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            input_layernorm: RmsNorm::new(hidden, eps, vb.pp("input_layernorm"))?,
            attn: Attention::new(hidden, num_heads, head_dim, vb.pp("self_attn"))?,
            attn_layer_scale: LayerScale::new(hidden, vb.pp("self_attn_layer_scale"))?,
            post_attention_layernorm: RmsNorm::new(hidden, eps, vb.pp("post_attention_layernorm"))?,
            mlp: Mlp::new(hidden, intermediate, vb.pp("mlp"))?,
            mlp_layer_scale: LayerScale::new(hidden, vb.pp("mlp_layer_scale"))?,
        })
    }

    fn forward(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, seq_offset: usize) -> Result<Tensor> {
        // Pre-norm attention + layer scale + residual
        let residual = x;
        let h = self.input_layernorm.forward(x)?;
        let h = self.attn.forward(&h, cos, sin, seq_offset)?;
        let h = self.attn_layer_scale.forward(&h)?;
        let x = (residual + h)?;

        // Pre-norm MLP + layer scale + residual
        let residual = &x;
        let h = self.post_attention_layernorm.forward(&x)?;
        let h = self.mlp.forward(&h)?;
        let h = self.mlp_layer_scale.forward(&h)?;
        residual + h
    }
}

// ---------------------------------------------------------------------------
// Linear with bias
// ---------------------------------------------------------------------------

struct LinearBias {
    weight: Tensor,
    bias: Tensor,
}

impl LinearBias {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = vb.get(out_dim, "bias")?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = linear_fwd(x, &self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------------------------
// PreTransformer (full block)
// ---------------------------------------------------------------------------

/// Pre-transformer: input_proj → 8 transformer layers → norm → output_proj.
///
/// Input:  `[B, 1024, T]` (from pre_conv, channel-first)
/// Output: `[B, 1024, T]` (back to channel-first for upsampling)
pub(crate) struct PreTransformer {
    input_proj: LinearBias,
    layers: Vec<TransformerLayer>,
    norm: RmsNorm,
    output_proj: LinearBias,
    cos: Tensor,
    sin: Tensor,
}

impl PreTransformer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        hidden_size: usize,       // 512
        num_layers: usize,        // 8
        num_heads: usize,         // 16
        head_dim: usize,          // 64
        intermediate_size: usize, // 1024
        latent_dim: usize,        // 1024
        rms_norm_eps: f64,
        rope_theta: f64,
        max_len: usize,
        device: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let input_proj = LinearBias::new(latent_dim, hidden_size, vb.pp("input_proj"))?;

        let mut layers = Vec::with_capacity(num_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..num_layers {
            layers.push(TransformerLayer::new(
                hidden_size,
                num_heads,
                head_dim,
                intermediate_size,
                rms_norm_eps,
                vb_layers.pp(i),
            )?);
        }

        let norm = RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("norm"))?;
        let output_proj = LinearBias::new(hidden_size, latent_dim, vb.pp("output_proj"))?;

        let (cos, sin) = precompute_rope(max_len, head_dim, rope_theta, device)?;

        Ok(Self {
            input_proj,
            layers,
            norm,
            output_proj,
            cos,
            sin,
        })
    }

    /// Forward pass.
    ///
    /// `x`: `[B, latent_dim, T]` (channel-first from conv layers).
    /// Returns: `[B, latent_dim, T]`.
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Conv format [B, C, T] → sequence format [B, T, C]
        let x = x.transpose(1, 2)?;

        // Project down: [B, T, 1024] → [B, T, 512]
        let mut h = self.input_proj.forward(&x)?;

        // Transformer layers
        for layer in &self.layers {
            h = layer.forward(&h, &self.cos, &self.sin, 0)?;
        }

        // Final norm
        h = self.norm.forward(&h)?;

        // Project up: [B, T, 512] → [B, T, 1024]
        h = self.output_proj.forward(&h)?;

        // Back to [B, C, T]
        h.transpose(1, 2)
    }
}
