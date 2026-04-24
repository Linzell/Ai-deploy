//! Shared layer building blocks for Qwen3 TTS.
//!
//! Both the talker and code predictor use the same fundamental layer types:
//! - `Attention` — GQA with QK normalization (RmsNorm on Q and K per head)
//! - `Mlp` — SwiGLU (gate_proj + up_proj → SiLU → down_proj)
//! - `DecoderLayer` — Pre-norm transformer layer (norm → attn → residual → norm → mlp → residual)
//! - `ResizeMlp` — Simple projection MLP (linear → SiLU → linear) used for text_projection

use candle_core::{Module, Result, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};
use candle_transformers::utils::repeat_kv;

// ---------------------------------------------------------------------------
// Attention (GQA + QK normalization)
// ---------------------------------------------------------------------------

/// Grouped-query attention with per-head QK normalization.
///
/// Key differences from upstream Qwen2:
/// - No bias on Q/K/V/O projections
/// - RmsNorm applied to Q and K after projection, before RoPE
///
/// RoPE is NOT applied here — the caller applies it (MRoPE for talker,
/// standard RoPE for code predictor) and passes already-rotated Q/K.
pub struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_kv_groups: usize,
    kv_cache: Option<(Tensor, Tensor)>,
}

impl Attention {
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        let q_proj = linear_no_bias(hidden_size, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = linear_no_bias(hidden_size, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = linear_no_bias(num_heads * head_dim, hidden_size, vb.pp("o_proj"))?;
        let q_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = RmsNorm::new(head_dim, rms_norm_eps, vb.pp("k_norm"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            num_kv_groups: num_heads / num_kv_heads,
            kv_cache: None,
        })
    }

    /// Project Q, K, V and apply QK normalization.
    ///
    /// Returns `(q, k, v)` where:
    /// - `q` shape: `[batch, num_heads, seq_len, head_dim]`
    /// - `k` shape: `[batch, num_kv_heads, seq_len, head_dim]`
    /// - `v` shape: `[batch, num_kv_heads, seq_len, head_dim]`
    ///
    /// Q and K have been normalized but NOT rotated (caller applies RoPE).
    pub fn project_qkv(&self, x: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (b, seq_len, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // Reshape to [batch, seq, num_heads, head_dim]
        let q = q.reshape((b, seq_len, self.num_heads, self.head_dim))?;
        let k = k.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;
        let v = v.reshape((b, seq_len, self.num_kv_heads, self.head_dim))?;

        // Apply QK normalization (per-head RmsNorm on the head_dim dimension)
        let q = self.q_norm.forward(&q)?;
        let k = self.k_norm.forward(&k)?;

        // Transpose to [batch, heads, seq, head_dim]
        let q = q.transpose(1, 2)?.contiguous()?;
        let k = k.transpose(1, 2)?.contiguous()?;
        let v = v.transpose(1, 2)?.contiguous()?;

        Ok((q, k, v))
    }

    /// Compute attention given already-rotated Q, K and unmodified V.
    ///
    /// `q` shape: `[batch, num_heads, seq_len, head_dim]` (already rotated)
    /// `k` shape: `[batch, num_kv_heads, seq_len, head_dim]` (already rotated)
    /// `v` shape: `[batch, num_kv_heads, seq_len, head_dim]`
    pub fn forward_after_rope(&mut self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (b, _h, seq_len, _d) = q.dims4()?;

        // Append to KV cache
        let (k, v) = match &self.kv_cache {
            Some((prev_k, prev_v)) => {
                let k = Tensor::cat(&[prev_k, k], 2)?;
                let v = Tensor::cat(&[prev_v, v], 2)?;
                (k, v)
            }
            None => (k.clone(), v.clone()),
        };
        self.kv_cache = Some((k.clone(), v.clone()));

        // Expand KV heads for GQA
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(2, 3)?.contiguous()?;
        let attn = (q.matmul(&k_t)? * scale)?;

        // Causal mask: only needed when seq_len > 1 (prefill)
        let attn = if seq_len > 1 {
            let kv_len = attn.dim(3)?;
            let mask = create_causal_mask(seq_len, kv_len, q.device())?;
            attn.broadcast_add(&mask)?
        } else {
            attn
        };

        let attn = candle_nn::ops::softmax_last_dim(&attn)?;
        let out = attn.matmul(&v)?;

        // Reshape back: [batch, num_heads, seq, head_dim] -> [batch, seq, hidden]
        let out = out.transpose(1, 2)?.contiguous()?.reshape((
            b,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&out)
    }

    pub fn clear_kv_cache(&mut self) {
        self.kv_cache = None;
    }
}

/// Create a causal attention mask.
///
/// Returns a tensor of shape `[1, 1, seq_len, kv_len]` with 0.0 for
/// allowed positions and `-inf` for masked positions.
fn create_causal_mask(
    seq_len: usize,
    kv_len: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let offset = kv_len - seq_len;
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|i| {
            (0..kv_len).map(move |j| {
                if j <= i + offset {
                    0.0
                } else {
                    f32::NEG_INFINITY
                }
            })
        })
        .collect();
    Tensor::new(mask, device)?.reshape((1, 1, seq_len, kv_len))
}

// ---------------------------------------------------------------------------
// MLP (SwiGLU)
// ---------------------------------------------------------------------------

/// SwiGLU MLP: gate_proj + up_proj → SiLU gate → down_proj.
pub struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Mlp {
    pub fn new(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

// ---------------------------------------------------------------------------
// DecoderLayer
// ---------------------------------------------------------------------------

/// Pre-norm transformer decoder layer.
///
/// Structure: input_layernorm → self_attn → residual → post_attn_layernorm → mlp → residual.
pub struct DecoderLayer {
    pub self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        vb: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            self_attn: Attention::new(
                hidden_size,
                num_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                vb.pp("self_attn"),
            )?,
            mlp: Mlp::new(hidden_size, intermediate_size, vb.pp("mlp"))?,
            input_layernorm: RmsNorm::new(hidden_size, rms_norm_eps, vb.pp("input_layernorm"))?,
            post_attention_layernorm: RmsNorm::new(
                hidden_size,
                rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    /// Forward pass. Caller must supply already-rotated Q and K.
    ///
    /// The typical call pattern:
    /// 1. Caller computes `(q, k, v) = layer.self_attn.project_qkv(&normed_x)`
    /// 2. Caller applies RoPE to q, k
    /// 3. Caller calls `layer.forward_with_rotated_qkv(x, q, k, v)`
    ///
    /// But for convenience we also provide `forward_with_rope_fn` which
    /// takes a closure for the rotation step.
    pub fn forward_with_rope_fn(
        &mut self,
        x: &Tensor,
        apply_rope: &mut dyn FnMut(&Tensor, &Tensor) -> Result<(Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // Pre-norm
        let residual = x;
        let normed = self.input_layernorm.forward(x)?;

        // Project Q, K, V (with QK norm)
        let (q, k, v) = self.self_attn.project_qkv(&normed)?;

        // Apply rotation (MRoPE or standard RoPE)
        let (q, k) = apply_rope(&q, &k)?;

        // Self-attention
        let attn_out = self.self_attn.forward_after_rope(&q, &k, &v)?;
        let x = (residual + attn_out)?;

        // Post-norm + MLP
        let residual = &x;
        let normed = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        residual + mlp_out
    }

    pub fn clear_kv_cache(&mut self) {
        self.self_attn.clear_kv_cache();
    }
}

// ---------------------------------------------------------------------------
// ResizeMLP (text_projection)
// ---------------------------------------------------------------------------

/// Simple 2-layer MLP with SiLU activation for projecting between different
/// hidden dimensions (e.g., text encoder hidden → talker hidden).
///
/// Weight names: `linear_fc1.{weight,bias}`, `linear_fc2.{weight,bias}`.
pub struct ResizeMlp {
    fc1: Linear,
    fc2: Linear,
}

impl ResizeMlp {
    pub fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            fc1: linear(in_dim, out_dim, vb.pp("linear_fc1"))?,
            fc2: linear(out_dim, out_dim, vb.pp("linear_fc2"))?,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?.silu()?;
        self.fc2.forward(&x)
    }
}
