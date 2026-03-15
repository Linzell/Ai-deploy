//! Transformer encoder for Wav2Vec2 / HuBERT.
//!
//! Contains:
//! - `PositionalConvEmbedding` — grouped 1D conv with weight normalization
//! - `EncoderLayer` — self-attention + FFN with pre-norm or post-norm
//! - `Wav2Vec2Encoder` — full transformer encoder stack

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::config::Wav2Vec2Config;
use crate::{TaskError, TaskResult};

// ---------------------------------------------------------------------------
// LayerNorm helper (manual, operates on last dim)
// ---------------------------------------------------------------------------

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn load(vb: &VarBuilder, dim: usize, eps: f64) -> TaskResult<Self> {
        let weight = vb
            .get(dim, "weight")
            .map_err(|e| TaskError::ModelLoad(format!("LayerNorm weight: {e}")))?;
        let bias = vb
            .get(dim, "bias")
            .map_err(|e| TaskError::ModelLoad(format!("LayerNorm bias: {e}")))?;
        Ok(Self { weight, bias, eps })
    }

    /// Apply LayerNorm over the last dimension.
    /// Input: [..., D], output: [..., D]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let rank = xs.rank();
        let d = xs.dim(rank - 1)? as f64;
        let mean = xs.sum_keepdim(rank - 1)?.affine(1.0 / d, 0.0)?;
        let diff = xs.broadcast_sub(&mean)?;
        let var = (&diff * &diff)?
            .sum_keepdim(rank - 1)?
            .affine(1.0 / d, 0.0)?;
        let normed = diff.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normed
            .broadcast_mul(&self.weight.unsqueeze(0)?.unsqueeze(0)?)?
            .broadcast_add(&self.bias.unsqueeze(0)?.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Positional convolutional embedding (with weight normalization)
// ---------------------------------------------------------------------------

/// Positional convolution embedding.
///
/// Uses grouped 1D convolution with weight normalization (stored as
/// `parametrizations.weight.original0` (g) and `original1` (v) in HF checkpoints).
pub struct PositionalConvEmbedding {
    conv_weight: Tensor, // materialized weight after weight-norm
    conv_bias: Tensor,
    padding: usize,
    groups: usize,
}

impl PositionalConvEmbedding {
    pub fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let prefix = cfg.backbone_prefix();
        let vb_conv = vb.pp(format!("{prefix}.encoder.pos_conv_embed.conv"));

        let kernel = cfg.num_conv_pos_embeddings;
        let groups = cfg.num_conv_pos_embedding_groups;
        let hidden = cfg.hidden_size;

        // Weight normalization: w = g * (v / ||v||)
        // g: [1, 1, kernel] or broadcastable
        // v: [hidden, hidden/groups, kernel]
        let g = vb_conv
            .get_unchecked("parametrizations.weight.original0")
            .map_err(|e| TaskError::ModelLoad(format!("pos_conv g: {e}")))?;
        let v = vb_conv
            .get_unchecked("parametrizations.weight.original1")
            .map_err(|e| TaskError::ModelLoad(format!("pos_conv v: {e}")))?;

        // Compute ||v|| over (in_ch, kernel) dims per output channel
        // v shape: [out_ch, in_ch_per_group, kernel]
        let v_norm = v.sqr()?.sum_keepdim(2)?.sum_keepdim(1)?.sqrt()?;
        let v_normalized = v.broadcast_div(&(v_norm + 1e-12)?)?;
        let conv_weight = v_normalized.broadcast_mul(&g)?;

        let conv_bias = vb_conv
            .get(hidden, "bias")
            .map_err(|e| TaskError::ModelLoad(format!("pos_conv bias: {e}")))?;

        let padding = kernel / 2;

        Ok(Self {
            conv_weight,
            conv_bias,
            padding,
            groups,
        })
    }

    /// Forward: `[B, T, C]` → `[B, T, C]`
    pub fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // Transpose to [B, C, T] for conv1d
        let x = xs.transpose(1, 2)?.contiguous()?;
        let conv_weight = self.conv_weight.contiguous()?;

        let x = x.conv1d(&conv_weight, self.padding, 1, 1, self.groups)?;
        // Add bias: [hidden] → [1, hidden, 1]
        let x = x.broadcast_add(&self.conv_bias.unsqueeze(0)?.unsqueeze(2)?)?;
        // GELU activation
        let x = x.gelu_erf()?;

        // Transpose back to [B, T, C] — may have different T due to conv, truncate
        let x = x.transpose(1, 2)?;

        // Truncate/pad to match input length
        let out_t = x.dim(1)?;
        let in_t = xs.dim(1)?;
        if out_t > in_t {
            x.narrow(1, 0, in_t)
        } else {
            Ok(x)
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-head self-attention
// ---------------------------------------------------------------------------

struct SelfAttention {
    q_proj_weight: Tensor,
    q_proj_bias: Tensor,
    k_proj_weight: Tensor,
    k_proj_bias: Tensor,
    v_proj_weight: Tensor,
    v_proj_bias: Tensor,
    out_proj_weight: Tensor,
    out_proj_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl SelfAttention {
    fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let h = cfg.hidden_size;
        let load = |name: &str| -> TaskResult<(Tensor, Tensor)> {
            let w = vb
                .get((h, h), &format!("{name}.weight"))
                .map_err(|e| TaskError::ModelLoad(format!("attention {name}.weight: {e}")))?;
            let b = vb
                .get(h, &format!("{name}.bias"))
                .map_err(|e| TaskError::ModelLoad(format!("attention {name}.bias: {e}")))?;
            Ok((w, b))
        };

        let (qw, qb) = load("q_proj")?;
        let (kw, kb) = load("k_proj")?;
        let (vw, vb_t) = load("v_proj")?;
        let (ow, ob) = load("out_proj")?;

        Ok(Self {
            q_proj_weight: qw,
            q_proj_bias: qb,
            k_proj_weight: kw,
            k_proj_bias: kb,
            v_proj_weight: vw,
            v_proj_bias: vb_t,
            out_proj_weight: ow,
            out_proj_bias: ob,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Linear projection: [B, T, H] @ W^T + b
    fn linear(xs: &Tensor, weight: &Tensor, bias: &Tensor) -> candle_core::Result<Tensor> {
        let w_t = weight.t()?.contiguous()?;
        let xs_c = xs.contiguous()?;
        xs_c.broadcast_matmul(&w_t)?
            .broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)
    }

    /// Forward: [B, T, H] → [B, T, H]
    #[allow(clippy::many_single_char_names)]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let (b, t, _h) = xs.dims3()?;

        let q = Self::linear(xs, &self.q_proj_weight, &self.q_proj_bias)?;
        let k = Self::linear(xs, &self.k_proj_weight, &self.k_proj_bias)?;
        let v = Self::linear(xs, &self.v_proj_weight, &self.v_proj_bias)?;

        // Reshape to [B, num_heads, T, head_dim]
        let q = q
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((b, t, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let attn_weights = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .affine(1.0 / scale, 0.0)?;
        // Softmax over last dim
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;

        let attn_out = attn_weights.matmul(&v)?;
        // [B, num_heads, T, head_dim] → [B, T, H]
        let attn_out = attn_out.transpose(1, 2)?.contiguous()?.reshape((
            b,
            t,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection
        Self::linear(&attn_out, &self.out_proj_weight, &self.out_proj_bias)
    }
}

// ---------------------------------------------------------------------------
// Feed-forward network
// ---------------------------------------------------------------------------

struct FeedForward {
    intermediate_weight: Tensor,
    intermediate_bias: Tensor,
    output_weight: Tensor,
    output_bias: Tensor,
}

impl FeedForward {
    fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let h = cfg.hidden_size;
        let inter = cfg.intermediate_size;

        let iw = vb
            .get((inter, h), "intermediate_dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("ffn intermediate.weight: {e}")))?;
        let ib = vb
            .get(inter, "intermediate_dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("ffn intermediate.bias: {e}")))?;
        let ow = vb
            .get((h, inter), "output_dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("ffn output.weight: {e}")))?;
        let ob = vb
            .get(h, "output_dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("ffn output.bias: {e}")))?;

        Ok(Self {
            intermediate_weight: iw,
            intermediate_bias: ib,
            output_weight: ow,
            output_bias: ob,
        })
    }

    /// Forward: [B, T, H] → [B, T, H]
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let xs_c = xs.contiguous()?;
        let x = xs_c.broadcast_matmul(&self.intermediate_weight.t()?.contiguous()?)?;
        let x = x.broadcast_add(&self.intermediate_bias.unsqueeze(0)?.unsqueeze(0)?)?;
        let x = x.gelu_erf()?;
        let x = x.contiguous()?;
        let x = x.broadcast_matmul(&self.output_weight.t()?.contiguous()?)?;
        x.broadcast_add(&self.output_bias.unsqueeze(0)?.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Single encoder layer
// ---------------------------------------------------------------------------

struct EncoderLayer {
    attention: SelfAttention,
    feed_forward: FeedForward,
    layer_norm: LayerNorm,
    final_layer_norm: LayerNorm,
    do_stable_layer_norm: bool,
}

impl EncoderLayer {
    fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let attention = SelfAttention::load(&vb.pp("attention"), cfg)?;
        let feed_forward = FeedForward::load(&vb.pp("feed_forward"), cfg)?;
        let layer_norm =
            LayerNorm::load(&vb.pp("layer_norm"), cfg.hidden_size, cfg.layer_norm_eps)?;
        let final_layer_norm = LayerNorm::load(
            &vb.pp("final_layer_norm"),
            cfg.hidden_size,
            cfg.layer_norm_eps,
        )?;

        Ok(Self {
            attention,
            feed_forward,
            layer_norm,
            final_layer_norm,
            do_stable_layer_norm: cfg.do_stable_layer_norm,
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        if self.do_stable_layer_norm {
            // Pre-norm: norm → attn → residual → norm → ffn → residual
            let residual = xs;
            let x = self.layer_norm.forward(xs)?;
            let x = self.attention.forward(&x)?;
            let x = (residual + &x)?;

            let residual = &x;
            let x = self.final_layer_norm.forward(&x)?;
            let x = self.feed_forward.forward(&x)?;
            residual + &x
        } else {
            // Post-norm: attn → residual → norm → ffn → residual → norm
            let residual = xs;
            let x = self.attention.forward(xs)?;
            let x = (residual + &x)?;
            let x = self.layer_norm.forward(&x)?;

            let residual = &x;
            let x = self.feed_forward.forward(&x)?;
            let x = (residual + &x)?;
            self.final_layer_norm.forward(&x)
        }
    }
}

// ---------------------------------------------------------------------------
// Full encoder
// ---------------------------------------------------------------------------

/// The complete Wav2Vec2 transformer encoder.
pub struct Wav2Vec2Encoder {
    pos_conv: PositionalConvEmbedding,
    layer_norm: LayerNorm,
    layers: Vec<EncoderLayer>,
}

impl Wav2Vec2Encoder {
    pub fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let prefix = cfg.backbone_prefix();
        let vb_enc = vb.pp(format!("{prefix}.encoder"));

        let pos_conv = PositionalConvEmbedding::load(vb, cfg)?;
        let layer_norm = LayerNorm::load(
            &vb_enc.pp("layer_norm"),
            cfg.hidden_size,
            cfg.layer_norm_eps,
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = EncoderLayer::load(&vb_enc.pp(format!("layers.{i}")), cfg)?;
            layers.push(layer);
        }

        Ok(Self {
            pos_conv,
            layer_norm,
            layers,
        })
    }

    /// Forward: hidden_states `[B, T, H]` → encoded `[B, T, H]`
    pub fn forward(&self, hidden_states: &Tensor) -> TaskResult<Tensor> {
        // Add positional conv embeddings
        let pos_emb = self
            .pos_conv
            .forward(hidden_states)
            .map_err(|e| TaskError::Inference(format!("pos_conv: {e}")))?;
        let x = (hidden_states + &pos_emb)
            .map_err(|e| TaskError::Inference(format!("pos add: {e}")))?;

        // Outer layer norm
        let mut x = self
            .layer_norm
            .forward(&x)
            .map_err(|e| TaskError::Inference(format!("encoder layer_norm: {e}")))?;

        // Transformer layers
        for (i, layer) in self.layers.iter().enumerate() {
            x = layer
                .forward(&x)
                .map_err(|e| TaskError::Inference(format!("encoder layer {i}: {e}")))?;
        }

        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_head_dim() {
        let cfg = Wav2Vec2Config {
            hidden_size: 768,
            num_attention_heads: 12,
            ..default_test_config()
        };
        assert_eq!(cfg.head_dim(), 64);
    }

    fn default_test_config() -> Wav2Vec2Config {
        Wav2Vec2Config {
            model_type: Some("wav2vec2".into()),
            conv_dim: vec![512; 7],
            conv_kernel: vec![10, 3, 3, 3, 3, 2, 2],
            conv_stride: vec![5, 2, 2, 2, 2, 2, 2],
            conv_bias: false,
            feat_extract_norm: "group".into(),
            num_conv_pos_embeddings: 128,
            num_conv_pos_embedding_groups: 16,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".into(),
            layer_norm_eps: 1e-5,
            do_stable_layer_norm: false,
            classifier_proj_size: 256,
            num_labels: 2,
        }
    }
}
