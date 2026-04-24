//! CLAP text encoder (RoBERTa-based).
//!
//! Implements the text encoder for CLAP's zero-shot classification.
//! Weight prefix: `text_model.*` (not the standard `bert.*`).
//!
//! Architecture: RoBERTa = BERT with different training, same architecture.
//! - Embeddings: word + position + token_type + LayerNorm
//! - 12 transformer layers
//! - Pooler (dense + tanh on [CLS] token)

// ML code uses standard mathematical variable names (b, s, q, k, v)
// and usize→i64 casts are safe for realistic tensor dimensions.
#![allow(clippy::many_single_char_names, clippy::cast_possible_wrap)]

use candle_core::{DType, IndexOp, Tensor};
use candle_nn::VarBuilder;

use super::clap_config::ClapTextConfig;
use crate::{TaskError, TaskResult};

// ---------------------------------------------------------------------------
// Layer norm
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
            .map_err(|e| TaskError::ModelLoad(format!("ln weight: {e}")))?;
        let bias = vb
            .get(dim, "bias")
            .map_err(|e| TaskError::ModelLoad(format!("ln bias: {e}")))?;
        Ok(Self { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_c = x.broadcast_sub(&mean)?;
        let var = (&x_c * &x_c)?.mean_keepdim(candle_core::D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let norm = x_c.broadcast_div(&std)?;
        norm.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

struct RobertaEmbeddings {
    word_embeddings: Tensor,       // [vocab_size, hidden_size]
    position_embeddings: Tensor,   // [max_pos, hidden_size]
    token_type_embeddings: Tensor, // [type_vocab_size, hidden_size]
    layer_norm: LayerNorm,
    // RoBERTa padding_idx = 1, position offset = 2
    padding_idx: usize,
}

impl RobertaEmbeddings {
    fn load(vb: &VarBuilder, cfg: &ClapTextConfig) -> TaskResult<Self> {
        let vb_emb = vb.pp("embeddings");
        let word_embeddings = vb_emb
            .get((cfg.vocab_size, cfg.hidden_size), "word_embeddings.weight")
            .map_err(|e| TaskError::ModelLoad(format!("word_embeddings: {e}")))?;
        let position_embeddings = vb_emb
            .get(
                (cfg.max_position_embeddings, cfg.hidden_size),
                "position_embeddings.weight",
            )
            .map_err(|e| TaskError::ModelLoad(format!("position_embeddings: {e}")))?;
        let token_type_embeddings = vb_emb
            .get(
                (cfg.type_vocab_size, cfg.hidden_size),
                "token_type_embeddings.weight",
            )
            .map_err(|e| TaskError::ModelLoad(format!("token_type_embeddings: {e}")))?;
        let layer_norm =
            LayerNorm::load(&vb_emb.pp("LayerNorm"), cfg.hidden_size, cfg.layer_norm_eps)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            padding_idx: cfg.pad_token_id,
        })
    }

    /// Forward: input_ids `[B, S]` → embeddings `[B, S, H]`
    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        let device = input_ids.device();
        let (b, s) = input_ids.dims2()?;

        // Word embeddings
        let word_emb = self
            .word_embeddings
            .index_select(&input_ids.flatten_all()?, 0)?;
        let word_emb = word_emb.reshape((b, s, ()))?;

        // Position IDs: RoBERTa uses padding_idx + 1 as start
        // create_position_ids_from_input_ids: positions start after padding_idx
        let position_ids: Vec<i64> = (0..s as i64)
            .map(|i| i + self.padding_idx as i64 + 1)
            .collect();
        let position_ids = Tensor::from_vec(position_ids, (1, s), device)?;
        let pos_emb = self
            .position_embeddings
            .index_select(&position_ids.flatten_all()?, 0)?;
        let pos_emb = pos_emb.reshape((1, s, ()))?;

        // Token type IDs: all zeros for single segment
        let tt_ids = Tensor::zeros((1, s), DType::I64, device)?;
        let tt_emb = self
            .token_type_embeddings
            .index_select(&tt_ids.flatten_all()?, 0)?;
        let tt_emb = tt_emb.reshape((1, s, ()))?;

        let embeddings = (word_emb
            + pos_emb.broadcast_add(&Tensor::zeros((b, 1, 1), pos_emb.dtype(), device)?)?)?;
        let embeddings = embeddings.broadcast_add(&tt_emb)?;
        self.layer_norm.forward(&embeddings)
    }
}

// ---------------------------------------------------------------------------
// Self-Attention
// ---------------------------------------------------------------------------

struct BertSelfAttention {
    q_weight: Tensor,
    q_bias: Tensor,
    k_weight: Tensor,
    k_bias: Tensor,
    v_weight: Tensor,
    v_bias: Tensor,
    out_weight: Tensor,
    out_bias: Tensor,
    out_ln: LayerNorm,
    num_heads: usize,
    head_dim: usize,
}

impl BertSelfAttention {
    fn load(vb: &VarBuilder, cfg: &ClapTextConfig) -> TaskResult<Self> {
        let h = cfg.hidden_size;
        let vb_self = vb.pp("attention").pp("self");
        let q_weight = vb_self
            .get((h, h), "query.weight")
            .map_err(|e| TaskError::ModelLoad(format!("q.w: {e}")))?;
        let q_bias = vb_self
            .get(h, "query.bias")
            .map_err(|e| TaskError::ModelLoad(format!("q.b: {e}")))?;
        let k_weight = vb_self
            .get((h, h), "key.weight")
            .map_err(|e| TaskError::ModelLoad(format!("k.w: {e}")))?;
        let k_bias = vb_self
            .get(h, "key.bias")
            .map_err(|e| TaskError::ModelLoad(format!("k.b: {e}")))?;
        let v_weight = vb_self
            .get((h, h), "value.weight")
            .map_err(|e| TaskError::ModelLoad(format!("v.w: {e}")))?;
        let v_bias = vb_self
            .get(h, "value.bias")
            .map_err(|e| TaskError::ModelLoad(format!("v.b: {e}")))?;

        let vb_out = vb.pp("attention").pp("output");
        let out_weight = vb_out
            .get((h, h), "dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("out.w: {e}")))?;
        let out_bias = vb_out
            .get(h, "dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("out.b: {e}")))?;
        let out_ln = LayerNorm::load(&vb_out.pp("LayerNorm"), h, cfg.layer_norm_eps)?;

        Ok(Self {
            q_weight,
            q_bias,
            k_weight,
            k_bias,
            v_weight,
            v_bias,
            out_weight,
            out_bias,
            out_ln,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    /// Forward: `[B, S, H]` → `[B, S, H]` (with residual + LN)
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, s, _h) = x.dims3()?;
        let residual = x.clone();

        // Q, K, V projections
        let q = x
            .contiguous()?
            .broadcast_matmul(&self.q_weight.t()?.contiguous()?)?
            .broadcast_add(&self.q_bias)?;
        let k = x
            .contiguous()?
            .broadcast_matmul(&self.k_weight.t()?.contiguous()?)?
            .broadcast_add(&self.k_bias)?;
        let v = x
            .contiguous()?
            .broadcast_matmul(&self.v_weight.t()?.contiguous()?)?
            .broadcast_add(&self.v_bias)?;

        // Reshape to multi-head: [B, S, H] → [B, S, num_heads, head_dim] → [B, num_heads, S, head_dim]
        let q = q
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((b, s, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        // Attention: [B, heads, S, S]
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let attn = (q.contiguous()? * scale)?.matmul(&k.t()?.contiguous()?)?;
        let attn = candle_nn::ops::softmax_last_dim(&attn)?;

        // Weighted values: [B, heads, S, head_dim] → [B, S, H]
        let out = attn.matmul(&v.contiguous()?)?;
        let out = out.permute((0, 2, 1, 3))?.contiguous()?.reshape((
            b,
            s,
            self.num_heads * self.head_dim,
        ))?;

        // Output projection + residual + LN
        let out = out
            .broadcast_matmul(&self.out_weight.t()?.contiguous()?)?
            .broadcast_add(&self.out_bias)?;
        let out = (out + residual)?;
        self.out_ln.forward(&out)
    }
}

// ---------------------------------------------------------------------------
// Feed-Forward (Intermediate + Output)
// ---------------------------------------------------------------------------

struct BertFeedForward {
    fc1_weight: Tensor,
    fc1_bias: Tensor,
    fc2_weight: Tensor,
    fc2_bias: Tensor,
    ln: LayerNorm,
}

impl BertFeedForward {
    fn load(vb: &VarBuilder, cfg: &ClapTextConfig) -> TaskResult<Self> {
        let h = cfg.hidden_size;
        let mid = cfg.intermediate_size;
        let fc1_weight = vb
            .get((mid, h), "intermediate.dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("ff1.w: {e}")))?;
        let fc1_bias = vb
            .get(mid, "intermediate.dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("ff1.b: {e}")))?;
        let fc2_weight = vb
            .get((h, mid), "output.dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("ff2.w: {e}")))?;
        let fc2_bias = vb
            .get(h, "output.dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("ff2.b: {e}")))?;
        let ln = LayerNorm::load(&vb.pp("output").pp("LayerNorm"), h, cfg.layer_norm_eps)?;
        Ok(Self {
            fc1_weight,
            fc1_bias,
            fc2_weight,
            fc2_bias,
            ln,
        })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let residual = x.clone();
        let h = x
            .contiguous()?
            .broadcast_matmul(&self.fc1_weight.t()?.contiguous()?)?
            .broadcast_add(&self.fc1_bias)?;
        let h = h.gelu_erf()?;
        let h = h
            .contiguous()?
            .broadcast_matmul(&self.fc2_weight.t()?.contiguous()?)?
            .broadcast_add(&self.fc2_bias)?;
        let h = (h + residual)?;
        self.ln.forward(&h)
    }
}

// ---------------------------------------------------------------------------
// Transformer Layer
// ---------------------------------------------------------------------------

struct BertLayer {
    attention: BertSelfAttention,
    ff: BertFeedForward,
}

impl BertLayer {
    fn load(vb: &VarBuilder, cfg: &ClapTextConfig) -> TaskResult<Self> {
        let attention = BertSelfAttention::load(vb, cfg)?;
        let ff = BertFeedForward::load(vb, cfg)?;
        Ok(Self { attention, ff })
    }

    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.attention.forward(x)?;
        self.ff.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Pooler
// ---------------------------------------------------------------------------

struct BertPooler {
    dense_weight: Tensor,
    dense_bias: Tensor,
}

impl BertPooler {
    fn load(vb: &VarBuilder, hidden_size: usize) -> TaskResult<Self> {
        let dense_weight = vb
            .get((hidden_size, hidden_size), "dense.weight")
            .map_err(|e| TaskError::ModelLoad(format!("pooler.w: {e}")))?;
        let dense_bias = vb
            .get(hidden_size, "dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("pooler.b: {e}")))?;
        Ok(Self {
            dense_weight,
            dense_bias,
        })
    }

    /// Pool [CLS] token: `[B, S, H]` → `[B, H]`
    fn forward(&self, hidden: &Tensor) -> candle_core::Result<Tensor> {
        let cls = hidden.i((.., 0, ..))?; // [B, H]
        let out = cls
            .contiguous()?
            .broadcast_matmul(&self.dense_weight.t()?.contiguous()?)?
            .broadcast_add(&self.dense_bias)?;
        out.tanh()
    }
}

// ---------------------------------------------------------------------------
// Full CLAP Text Encoder
// ---------------------------------------------------------------------------

/// RoBERTa-based text encoder for CLAP.
///
/// Weight prefix: `text_model.*`
pub struct ClapTextEncoder {
    embeddings: RobertaEmbeddings,
    layers: Vec<BertLayer>,
    pooler: BertPooler,
}

impl ClapTextEncoder {
    pub fn load(vb: &VarBuilder, cfg: &ClapTextConfig) -> TaskResult<Self> {
        let vb_text = vb.pp("text_model");

        let embeddings = RobertaEmbeddings::load(&vb_text, cfg)?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = BertLayer::load(&vb_text.pp("encoder").pp("layer").pp(i.to_string()), cfg)?;
            layers.push(layer);
        }

        let pooler = BertPooler::load(&vb_text.pp("pooler"), cfg.hidden_size)?;

        Ok(Self {
            embeddings,
            layers,
            pooler,
        })
    }

    /// Forward: input_ids `[B, S]` → pooled output `[B, H]`
    pub fn forward(&self, input_ids: &Tensor) -> TaskResult<Tensor> {
        let mut x = self
            .embeddings
            .forward(input_ids)
            .map_err(|e| TaskError::Inference(format!("text embeddings: {e}")))?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer
                .forward(&x)
                .map_err(|e| TaskError::Inference(format!("text layer {i}: {e}")))?;
        }

        self.pooler
            .forward(&x)
            .map_err(|e| TaskError::Inference(format!("text pooler: {e}")))
    }
}
