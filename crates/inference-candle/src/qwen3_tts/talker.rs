//! Talker model — the main transformer for Qwen3 TTS.
//!
//! Responsible for generating the first codebook (CB0) tokens from text input.
//!
//! Architecture:
//! - `text_embedding` [vocab_size, hidden_size] — for text token inputs
//! - `codec_embedding` [codec_vocab_size, hidden_size] — for codec token inputs
//! - `text_projection` (ResizeMlp) — projects text encoder hidden → talker hidden
//! - 28 transformer layers with MRoPE, GQA, QK normalization
//! - `norm` — final RmsNorm
//! - `codec_head` [codec_vocab_size, hidden_size] — predicts CB0 token logits
//!
//! Weight prefix in safetensors: `talker.model.*`, `talker.text_projection.*`,
//! `talker.codec_head.*`.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::with_tracing::RmsNorm;

use super::config::TalkerConfig;
use super::layers::{DecoderLayer, ResizeMlp};
use super::rope::{
    apply_rotary_emb, make_mrope_position_ids, precompute_mrope_cos_sin, RotaryEmbedding,
};

// ---------------------------------------------------------------------------
// TalkerModel (the bare transformer without heads)
// ---------------------------------------------------------------------------

/// The core transformer stack with dual embeddings and MRoPE.
pub struct TalkerModel {
    text_embedding: Embedding,
    codec_embedding: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    mrope_section: Vec<usize>,
}

impl TalkerModel {
    pub fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let text_embedding = candle_nn::embedding(
            cfg.text_vocab_size,
            cfg.hidden_size,
            vb.pp("text_embedding"),
        )?;
        let codec_embedding = candle_nn::embedding(
            cfg.codec_vocab_size(),
            cfg.hidden_size,
            vb.pp("codec_embedding"),
        )?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("layers");
        for i in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                cfg.hidden_size,
                cfg.intermediate_size,
                cfg.num_attention_heads,
                cfg.num_key_value_heads,
                cfg.head_dim,
                cfg.rms_norm_eps,
                vb_layers.pp(i),
            )?);
        }

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;

        let rotary = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.rope_theta,
            cfg.max_position_embeddings.min(8192), // start small, grows on demand
            vb.pp("text_embedding").device(),      // use same device
        )?;

        Ok(Self {
            text_embedding,
            codec_embedding,
            layers,
            norm,
            rotary,
            mrope_section: cfg.mrope_section(),
        })
    }

    /// Forward pass through the transformer.
    ///
    /// `input_embeds`: pre-computed embeddings, shape `[batch, seq_len, hidden_size]`.
    /// `seqlen_offset`: position offset for KV cache (0 on first call, grows each step).
    ///
    /// Returns hidden states `[batch, seq_len, hidden_size]`.
    pub fn forward(&mut self, input_embeds: &Tensor, seqlen_offset: usize) -> Result<Tensor> {
        let (_b, seq_len, _h) = input_embeds.dims3()?;
        let device = input_embeds.device();

        // Build MRoPE position IDs
        let position_ids = make_mrope_position_ids(seq_len, seqlen_offset, device)?;

        // Pre-compute MRoPE cos/sin sections so we don't need &mut self.rotary inside the loop
        let mrope_cos_sin = precompute_mrope_cos_sin(
            &mut self.rotary,
            &position_ids,
            &self.mrope_section,
            input_embeds.dtype(),
        )?;

        let mut x = input_embeds.clone();

        for layer in &mut self.layers {
            let cos = &mrope_cos_sin.0;
            let sin = &mrope_cos_sin.1;
            x = layer.forward_with_rope_fn(&x, &mut |q, k| {
                let q_rot = apply_rotary_emb(q, cos, sin)?;
                let k_rot = apply_rotary_emb(k, cos, sin)?;
                Ok((q_rot, k_rot))
            })?;
        }

        self.norm.forward(&x)
    }

    /// Embed text token IDs.
    pub fn embed_text(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.text_embedding.forward(token_ids)
    }

    /// Embed codec token IDs.
    pub fn embed_codec(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.codec_embedding.forward(token_ids)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

// ---------------------------------------------------------------------------
// Talker (model + text_projection + codec_head)
// ---------------------------------------------------------------------------

/// Full talker: transformer + text_projection + codec_head.
///
/// Weight prefix: `talker.*`
/// - `talker.model.*` — transformer layers
/// - `talker.text_projection.*` — ResizeMlp
/// - `talker.codec_head.weight` — Linear (no bias)
pub struct Talker {
    pub model: TalkerModel,
    pub text_projection: ResizeMlp,
    /// codec_head: [codec_vocab_size, hidden_size] — predicts CB0 logits.
    pub codec_head: candle_transformers::models::with_tracing::Linear,
    pub hidden_size: usize,
}

impl Talker {
    pub fn new(cfg: &TalkerConfig, vb: VarBuilder) -> Result<Self> {
        let model = TalkerModel::new(cfg, vb.pp("model"))?;
        let text_projection =
            ResizeMlp::new(cfg.hidden_size, cfg.hidden_size, vb.pp("text_projection"))?;
        let codec_head = candle_transformers::models::with_tracing::linear_no_bias(
            cfg.hidden_size,
            cfg.codec_vocab_size(),
            vb.pp("codec_head"),
        )?;

        Ok(Self {
            model,
            text_projection,
            codec_head,
            hidden_size: cfg.hidden_size,
        })
    }

    /// Get CB0 logits from hidden states.
    ///
    /// `hidden`: `[batch, seq_len, hidden_size]`
    /// Returns: `[batch, seq_len, codec_vocab_size]`
    pub fn codec_logits(&self, hidden: &Tensor) -> Result<Tensor> {
        self.codec_head.forward(hidden)
    }

    pub fn clear_kv_cache(&mut self) {
        self.model.clear_kv_cache();
    }
}
