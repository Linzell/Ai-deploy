//! Transformer encoder and decoder for DETR.
//!
//! Implements the standard transformer with:
//! - Encoder: self-attention + FFN (position embeddings added to Q and K)
//! - Decoder: self-attention + cross-attention + FFN
//!
//! Weight key layout (under `model.encoder.` or `model.decoder.`):
//! - `layers.{i}.self_attn.{q,k,v,out}_proj.{weight,bias}`
//! - `layers.{i}.self_attn_layer_norm.{weight,bias}`
//! - `layers.{i}.fc1.{weight,bias}`
//! - `layers.{i}.fc2.{weight,bias}`
//! - `layers.{i}.final_layer_norm.{weight,bias}`
//! - `layernorm.{weight,bias}`
//!
//! Decoder additionally:
//!
//! - `layers.{i}.encoder_attn.{q,k,v,out}_proj.{weight,bias}`
//! - `layers.{i}.encoder_attn_layer_norm.{weight,bias}`

use candle_core::{Module, Result, Tensor};
use candle_nn::{layer_norm, linear, LayerNorm, Linear, VarBuilder};

use super::config::DetrConfig;

// ---------------------------------------------------------------------------
// Multi-head attention
// ---------------------------------------------------------------------------

/// Multi-head attention with optional position embedding addition.
struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    fn load(vb: &VarBuilder, d_model: usize, num_heads: usize) -> Result<Self> {
        let head_dim = d_model / num_heads;
        let q_proj = linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = linear(d_model, d_model, vb.pp("v_proj"))?;
        let o_proj = linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    /// Self-attention: position embeddings added to both Q and K (not V).
    fn forward_self_attn(
        &self,
        hidden_states: &Tensor,
        position_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let qk_input = match position_embeddings {
            Some(pos) => hidden_states.broadcast_add(pos)?,
            None => hidden_states.clone(),
        };

        let q = self.q_proj.forward(&qk_input)?;
        let k = self.k_proj.forward(&qk_input)?;
        let v = self.v_proj.forward(hidden_states)?;

        self.attention(&q, &k, &v)
    }

    /// Cross-attention: query gets query_pos, key gets encoder_pos, value gets nothing.
    fn forward_cross_attn(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        query_position: Option<&Tensor>,
        encoder_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_input = match query_position {
            Some(pos) => hidden_states.broadcast_add(pos)?,
            None => hidden_states.clone(),
        };
        let k_input = match encoder_position {
            Some(pos) => encoder_hidden_states.broadcast_add(pos)?,
            None => encoder_hidden_states.clone(),
        };

        let q = self.q_proj.forward(&q_input)?;
        let k = self.k_proj.forward(&k_input)?;
        let v = self.v_proj.forward(encoder_hidden_states)?;

        self.attention(&q, &k, &v)
    }

    /// Scaled dot-product attention.
    fn attention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = q.dims3()?;
        let (_, kv_len, _) = k.dims3()?;

        // Reshape to [batch, num_heads, seq_len, head_dim]
        let q = q
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Attention: Q @ K^T / sqrt(d_k)
        let attn_weights = q.matmul(&k.transpose(2, 3)?.contiguous()?)?;
        let attn_weights = (attn_weights * self.scale)?;
        let attn_weights = candle_nn::ops::softmax(&attn_weights, 3)?;

        // Weighted values
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to [batch, seq_len, d_model]
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.o_proj.forward(&attn_output)
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

struct EncoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl EncoderLayer {
    fn load(vb: &VarBuilder, config: &DetrConfig) -> Result<Self> {
        let d = config.d_model;
        let self_attn =
            MultiHeadAttention::load(&vb.pp("self_attn"), d, config.encoder_attention_heads)?;
        let self_attn_layer_norm = layer_norm(d, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = linear(d, config.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(config.encoder_ffn_dim, d, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(d, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        position_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention with pre-norm (position added to Q,K only)
        let residual = hidden_states.clone();
        let x = self
            .self_attn
            .forward_self_attn(hidden_states, position_embeddings)?;
        let x = (residual + x)?;
        let x = self.self_attn_layer_norm.forward(&x)?;

        // FFN
        let residual = x.clone();
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = (residual + x)?;
        self.final_layer_norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

struct DecoderLayer {
    self_attn: MultiHeadAttention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: MultiHeadAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
}

impl DecoderLayer {
    fn load(vb: &VarBuilder, config: &DetrConfig) -> Result<Self> {
        let d = config.d_model;
        let self_attn =
            MultiHeadAttention::load(&vb.pp("self_attn"), d, config.decoder_attention_heads)?;
        let self_attn_layer_norm = layer_norm(d, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn =
            MultiHeadAttention::load(&vb.pp("encoder_attn"), d, config.decoder_attention_heads)?;
        let encoder_attn_layer_norm = layer_norm(d, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = linear(d, config.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = linear(config.decoder_ffn_dim, d, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(d, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        object_query_pos: Option<&Tensor>,
        encoder_pos: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Self-attention (object queries attend to each other)
        let residual = hidden_states.clone();
        let x = self
            .self_attn
            .forward_self_attn(hidden_states, object_query_pos)?;
        let x = (residual + x)?;
        let x = self.self_attn_layer_norm.forward(&x)?;

        // Cross-attention (object queries attend to encoder output)
        let residual = x.clone();
        let x = self.encoder_attn.forward_cross_attn(
            &x,
            encoder_hidden_states,
            object_query_pos,
            encoder_pos,
        )?;
        let x = (residual + x)?;
        let x = self.encoder_attn_layer_norm.forward(&x)?;

        // FFN
        let residual = x.clone();
        let x = self.fc1.forward(&x)?;
        let x = x.relu()?;
        let x = self.fc2.forward(&x)?;
        let x = (residual + x)?;
        self.final_layer_norm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Encoder
// ---------------------------------------------------------------------------

/// Transformer encoder: stack of encoder layers + final LayerNorm.
pub struct DetrEncoder {
    layers: Vec<EncoderLayer>,
    layernorm: LayerNorm,
}

impl DetrEncoder {
    pub fn load(vb: &VarBuilder, config: &DetrConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.encoder_layers);
        for i in 0..config.encoder_layers {
            layers.push(EncoderLayer::load(
                &vb.pp("layers").pp(i.to_string()),
                config,
            )?);
        }
        let layernorm = layer_norm(config.d_model, 1e-5, vb.pp("layernorm"))?;
        Ok(Self { layers, layernorm })
    }

    /// Forward pass.
    /// - `inputs_embeds`: [batch, seq_len, d_model] (projected + flattened feature map)
    /// - `position_embeddings`: [batch, seq_len, d_model] (sine positional encoding)
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_embeddings: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = inputs_embeds.clone();
        for layer in &self.layers {
            x = layer.forward(&x, position_embeddings)?;
        }
        self.layernorm.forward(&x)
    }
}

// ---------------------------------------------------------------------------
// Decoder
// ---------------------------------------------------------------------------

/// Transformer decoder: stack of decoder layers + final LayerNorm.
pub struct DetrDecoder {
    layers: Vec<DecoderLayer>,
    layernorm: LayerNorm,
}

impl DetrDecoder {
    pub fn load(vb: &VarBuilder, config: &DetrConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(config.decoder_layers);
        for i in 0..config.decoder_layers {
            layers.push(DecoderLayer::load(
                &vb.pp("layers").pp(i.to_string()),
                config,
            )?);
        }
        let layernorm = layer_norm(config.d_model, 1e-5, vb.pp("layernorm"))?;
        Ok(Self { layers, layernorm })
    }

    /// Forward pass.
    /// - `object_queries`: [batch, num_queries, d_model] (initially zeros)
    /// - `encoder_output`: [batch, seq_len, d_model]
    /// - `query_position`: [batch, num_queries, d_model] (learned embeddings)
    /// - `encoder_position`: [batch, seq_len, d_model] (sine positional encoding)
    pub fn forward(
        &self,
        object_queries: &Tensor,
        encoder_output: &Tensor,
        query_position: Option<&Tensor>,
        encoder_position: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut x = object_queries.clone();
        for layer in &self.layers {
            x = layer.forward(&x, encoder_output, query_position, encoder_position)?;
        }
        self.layernorm.forward(&x)
    }
}
