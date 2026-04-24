//! Code predictor — generates codebooks 1-15 from talker hidden states.
//!
//! Architecture:
//! - `small_to_mtp_projection`: Linear(talker_hidden → predictor_hidden, bias=True)
//!   Projects from talker_hidden_size (2048) to predictor_hidden_size (1024).
//!   Applied to BOTH the initial [talker_hidden, semantic_embed] prefill and
//!   each subsequent codec embedding before the transformer layers.
//! - 15 `codec_embedding` layers, each [predictor_vocab_size, talker_hidden_size]
//! - 5 transformer layers (standard RoPE, GQA, QK norm)
//! - `norm`: final RmsNorm
//! - 15 `lm_head` layers, each [predictor_vocab_size, predictor_hidden_size]
//!
//! Weight prefix: `talker.code_predictor.*`
//!   - `codec_embedding` and `layers` live under `model.` sub-prefix
//!   - `lm_head` and `small_to_mtp_projection` live at the top level
//!
//! Generation strategy (two-phase per frame):
//!   Prefill: cat [talker_hidden, semantic_embed] → project → run 5 layers →
//!            norm → lm_heads[0] → first acoustic code
//!   Decode:  for each subsequent code, embed previous code with
//!            codec_embeddings[i-1] → project → run 5 layers (KV cached) →
//!            norm → lm_heads[i] → next acoustic code

use candle_core::{IndexOp, Module, Result, Tensor, D};
use candle_nn::{Embedding, VarBuilder};
use candle_transformers::models::with_tracing::{linear, linear_no_bias, Linear, RmsNorm};

use super::config::CodePredictorConfig;
use super::layers::DecoderLayer;
use super::rope::{apply_rotary_emb, RotaryEmbedding};

/// Code predictor sub-model.
pub struct CodePredictor {
    /// Projects talker_hidden_size (2048) → predictor_hidden_size (1024). Has bias.
    /// Applied to the concatenated prefill input AND each codec embedding.
    small_to_mtp_projection: Linear,
    /// Per-codebook embeddings (CB1-CB15). Each maps codec token → talker_hidden_size.
    /// codec_embeddings[i] embeds the code at acoustic group i.
    /// During decode, to predict code[i+1], we embed code[i] using codec_embeddings[i].
    codec_embeddings: Vec<Embedding>,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    /// Per-codebook LM heads (CB1-CB15). Each maps predictor_hidden → predictor_vocab_size.
    /// lm_heads[i] predicts acoustic code i (0-indexed).
    lm_heads: Vec<Linear>,
    rotary: RotaryEmbedding,
    num_codebooks: usize,
}

impl CodePredictor {
    pub fn new(
        cfg: &CodePredictorConfig,
        talker_hidden_size: usize,
        _talker_codec_vocab_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let num_codebooks = cfg.num_codebooks();
        let predictor_vocab_size = cfg.predictor_vocab_size();

        // small_to_mtp_projection has bias (Linear, not linear_no_bias)
        let small_to_mtp_projection = linear(
            talker_hidden_size,
            cfg.hidden_size,
            vb.pp("small_to_mtp_projection"),
        )?;

        // Codec embeddings: num_codebooks entries, each [predictor_vocab_size, talker_hidden_size]
        // In the HF weights: `talker.code_predictor.model.codec_embedding.{0..14}.weight`
        let mut codec_embeddings = Vec::with_capacity(num_codebooks);
        let vb_emb = vb.pp("model").pp("codec_embedding");
        for i in 0..num_codebooks {
            codec_embeddings.push(candle_nn::embedding(
                predictor_vocab_size,
                talker_hidden_size,
                vb_emb.pp(i),
            )?);
        }

        // Transformer layers
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_layers = vb.pp("model").pp("layers");
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

        let norm = RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("model").pp("norm"))?;

        // LM heads: num_codebooks entries (CB1-CB15), each [predictor_hidden → predictor_vocab_size].
        // In the HF weights: `talker.code_predictor.lm_head.{0..14}.weight`
        let mut lm_heads = Vec::with_capacity(num_codebooks);
        let vb_heads = vb.pp("lm_head");
        for i in 0..num_codebooks {
            lm_heads.push(linear_no_bias(
                cfg.hidden_size,
                predictor_vocab_size,
                vb_heads.pp(i),
            )?);
        }

        let rotary = RotaryEmbedding::new(
            cfg.head_dim,
            cfg.rope_theta,
            256, // code predictor only runs short sequences (~17 tokens max)
            vb.pp("small_to_mtp_projection").device(),
        )?;

        Ok(Self {
            small_to_mtp_projection,
            codec_embeddings,
            layers,
            norm,
            lm_heads,
            rotary,
            num_codebooks,
        })
    }

    /// Predict codebooks 1-15 for a single frame.
    ///
    /// Two-phase generation per frame:
    ///   1. **Prefill**: concat `[talker_hidden, semantic_embed]` → project →
    ///      run transformer layers → norm → `lm_heads[0]` at last position → code[0]
    ///   2. **Decode**: for i in 1..15, embed code[i-1] with `codec_embeddings[i-1]` →
    ///      project → run transformer layers (KV cached) → norm → `lm_heads[i]` → code[i]
    ///
    /// # Arguments
    /// * `talker_hidden` — hidden state from talker at this frame, shape `[1, 1, talker_hidden_size]`
    /// * `semantic_embed` — embedding of CB0 token from talker's codec_embedding, shape `[1, 1, talker_hidden_size]`
    /// * `temperature` — sampling temperature (0 = greedy)
    ///
    /// # Returns
    /// The 15 predicted acoustic code token values (CB1-CB15).
    pub fn predict_codebooks(
        &mut self,
        talker_hidden: &Tensor,
        semantic_embed: &Tensor,
        temperature: f64,
    ) -> Result<Vec<u32>> {
        let device = talker_hidden.device();

        // Clear KV cache for fresh sequence
        self.clear_kv_cache();

        // Phase 1: Prefill with [talker_hidden, semantic_embed]
        // Both are [1, 1, talker_hidden_size], concatenate along seq dim → [1, 2, talker_hidden_size]
        let prefill_input = Tensor::cat(&[talker_hidden, semantic_embed], 1)?;

        // Project from talker_hidden_size (2048) → predictor_hidden_size (1024)
        // [1, 2, 2048] → [1, 2, 1024]
        let projected = self.small_to_mtp_projection.forward(&prefill_input)?;

        // Run through transformer layers with standard RoPE (prefill: seq_len=2, offset=0)
        let seq_len = 2usize;
        let mut hidden = projected;
        let rotary = &mut self.rotary;
        for layer in &mut self.layers {
            hidden = layer.forward_with_rope_fn(&hidden, &mut |q, k| {
                let (cos, sin) = rotary.get_cos_sin(seq_len, 0)?;
                let q_rot = apply_rotary_emb(q, &cos, &sin)?;
                let k_rot = apply_rotary_emb(k, &cos, &sin)?;
                Ok((q_rot, k_rot))
            })?;
        }

        let normed = self.norm.forward(&hidden)?;

        // Predict first acoustic code from last position of prefill
        // normed: [1, 2, predictor_hidden] → take position 1 → [1, 1, predictor_hidden]
        let last_hidden = normed.i((.., seq_len - 1..seq_len, ..))?;
        let logits = self.lm_heads[0].forward(&last_hidden)?;
        let first_code = greedy_or_sample(&logits, temperature)?;

        let mut predicted_tokens = Vec::with_capacity(self.num_codebooks);
        predicted_tokens.push(first_code);

        // Phase 2: Autoregressively generate remaining 14 codes
        let mut prev_code = first_code;

        for (offset, group_idx) in (seq_len..).zip(1..self.num_codebooks) {
            // Embed previous code using codec_embeddings[group_idx - 1]
            let code_tensor = Tensor::new(&[prev_code], device)?;
            let code_emb = self.codec_embeddings[group_idx - 1].forward(&code_tensor)?;
            // [1, talker_hidden_size] → [1, 1, talker_hidden_size]
            let code_emb = code_emb.unsqueeze(0)?;

            // Project to predictor hidden size
            // [1, 1, talker_hidden_size] → [1, 1, predictor_hidden_size]
            let code_proj = self.small_to_mtp_projection.forward(&code_emb)?;

            // Run through transformer layers (single token, KV cached, no causal mask needed)
            let mut h = code_proj;
            let cur_offset = offset;
            let rotary = &mut self.rotary;
            for layer in &mut self.layers {
                h = layer.forward_with_rope_fn(&h, &mut |q, k| {
                    let (cos, sin) = rotary.get_cos_sin(1, cur_offset)?;
                    let q_rot = apply_rotary_emb(q, &cos, &sin)?;
                    let k_rot = apply_rotary_emb(k, &cos, &sin)?;
                    Ok((q_rot, k_rot))
                })?;
            }

            let normed = self.norm.forward(&h)?;

            // Predict next code
            let logits = self.lm_heads[group_idx].forward(&normed)?;
            let next_code = greedy_or_sample(&logits, temperature)?;

            predicted_tokens.push(next_code);
            prev_code = next_code;
        }

        Ok(predicted_tokens)
    }

    pub fn clear_kv_cache(&mut self) {
        for layer in &mut self.layers {
            layer.clear_kv_cache();
        }
    }
}

/// Greedy or temperature-scaled sampling from logits.
///
/// `logits`: any shape (flattened to 1-D internally).
/// Returns: a single `u32` token value.
fn greedy_or_sample(logits: &Tensor, temperature: f64) -> Result<u32> {
    let flat = logits.flatten_all()?;
    let token = if temperature <= 0.0 {
        flat.argmax(0)?
    } else {
        let scaled = (flat / temperature)?;
        let probs = candle_nn::ops::softmax_last_dim(&scaled)?;
        probs.argmax(D::Minus1)? // argmax on probs == greedy on scaled logits
    };
    token.to_scalar()
}
