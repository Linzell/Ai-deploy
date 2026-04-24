//! Full CLAP model for zero-shot audio classification.
//!
//! CLAP = Contrastive Language-Audio Pretraining.
//! Dual-encoder architecture:
//! - Audio encoder (HTS-AT) → audio embedding → audio projection → [B, 512]
//! - Text encoder (RoBERTa) → text embedding → text projection → [N_labels, 512]
//! - Cosine similarity → softmax → class scores
//!
//! For audio classification, we encode the audio once and encode each candidate
//! label text, then pick the best match (zero-shot classification).

use candle_core::{DType, Tensor};
use candle_nn::VarBuilder;

use super::clap_audio_encoder::ClapAudioEncoder;
use super::clap_config::ClapConfig;
use super::clap_text_encoder::ClapTextEncoder;
use crate::{TaskError, TaskResult};

// ---------------------------------------------------------------------------
// Projection head: Linear → ReLU → Linear
// ---------------------------------------------------------------------------

struct ProjectionHead {
    linear1_weight: Tensor,
    linear1_bias: Tensor,
    linear2_weight: Tensor,
    linear2_bias: Tensor,
}

impl ProjectionHead {
    fn load(vb: &VarBuilder, in_dim: usize, proj_dim: usize) -> TaskResult<Self> {
        let linear1_weight = vb
            .get((proj_dim, in_dim), "linear1.weight")
            .map_err(|e| TaskError::ModelLoad(format!("proj linear1.weight: {e}")))?;
        let linear1_bias = vb
            .get(proj_dim, "linear1.bias")
            .map_err(|e| TaskError::ModelLoad(format!("proj linear1.bias: {e}")))?;
        let linear2_weight = vb
            .get((proj_dim, proj_dim), "linear2.weight")
            .map_err(|e| TaskError::ModelLoad(format!("proj linear2.weight: {e}")))?;
        let linear2_bias = vb
            .get(proj_dim, "linear2.bias")
            .map_err(|e| TaskError::ModelLoad(format!("proj linear2.bias: {e}")))?;
        Ok(Self {
            linear1_weight,
            linear1_bias,
            linear2_weight,
            linear2_bias,
        })
    }

    /// Forward: `[B, in_dim]` → `[B, proj_dim]`
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x = x
            .contiguous()?
            .broadcast_matmul(&self.linear1_weight.t()?.contiguous()?)?
            .broadcast_add(&self.linear1_bias.unsqueeze(0)?)?;
        let x = x.relu()?;
        x.contiguous()?
            .broadcast_matmul(&self.linear2_weight.t()?.contiguous()?)?
            .broadcast_add(&self.linear2_bias.unsqueeze(0)?)
    }
}

// ---------------------------------------------------------------------------
// Full CLAP model
// ---------------------------------------------------------------------------

/// CLAP model for zero-shot audio classification.
pub struct ClapModel {
    audio_encoder: ClapAudioEncoder,
    text_encoder: ClapTextEncoder,
    audio_projection: ProjectionHead,
    text_projection: ProjectionHead,
    logit_scale_a: Tensor, // scalar
}

impl ClapModel {
    pub fn load(vb: &VarBuilder, cfg: &ClapConfig) -> TaskResult<Self> {
        let audio_encoder = ClapAudioEncoder::load(vb, &cfg.audio_config)?;
        let text_encoder = ClapTextEncoder::load(vb, &cfg.text_config)?;

        let audio_projection = ProjectionHead::load(
            &vb.pp("audio_projection"),
            cfg.audio_config.hidden_size,
            cfg.projection_dim,
        )?;

        let text_projection = ProjectionHead::load(
            &vb.pp("text_projection"),
            cfg.text_config.hidden_size,
            cfg.projection_dim,
        )?;

        let logit_scale_a = vb
            .get(1, "logit_scale_a")
            .or_else(|_| vb.get((), "logit_scale_a"))
            .map_err(|e| TaskError::ModelLoad(format!("logit_scale_a: {e}")))?;

        Ok(Self {
            audio_encoder,
            text_encoder,
            audio_projection,
            text_projection,
            logit_scale_a,
        })
    }

    /// Encode audio: mel spectrogram `[B, 1, time, freq]` → normalized embedding `[B, proj_dim]`
    pub fn encode_audio(&self, mel: &Tensor) -> TaskResult<Tensor> {
        let audio_features = self.audio_encoder.forward(mel)?;
        let projected = self
            .audio_projection
            .forward(&audio_features)
            .map_err(|e| TaskError::Inference(format!("audio projection: {e}")))?;
        // L2 normalize
        l2_normalize(&projected)
    }

    /// Encode text: input_ids `[N, S]` → normalized embedding `[N, proj_dim]`
    pub fn encode_text(&self, input_ids: &Tensor) -> TaskResult<Tensor> {
        let text_features = self.text_encoder.forward(input_ids)?;
        let projected = self
            .text_projection
            .forward(&text_features)
            .map_err(|e| TaskError::Inference(format!("text projection: {e}")))?;
        // L2 normalize
        l2_normalize(&projected)
    }

    /// Zero-shot audio classification.
    ///
    /// Given audio mel spectrogram and tokenized label texts, returns
    /// per-label scores (softmax of cosine similarities).
    ///
    /// - `mel`: `[1, 1, time, freq]` — single audio sample
    /// - `label_input_ids`: `[N_labels, S]` — tokenized label texts
    ///
    /// Returns: `[N_labels]` scores summing to 1.
    pub fn classify_zero_shot(
        &self,
        mel: &Tensor,
        label_input_ids: &Tensor,
    ) -> TaskResult<Vec<f32>> {
        let audio_emb = self.encode_audio(mel)?; // [1, proj_dim]
        let text_emb = self.encode_text(label_input_ids)?; // [N, proj_dim]

        // Cosine similarity: audio_emb @ text_emb^T → [1, N]
        let logit_scale = self
            .logit_scale_a
            .flatten_all()
            .and_then(|t| t.to_dtype(DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| TaskError::Inference(format!("logit_scale: {e}")))?;
        let scale = logit_scale[0].exp();

        let similarity = audio_emb
            .contiguous()
            .and_then(|a| a.broadcast_matmul(&text_emb.t()?.contiguous()?))
            .map_err(|e| TaskError::Inference(format!("similarity: {e}")))?;
        // Scale
        let similarity = (similarity * f64::from(scale))
            .map_err(|e| TaskError::Inference(format!("scale similarity: {e}")))?;

        // Softmax over labels dim
        let scores = similarity
            .squeeze(0)
            .and_then(|s| candle_nn::ops::softmax_last_dim(&s))
            .and_then(|s| s.to_dtype(DType::F32))
            .and_then(|s| s.to_vec1::<f32>())
            .map_err(|e| TaskError::Inference(format!("softmax scores: {e}")))?;

        Ok(scores)
    }
}

/// L2-normalize along the last dimension.
fn l2_normalize(x: &Tensor) -> TaskResult<Tensor> {
    let norm = x
        .sqr()
        .and_then(|s| s.sum_keepdim(candle_core::D::Minus1))
        .and_then(|s| (s + 1e-12)?.sqrt())
        .map_err(|e| TaskError::Inference(format!("l2 norm: {e}")))?;
    x.broadcast_div(&norm)
        .map_err(|e| TaskError::Inference(format!("l2 div: {e}")))
}
