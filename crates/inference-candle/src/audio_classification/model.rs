//! Wav2Vec2 / HuBERT complete model for sequence classification.
//!
//! Assembles: FeatureExtractor → FeatureProjection → Encoder → Classification Head.

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::config::Wav2Vec2Config;
use super::encoder::Wav2Vec2Encoder;
use super::feature_extractor::{FeatureExtractor, FeatureProjection};
use crate::{TaskError, TaskResult};

/// Classification head.
///
/// Supports two HuggingFace variants:
/// 1. Standard: `projector` → Tanh → `classifier` (Wav2Vec2ForSequenceClassification)
/// 2. Dense:    `classifier.dense` → Tanh → `classifier.output` (older emotion models)
struct ClassificationHead {
    projector_weight: Tensor,
    projector_bias: Tensor,
    classifier_weight: Tensor,
    classifier_bias: Tensor,
}

impl ClassificationHead {
    fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        // Try standard layout first: projector.weight + classifier.weight
        let standard = vb
            .get(
                (cfg.classifier_proj_size, cfg.hidden_size),
                "projector.weight",
            )
            .ok();

        if let Some(pw) = standard {
            let pb = vb
                .get(cfg.classifier_proj_size, "projector.bias")
                .map_err(|e| TaskError::ModelLoad(format!("projector.bias: {e}")))?;
            let cw = vb
                .get(
                    (cfg.num_labels, cfg.classifier_proj_size),
                    "classifier.weight",
                )
                .map_err(|e| TaskError::ModelLoad(format!("classifier.weight: {e}")))?;
            let cb = vb
                .get(cfg.num_labels, "classifier.bias")
                .map_err(|e| TaskError::ModelLoad(format!("classifier.bias: {e}")))?;
            return Ok(Self {
                projector_weight: pw,
                projector_bias: pb,
                classifier_weight: cw,
                classifier_bias: cb,
            });
        }

        // Alternate layout: classifier.dense + classifier.output (or classifier.out_proj)
        let vb_cls = vb.pp("classifier");
        let pw = vb_cls
            .get((cfg.classifier_proj_size, cfg.hidden_size), "dense.weight")
            .map_err(|e| {
                TaskError::ModelLoad(format!(
                    "Neither projector.weight nor classifier.dense.weight found: {e}"
                ))
            })?;
        let pb = vb_cls
            .get(cfg.classifier_proj_size, "dense.bias")
            .map_err(|e| TaskError::ModelLoad(format!("classifier.dense.bias: {e}")))?;

        // Try classifier.output first, then classifier.out_proj
        let (cw, cb) = if let Ok(w) =
            vb_cls.get((cfg.num_labels, cfg.classifier_proj_size), "output.weight")
        {
            let b = vb_cls
                .get(cfg.num_labels, "output.bias")
                .map_err(|e| TaskError::ModelLoad(format!("classifier.output.bias: {e}")))?;
            (w, b)
        } else {
            let w = vb_cls
                .get(
                    (cfg.num_labels, cfg.classifier_proj_size),
                    "out_proj.weight",
                )
                .map_err(|e| TaskError::ModelLoad(format!("classifier.out_proj.weight: {e}")))?;
            let b = vb_cls
                .get(cfg.num_labels, "out_proj.bias")
                .map_err(|e| TaskError::ModelLoad(format!("classifier.out_proj.bias: {e}")))?;
            (w, b)
        };

        Ok(Self {
            projector_weight: pw,
            projector_bias: pb,
            classifier_weight: cw,
            classifier_bias: cb,
        })
    }

    /// Forward: pooled `[B, H]` → logits `[B, num_labels]`
    fn forward(&self, pooled: &Tensor) -> candle_core::Result<Tensor> {
        // projector: [B, H] @ W^T + b → [B, proj_size]
        let x = pooled
            .contiguous()?
            .broadcast_matmul(&self.projector_weight.t()?.contiguous()?)?
            .broadcast_add(&self.projector_bias.unsqueeze(0)?)?;
        let x = x.tanh()?;
        // classifier: [B, proj_size] @ W^T + b → [B, num_labels]
        x.contiguous()?
            .broadcast_matmul(&self.classifier_weight.t()?.contiguous()?)?
            .broadcast_add(&self.classifier_bias.unsqueeze(0)?)
    }
}

/// Full Wav2Vec2 / HuBERT model for audio sequence classification.
pub struct Wav2Vec2ForSequenceClassification {
    feature_extractor: FeatureExtractor,
    feature_projection: FeatureProjection,
    encoder: Wav2Vec2Encoder,
    head: ClassificationHead,
}

impl Wav2Vec2ForSequenceClassification {
    pub fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let feature_extractor = FeatureExtractor::load(vb, cfg)?;
        let feature_projection = FeatureProjection::load(vb, cfg)?;
        let encoder = Wav2Vec2Encoder::load(vb, cfg)?;
        let head = ClassificationHead::load(vb, cfg)?;

        Ok(Self {
            feature_extractor,
            feature_projection,
            encoder,
            head,
        })
    }

    /// Forward: raw waveform `[B, T_raw]` → logits `[B, num_labels]`
    pub fn forward(&self, waveform: &Tensor) -> TaskResult<Tensor> {
        // CNN features: [B, T_raw] → [B, C, T']
        let features = self.feature_extractor.forward(waveform)?;

        // Project: [B, C, T'] → [B, T', H]
        let hidden = self.feature_projection.forward(&features)?;

        // Transformer encoder: [B, T', H] → [B, T', H]
        let encoded = self.encoder.forward(&hidden)?;

        // Mean pooling over time: [B, T', H] → [B, H]
        let pooled = encoded
            .mean(1)
            .map_err(|e| TaskError::Inference(format!("mean pool: {e}")))?;

        // Classification: [B, H] → [B, num_labels]
        self.head
            .forward(&pooled)
            .map_err(|e| TaskError::Inference(format!("classification head: {e}")))
    }
}
