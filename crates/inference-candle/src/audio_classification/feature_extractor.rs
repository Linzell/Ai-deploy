//! CNN feature extractor for Wav2Vec2 / HuBERT.
//!
//! 7 strided 1-D convolutions that downsample raw 16 kHz waveform
//! by a total factor of 320 (5×2×2×2×2×2×2).
//!
//! Weight prefix: `wav2vec2.feature_extractor.conv_layers.{i}.conv.weight`
//! Norm prefix:   `wav2vec2.feature_extractor.conv_layers.0.layer_norm.{weight,bias}`

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::config::Wav2Vec2Config;
use crate::{TaskError, TaskResult};

/// One conv layer in the feature extractor stack.
struct ConvLayer {
    conv_weight: Tensor, // [out_ch, in_ch, kernel]
    conv_bias: Option<Tensor>,
    /// GroupNorm or LayerNorm (only layer 0 for "group" mode, all layers for "layer" mode)
    norm_weight: Option<Tensor>,
    norm_bias: Option<Tensor>,
    stride: usize,
    apply_gelu: bool,
}

impl ConvLayer {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // xs: [B, C_in, T]
        let padding = 0;
        let x = xs.conv1d(&self.conv_weight, padding, self.stride, 1, 1)?;
        let x = if let Some(bias) = &self.conv_bias {
            x.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(2)?)?
        } else {
            x
        };

        // Normalization
        let x = if let (Some(w), Some(b)) = (&self.norm_weight, &self.norm_bias) {
            // LayerNorm over channel dim: transpose to [B, T, C], norm, transpose back
            let xt = x.transpose(1, 2)?; // [B, T, C]
            let c = xt.dim(2)?;
            let mean = (xt.sum_keepdim(2)? / c as f64)?;
            let diff = xt.broadcast_sub(&mean)?;
            let var = ((&diff * &diff)?.sum_keepdim(2)? / c as f64)?;
            let eps = 1e-5;
            let normed = diff.broadcast_div(&(var + eps)?.sqrt()?)?;
            let out = normed
                .broadcast_mul(&w.unsqueeze(0)?.unsqueeze(0)?)?
                .broadcast_add(&b.unsqueeze(0)?.unsqueeze(0)?)?;
            out.transpose(1, 2)? // back to [B, C, T]
        } else {
            x
        };

        if self.apply_gelu {
            x.gelu_erf()
        } else {
            Ok(x)
        }
    }
}

/// The full CNN feature extractor (7 layers by default).
pub struct FeatureExtractor {
    layers: Vec<ConvLayer>,
}

impl FeatureExtractor {
    pub fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let n = cfg.conv_dim.len();
        let is_group_norm = cfg.feat_extract_norm == "group";
        let prefix = cfg.backbone_prefix();
        let mut layers = Vec::with_capacity(n);

        for i in 0..n {
            let in_ch = if i == 0 { 1 } else { cfg.conv_dim[i - 1] };
            let out_ch = cfg.conv_dim[i];
            let kernel = cfg.conv_kernel[i];
            let stride = cfg.conv_stride[i];

            let vb_layer = vb.pp(format!("{prefix}.feature_extractor.conv_layers.{i}"));

            let conv_weight = vb_layer
                .get((out_ch, in_ch, kernel), "conv.weight")
                .map_err(|e| TaskError::ModelLoad(format!("conv_layers.{i}.conv.weight: {e}")))?;
            let conv_bias =
                if cfg.conv_bias {
                    Some(vb_layer.get(out_ch, "conv.bias").map_err(|e| {
                        TaskError::ModelLoad(format!("conv_layers.{i}.conv.bias: {e}"))
                    })?)
                } else {
                    None
                };

            // Norm: for "group" mode, only layer 0 has GroupNorm (stored as layer_norm weights).
            // For "layer" mode, every layer has LayerNorm.
            let has_norm = if is_group_norm { i == 0 } else { true };
            let (norm_weight, norm_bias) = if has_norm {
                let w = vb_layer.get(out_ch, "layer_norm.weight").map_err(|e| {
                    TaskError::ModelLoad(format!("conv_layers.{i}.layer_norm.weight: {e}"))
                })?;
                let b = vb_layer.get(out_ch, "layer_norm.bias").map_err(|e| {
                    TaskError::ModelLoad(format!("conv_layers.{i}.layer_norm.bias: {e}"))
                })?;
                (Some(w), Some(b))
            } else {
                (None, None)
            };

            // GELU activation: applied on all layers except layer 0 when group norm
            // (layer 0 already has norm but no activation in group mode).
            // Actually in HF: all layers apply GELU after norm.
            let apply_gelu = true;

            layers.push(ConvLayer {
                conv_weight,
                conv_bias,
                norm_weight,
                norm_bias,
                stride,
                apply_gelu,
            });
        }

        Ok(Self { layers })
    }

    /// Forward: raw waveform `[B, T_raw]` → features `[B, C, T']`
    pub fn forward(&self, waveform: &Tensor) -> TaskResult<Tensor> {
        // Add channel dim: [B, T] → [B, 1, T]
        let mut x = waveform
            .unsqueeze(1)
            .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?;

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer
                .forward(&x)
                .map_err(|e| TaskError::Inference(format!("feature_extractor layer {i}: {e}")))?;
        }

        Ok(x) // [B, conv_dim[-1], T']
    }
}

/// Feature projection: LayerNorm(conv_dim) → Linear(conv_dim, hidden_size)
pub struct FeatureProjection {
    layer_norm_weight: Tensor,
    layer_norm_bias: Tensor,
    proj_weight: Tensor,
    proj_bias: Tensor,
    eps: f64,
}

impl FeatureProjection {
    pub fn load(vb: &VarBuilder, cfg: &Wav2Vec2Config) -> TaskResult<Self> {
        let prefix = cfg.backbone_prefix();
        let vb_fp = vb.pp(format!("{prefix}.feature_projection"));
        let conv_dim = cfg.last_conv_dim();

        let layer_norm_weight = vb_fp.get(conv_dim, "layer_norm.weight").map_err(|e| {
            TaskError::ModelLoad(format!("feature_projection layer_norm.weight: {e}"))
        })?;
        let layer_norm_bias = vb_fp.get(conv_dim, "layer_norm.bias").map_err(|e| {
            TaskError::ModelLoad(format!("feature_projection layer_norm.bias: {e}"))
        })?;
        let proj_weight = vb_fp
            .get((cfg.hidden_size, conv_dim), "projection.weight")
            .map_err(|e| {
                TaskError::ModelLoad(format!("feature_projection projection.weight: {e}"))
            })?;
        let proj_bias = vb_fp.get(cfg.hidden_size, "projection.bias").map_err(|e| {
            TaskError::ModelLoad(format!("feature_projection projection.bias: {e}"))
        })?;

        Ok(Self {
            layer_norm_weight,
            layer_norm_bias,
            proj_weight,
            proj_bias,
            eps: cfg.layer_norm_eps,
        })
    }

    /// Forward: `[B, C, T]` → `[B, T, hidden_size]`
    pub fn forward(&self, features: &Tensor) -> TaskResult<Tensor> {
        // Transpose to [B, T, C]
        let x = features
            .transpose(1, 2)
            .map_err(|e| TaskError::Inference(format!("transpose: {e}")))?;

        // LayerNorm
        let c = x
            .dim(2)
            .map_err(|e| TaskError::Inference(format!("dim: {e}")))? as f64;
        let mean = x
            .sum_keepdim(2)
            .map_err(|e| TaskError::Inference(format!("sum: {e}")))?
            .affine(1.0 / c, 0.0)
            .map_err(|e| TaskError::Inference(format!("affine: {e}")))?;
        let diff = x
            .broadcast_sub(&mean)
            .map_err(|e| TaskError::Inference(format!("sub: {e}")))?;
        let var = (&diff * &diff)
            .map_err(|e| TaskError::Inference(format!("mul: {e}")))?
            .sum_keepdim(2)
            .map_err(|e| TaskError::Inference(format!("sum: {e}")))?
            .affine(1.0 / c, 0.0)
            .map_err(|e| TaskError::Inference(format!("affine: {e}")))?;
        let normed = diff
            .broadcast_div(
                &(var + self.eps)
                    .map_err(|e| TaskError::Inference(format!("add eps: {e}")))?
                    .sqrt()
                    .map_err(|e| TaskError::Inference(format!("sqrt: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("div: {e}")))?;

        let x = normed
            .broadcast_mul(
                &self
                    .layer_norm_weight
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("mul: {e}")))?
            .broadcast_add(
                &self
                    .layer_norm_bias
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("add: {e}")))?;

        // Linear projection: [B, T, conv_dim] → [B, T, hidden_size]
        let out = x
            .broadcast_matmul(
                &self
                    .proj_weight
                    .t()
                    .map_err(|e| TaskError::Inference(format!("transpose: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("matmul: {e}")))?
            .broadcast_add(
                &self
                    .proj_bias
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?
                    .unsqueeze(0)
                    .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?,
            )
            .map_err(|e| TaskError::Inference(format!("add: {e}")))?;

        Ok(out)
    }
}
