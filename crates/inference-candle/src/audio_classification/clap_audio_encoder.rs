//! HTS-AT audio encoder for CLAP.
//!
//! Implements the audio encoder pipeline:
//! 1. BatchNorm2d on mel spectrogram
//! 2. reshape_mel2img (rearrange mel into 2D "image" for Swin)
//! 3. Patch embedding (Conv2d → flatten) with optional AFF fusion
//! 4. 4-stage Swin Transformer
//! 5. Final LayerNorm + reshape + average pool → [B, hidden_size]

use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::clap_config::ClapAudioConfig;
use super::clap_swin::SwinStage;
use crate::{TaskError, TaskResult};

// ---------------------------------------------------------------------------
// BatchNorm2d (inference-mode only, no running stats update)
// ---------------------------------------------------------------------------

struct BatchNorm2d {
    weight: Tensor,       // [C]
    bias: Tensor,         // [C]
    running_mean: Tensor, // [C]
    running_var: Tensor,  // [C]
    eps: f64,
}

impl BatchNorm2d {
    fn load(vb: &VarBuilder, num_features: usize) -> TaskResult<Self> {
        let weight = vb
            .get(num_features, "weight")
            .map_err(|e| TaskError::ModelLoad(format!("bn weight: {e}")))?;
        let bias = vb
            .get(num_features, "bias")
            .map_err(|e| TaskError::ModelLoad(format!("bn bias: {e}")))?;
        let running_mean = vb
            .get(num_features, "running_mean")
            .map_err(|e| TaskError::ModelLoad(format!("bn running_mean: {e}")))?;
        let running_var = vb
            .get(num_features, "running_var")
            .map_err(|e| TaskError::ModelLoad(format!("bn running_var: {e}")))?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    /// Forward: `[B, C, H, W]` → `[B, C, H, W]` (normalize along C dim).
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let device = x.device();
        let dtype = x.dtype();
        let mean = self.running_mean.to_device(device)?.to_dtype(dtype)?;
        let var = self.running_var.to_device(device)?.to_dtype(dtype)?;
        let w = self.weight.to_device(device)?.to_dtype(dtype)?;
        let b = self.bias.to_device(device)?.to_dtype(dtype)?;

        // Reshape for broadcasting: [1, C, 1, 1]
        let mean = mean.reshape((1, (), 1, 1))?;
        let var = var.reshape((1, (), 1, 1))?;
        let w = w.reshape((1, (), 1, 1))?;
        let b = b.reshape((1, (), 1, 1))?;

        let std = (var + self.eps)?.sqrt()?;
        let x_norm = x.broadcast_sub(&mean)?.broadcast_div(&std)?;
        x_norm.broadcast_mul(&w)?.broadcast_add(&b)
    }
}

// ---------------------------------------------------------------------------
// AFF (Attentional Feature Fusion)
// ---------------------------------------------------------------------------

/// Attentional Feature Fusion block.
///
/// Fuses two feature maps using learned attention:
/// `output = sigmoid(local_att(x1+x2) + global_att(x1+x2)) * x1 + (1-sig) * x2`
///
/// Weights are loaded but forward is only used when multi-crop fusion is active.
#[allow(dead_code)]
struct AffBlock {
    // Local attention: Conv2d(C, C/r, 1) → BN → ReLU → Conv2d(C/r, C, 1) → BN
    local_conv1_weight: Tensor,
    local_bn1: BatchNorm2d,
    local_conv2_weight: Tensor,
    local_bn2: BatchNorm2d,
    // Global attention: same but with AdaptiveAvgPool2d(1) first
    global_conv1_weight: Tensor,
    global_bn1: BatchNorm2d,
    global_conv2_weight: Tensor,
    global_bn2: BatchNorm2d,
}

impl AffBlock {
    fn load(vb: &VarBuilder, channels: usize, r: usize) -> TaskResult<Self> {
        let mid = channels / r;

        let vb_local = vb.pp("local_att");
        let local_conv1_weight = vb_local
            .get((mid, channels, 1, 1), "0.weight")
            .map_err(|e| TaskError::ModelLoad(format!("local_att.0.weight: {e}")))?;
        let local_bn1 = BatchNorm2d::load(&vb_local.pp("1"), mid)?;
        let local_conv2_weight = vb_local
            .get((channels, mid, 1, 1), "3.weight")
            .map_err(|e| TaskError::ModelLoad(format!("local_att.3.weight: {e}")))?;
        let local_bn2 = BatchNorm2d::load(&vb_local.pp("4"), channels)?;

        let vb_global = vb.pp("global_att");
        // Global skips index 0 (AdaptiveAvgPool2d), so Conv starts at index 1
        let global_conv1_weight = vb_global
            .get((mid, channels, 1, 1), "1.weight")
            .map_err(|e| TaskError::ModelLoad(format!("global_att.1.weight: {e}")))?;
        let global_bn1 = BatchNorm2d::load(&vb_global.pp("2"), mid)?;
        let global_conv2_weight = vb_global
            .get((channels, mid, 1, 1), "4.weight")
            .map_err(|e| TaskError::ModelLoad(format!("global_att.4.weight: {e}")))?;
        let global_bn2 = BatchNorm2d::load(&vb_global.pp("5"), channels)?;

        Ok(Self {
            local_conv1_weight,
            local_bn1,
            local_conv2_weight,
            local_bn2,
            global_conv1_weight,
            global_bn1,
            global_conv2_weight,
            global_bn2,
        })
    }

    /// Forward: fuse `x1` and `x2`, both `[B, C, H, W]`.
    #[allow(dead_code)]
    fn forward(&self, x1: &Tensor, x2: &Tensor) -> candle_core::Result<Tensor> {
        let feat = (x1 + x2)?;

        // Local path: 1×1 conv → BN → ReLU → 1×1 conv → BN
        let local = conv2d_1x1(&feat, &self.local_conv1_weight)?;
        let local = self.local_bn1.forward(&local)?;
        let local = local.relu()?;
        let local = conv2d_1x1(&local, &self.local_conv2_weight)?;
        let local = self.local_bn2.forward(&local)?;

        // Global path: avg pool to 1×1 → 1×1 conv → BN → ReLU → 1×1 conv → BN
        // AdaptiveAvgPool2d(1): mean over H, W dims
        let global = feat.mean_keepdim(2)?.mean_keepdim(3)?;
        let global = conv2d_1x1(&global, &self.global_conv1_weight)?;
        let global = self.global_bn1.forward(&global)?;
        let global = global.relu()?;
        let global = conv2d_1x1(&global, &self.global_conv2_weight)?;
        let global = self.global_bn2.forward(&global)?;

        // Attention weight: sigmoid(local + global broadcast to same shape)
        let attn = local.broadcast_add(&global)?;
        let attn = candle_nn::ops::sigmoid(&attn)?;

        // Weighted fusion: 2 * attn * x1 + 2 * (1 - attn) * x2
        // (matches Python: output = 2 * hidden_states * fused + 2 * residual * (1 - fused))
        let fused = (attn
            .broadcast_mul(x1)?
            .broadcast_add(&(1.0 - &attn)?.broadcast_mul(x2)?)?
            * 2.0)?;
        Ok(fused)
    }
}

/// 1×1 convolution: `[B, C_in, H, W]` × `[C_out, C_in, 1, 1]` → `[B, C_out, H, W]`
#[allow(dead_code)]
fn conv2d_1x1(x: &Tensor, weight: &Tensor) -> candle_core::Result<Tensor> {
    let (b, _c_in, h, w) = x.dims4()?;
    let c_out = weight.dim(0)?;
    let c_in = weight.dim(1)?;
    // Reshape to [B, C_in, H*W], weight to [C_out, C_in]
    let x_flat = x.reshape((b, c_in, h * w))?.contiguous()?;
    let w_flat = weight.reshape((c_out, c_in))?.contiguous()?;
    // [C_out, C_in] @ [B, C_in, H*W] → matmul per batch
    // Actually: [B, C_in, H*W] → permute → [B, H*W, C_in] @ [C_in, C_out] → [B, H*W, C_out]
    let x_flat = x_flat.permute((0, 2, 1))?.contiguous()?; // [B, H*W, C_in]
    let out = x_flat.broadcast_matmul(&w_flat.t()?.contiguous()?)?; // [B, H*W, C_out]
    out.permute((0, 2, 1))?.reshape((b, c_out, h, w))
}

// ---------------------------------------------------------------------------
// Patch Embedding
// ---------------------------------------------------------------------------

/// Patch embedding: Conv2d(1, embed_dim, patch_size, patch_stride) + optional AFF fusion.
struct PatchEmbed {
    proj_weight: Tensor, // [embed_dim, 1, patch_h, patch_w]
    proj_bias: Tensor,   // [embed_dim]
    norm: Option<PatchLayerNorm>,
    #[allow(dead_code)]
    fusion: Option<PatchFusion>,
    patch_stride: (usize, usize),
}

struct PatchLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

/// Weights are loaded but only used when multi-crop fusion is active during inference.
#[allow(dead_code)]
struct PatchFusion {
    mel_conv2d_weight: Tensor, // [embed_dim, 1, patch_h, patch_w]
    mel_conv2d_bias: Tensor,
    aff: AffBlock,
}

impl PatchEmbed {
    fn load(vb: &VarBuilder, cfg: &ClapAudioConfig) -> TaskResult<Self> {
        let embed_dim = cfg.patch_embeds_hidden_size;
        let ph = cfg.patch_stride[0];
        let pw = cfg.patch_stride[1];

        let proj_weight = vb
            .get((embed_dim, 1, ph, pw), "proj.weight")
            .map_err(|e| TaskError::ModelLoad(format!("patch_embed.proj.weight: {e}")))?;
        let proj_bias = vb
            .get(embed_dim, "proj.bias")
            .map_err(|e| TaskError::ModelLoad(format!("patch_embed.proj.bias: {e}")))?;

        let norm = if cfg.enable_patch_layer_norm {
            let w = vb
                .get(embed_dim, "norm.weight")
                .map_err(|e| TaskError::ModelLoad(format!("patch_embed.norm.weight: {e}")))?;
            let b = vb
                .get(embed_dim, "norm.bias")
                .map_err(|e| TaskError::ModelLoad(format!("patch_embed.norm.bias: {e}")))?;
            Some(PatchLayerNorm {
                weight: w,
                bias: b,
                eps: 1e-5,
            })
        } else {
            None
        };

        let fusion = if cfg.enable_fusion {
            // mel_conv2d kernel: (patch_h, patch_w * 3) — hardcoded multiplier from Python source
            let mel_kw = pw * 3;
            let mel_w = vb
                .get((embed_dim, 1, ph, mel_kw), "mel_conv2d.weight")
                .map_err(|e| TaskError::ModelLoad(format!("patch_embed.mel_conv2d.weight: {e}")))?;
            let mel_b = vb
                .get(embed_dim, "mel_conv2d.bias")
                .map_err(|e| TaskError::ModelLoad(format!("patch_embed.mel_conv2d.bias: {e}")))?;
            let aff = AffBlock::load(&vb.pp("fusion_model"), embed_dim, cfg.aff_block_r)?;
            Some(PatchFusion {
                mel_conv2d_weight: mel_w,
                mel_conv2d_bias: mel_b,
                aff,
            })
        } else {
            None
        };

        Ok(Self {
            proj_weight,
            proj_bias,
            norm,
            fusion,
            patch_stride: (ph, pw),
        })
    }

    /// Forward: `[B, 1, H, W]` → `[B, num_patches, embed_dim]`
    ///
    /// For inference, we only use the global path (no multi-crop fusion).
    /// The fusion weights are loaded but only used when `mel_for_fusion` is provided.
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let (b, _c, _h_in, _w_in) = x.dims4()?;
        let (ph, pw) = self.patch_stride;
        let embed_dim = self.proj_weight.dim(0)?;

        // Conv2d via im2col-style manual implementation
        let feat = conv2d_stride(x, &self.proj_weight, &self.proj_bias, ph, pw)?;
        // feat: [B, embed_dim, H/ph, W/pw]

        // Flatten spatial dims: [B, embed_dim, H', W'] → [B, H'*W', embed_dim]
        let (_, _, h_out, w_out) = feat.dims4()?;
        let feat = feat.reshape((b, embed_dim, h_out * w_out))?;
        let feat = feat.permute((0, 2, 1))?; // [B, N, embed_dim]

        // Optional LayerNorm
        if let Some(ln) = &self.norm {
            layer_norm_simple(&feat, &ln.weight, &ln.bias, ln.eps)
        } else {
            Ok(feat)
        }
    }
}

/// Strided 2D convolution: `[B, 1, H, W]` × `[C_out, 1, kH, kW]` → `[B, C_out, H/sH, W/sW]`
///
/// Only supports in_channels=1 (our use case).
fn conv2d_stride(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    stride_h: usize,
    stride_w: usize,
) -> candle_core::Result<Tensor> {
    let (b, _c, h, w) = x.dims4()?;
    let (c_out, _c_in, kh, kw) = weight.dims4()?;
    let out_h = h / stride_h;
    let out_w = w / stride_w;

    // Extract patches: for each output position, gather a kH×kW patch
    let x = x.squeeze(1)?; // [B, H, W]
    let mut patches = Vec::with_capacity(out_h * out_w);
    for i in 0..out_h {
        for j in 0..out_w {
            let patch = x.narrow(1, i * stride_h, kh)?.narrow(2, j * stride_w, kw)?;
            patches.push(patch.reshape((b, 1, kh * kw))?);
        }
    }
    // [B, out_h*out_w, kh*kw]
    let patches = Tensor::cat(&patches, 1)?;
    // Weight: [c_out, 1, kh, kw] → [c_out, kh*kw] → [kh*kw, c_out]
    let w_flat = weight.reshape((c_out, kh * kw))?.t()?.contiguous()?;
    // [B, out_h*out_w, kh*kw] @ [kh*kw, c_out] → [B, out_h*out_w, c_out]
    let out = patches.contiguous()?.broadcast_matmul(&w_flat)?;
    // Add bias [c_out]
    let out = out.broadcast_add(&bias.unsqueeze(0)?.unsqueeze(0)?)?;
    // Reshape to [B, c_out, out_h, out_w]
    out.permute((0, 2, 1))?.reshape((b, c_out, out_h, out_w))
}

/// Simple layer norm over last dim.
fn layer_norm_simple(
    x: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f64,
) -> candle_core::Result<Tensor> {
    let mean = x.mean_keepdim(candle_core::D::Minus1)?;
    let x_centered = x.broadcast_sub(&mean)?;
    let var = (&x_centered * &x_centered)?.mean_keepdim(candle_core::D::Minus1)?;
    let std = (var + eps)?.sqrt()?;
    let norm = x_centered.broadcast_div(&std)?;
    norm.broadcast_mul(weight)?.broadcast_add(bias)
}

// ---------------------------------------------------------------------------
// Full HTS-AT Audio Encoder
// ---------------------------------------------------------------------------

/// Complete HTS-AT audio encoder for CLAP.
///
/// Pipeline:
/// 1. BatchNorm2d(64) on transposed mel spectrogram
/// 2. reshape_mel2img: rearrange mel → 2D image for Swin
/// 3. PatchEmbed (with optional AFF fusion)
/// 4. 4 Swin Transformer stages
/// 5. Final LayerNorm + adaptive average pool → [B, hidden_size]
pub struct ClapAudioEncoder {
    batch_norm: BatchNorm2d,
    patch_embed: PatchEmbed,
    stages: Vec<SwinStage>,
    final_norm: FinalLayerNorm,
    spec_size: usize,
    num_mel_bins: usize,
}

struct FinalLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ClapAudioEncoder {
    pub fn load(vb: &VarBuilder, cfg: &ClapAudioConfig) -> TaskResult<Self> {
        let vb_enc = vb.pp("audio_model").pp("audio_encoder");

        // BatchNorm2d(num_mel_bins) — applied to transposed mel
        let batch_norm = BatchNorm2d::load(&vb_enc.pp("batch_norm"), cfg.num_mel_bins)?;

        // Patch embedding
        let patch_embed = PatchEmbed::load(&vb_enc.pp("patch_embed"), cfg)?;

        // Swin stages
        let (mut h, mut w) = cfg.patches_resolution();
        let num_stages = cfg.num_stages();
        let mut stages = Vec::with_capacity(num_stages);

        for i in 0..num_stages {
            let dim = cfg.stage_dim(i);
            let depth = cfg.depths[i];
            let heads = cfg.num_attention_heads[i];
            let is_last = i == num_stages - 1;

            let stage = SwinStage::load(
                &vb_enc.pp("layers").pp(i.to_string()),
                dim,
                depth,
                heads,
                cfg.window_size,
                cfg.mlp_ratio,
                cfg.qkv_bias,
                h,
                w,
                is_last,
            )?;

            stages.push(stage);
            if !is_last {
                h /= 2;
                w /= 2;
            }
        }

        // Final LayerNorm
        let norm_w = vb_enc
            .get(cfg.hidden_size, "norm.weight")
            .map_err(|e| TaskError::ModelLoad(format!("audio_encoder.norm.weight: {e}")))?;
        let norm_b = vb_enc
            .get(cfg.hidden_size, "norm.bias")
            .map_err(|e| TaskError::ModelLoad(format!("audio_encoder.norm.bias: {e}")))?;

        Ok(Self {
            batch_norm,
            patch_embed,
            stages,
            final_norm: FinalLayerNorm {
                weight: norm_w,
                bias: norm_b,
                eps: 1e-5,
            },
            spec_size: cfg.spec_size,
            num_mel_bins: cfg.num_mel_bins,
        })
    }

    /// Forward: mel spectrogram `[B, 1, time, freq]` → `[B, hidden_size]`
    ///
    /// Where `time` is the natural STFT frame count (e.g. ~998 for 10s at 48kHz),
    /// `freq = num_mel_bins` (64).
    ///
    /// Internally: BatchNorm → reshape_mel2img → PatchEmbed → Swin stages → pool.
    pub fn forward(&self, mel: &Tensor) -> TaskResult<Tensor> {
        let (b, _c, time, freq) = mel
            .dims4()
            .map_err(|e| TaskError::Inference(format!("mel dims: {e}")))?;

        // 1. BatchNorm: transpose to [B, freq, time, 1], apply BN on freq dim, transpose back
        let x = mel
            .permute((0, 3, 2, 1))
            .map_err(|e| TaskError::Inference(format!("permute for BN: {e}")))?;
        // x: [B, freq, time, 1]
        let x = self
            .batch_norm
            .forward(&x)
            .map_err(|e| TaskError::Inference(format!("batch_norm: {e}")))?;
        // Back to [B, 1, time, freq]
        let x = x
            .permute((0, 3, 2, 1))
            .map_err(|e| TaskError::Inference(format!("permute after BN: {e}")))?;

        // 2. reshape_mel2img: convert mel to square image
        //    freq_ratio = spec_size / num_mel_bins (typically 256 / 64 = 4)
        //    Target: spec_width = spec_size * freq_ratio = 1024 (time frames)
        //           spec_height = spec_size / freq_ratio = 64 (freq bins)
        let freq_ratio = self.spec_size / self.num_mel_bins;
        let spec_width = self.spec_size * freq_ratio; // target time frames (1024)

        // Pad/interpolate time dimension to spec_width if needed.
        // We use simple zero-padding or truncation (bicubic interpolation is complex).
        let x = match time.cmp(&spec_width) {
            std::cmp::Ordering::Less => {
                // Pad time dimension with zeros on the right
                let pad_size = spec_width - time;
                let padding = Tensor::zeros((b, 1, pad_size, freq), x.dtype(), x.device())
                    .map_err(|e| TaskError::Inference(format!("create padding: {e}")))?;
                Tensor::cat(&[&x, &padding], 2)
                    .map_err(|e| TaskError::Inference(format!("pad time: {e}")))?
            }
            std::cmp::Ordering::Greater => {
                // Truncate
                x.narrow(2, 0, spec_width)
                    .map_err(|e| TaskError::Inference(format!("truncate time: {e}")))?
            }
            std::cmp::Ordering::Equal => x,
        };
        // Now x is [B, 1, spec_width, freq] = [B, 1, 1024, 64]

        // Reshape to image: [B, C, time, freq] → [B, C*freq_ratio, time/freq_ratio, freq]
        let x = x
            .reshape((b, freq_ratio, spec_width / freq_ratio, freq))
            .map_err(|e| TaskError::Inference(format!("reshape1: {e}")))?;
        // x: [B, freq_ratio, spec_size, freq] = [B, 4, 256, 64]

        // → [B, freq_ratio, freq, spec_size]
        let x = x
            .permute((0, 1, 3, 2))
            .and_then(|t| t.contiguous())
            .map_err(|e| TaskError::Inference(format!("permute mel2img: {e}")))?;
        // x: [B, 4, 64, 256]

        // → [B, 1, freq*freq_ratio, spec_size] = [B, 1, 256, 256]
        let x = x
            .reshape((b, 1, freq * freq_ratio, spec_width / freq_ratio))
            .map_err(|e| TaskError::Inference(format!("reshape mel2img final: {e}")))?;
        // x: [B, 1, 256, 256] — square image for Swin Transformer

        // 3. Patch embedding (global path only, no fusion during inference)
        let x = self
            .patch_embed
            .forward(&x)
            .map_err(|e| TaskError::Inference(format!("patch_embed: {e}")))?;
        // x: [B, num_patches, embed_dim]

        // 4. Swin Transformer stages
        let (mut h, mut w) = (
            self.spec_size / self.patch_embed.patch_stride.0,
            self.spec_size / self.patch_embed.patch_stride.1,
        );
        let mut x = x;
        for (i, stage) in self.stages.iter().enumerate() {
            let (x_new, h_new, w_new) = stage
                .forward(&x, h, w)
                .map_err(|e| TaskError::Inference(format!("swin stage {i}: {e}")))?;
            x = x_new;
            h = h_new;
            w = w_new;
        }

        // 5. Final LayerNorm: x is [B, H*W, hidden_size]
        let x = layer_norm_simple(
            &x,
            &self.final_norm.weight,
            &self.final_norm.bias,
            self.final_norm.eps,
        )
        .map_err(|e| TaskError::Inference(format!("final norm: {e}")))?;

        // 6. Adaptive average pool: [B, H*W, hidden_size] → mean over spatial → [B, hidden_size]
        let x = x
            .mean(1)
            .map_err(|e| TaskError::Inference(format!("avg pool: {e}")))?;

        Ok(x)
    }
}
