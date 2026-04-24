//! ResNet backbone for DETR.
//!
//! Implements ResNet-18 and ResNet-50 using frozen BatchNorm
//! (weights from HuggingFace's DETR / Table-Transformer safetensors format).
//!
//! Weight key prefix: `model.backbone.conv_encoder.model.`
//!
//! Actual weight layout (torchvision-style):
//! - `conv1.weight` — initial 7x7 conv
//! - `bn1.{weight,bias,running_mean,running_var}` — initial BN
//! - `layer{1..4}.{j}.conv{1,2[,3]}.weight` — block convolutions
//! - `layer{1..4}.{j}.bn{1,2[,3]}.{weight,bias,running_mean,running_var}` — block BN
//! - `layer{1..4}.{j}.downsample.0.weight` — shortcut 1x1 conv
//! - `layer{1..4}.{j}.downsample.1.{weight,bias,running_mean,running_var}` — shortcut BN

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

/// Frozen batch normalization (inference-only).
///
/// Uses pre-computed running_mean, running_var, weight, bias.
/// Forward: `x * scale + bias` where `scale = weight / sqrt(running_var + eps)`.
pub struct FrozenBatchNorm2d {
    scale: Tensor, // [1, C, 1, 1]
    bias: Tensor,  // [1, C, 1, 1]
}

impl FrozenBatchNorm2d {
    pub fn load(vb: &VarBuilder, num_features: usize) -> Result<Self> {
        let weight = vb.get(num_features, "weight")?;
        let bias = vb.get(num_features, "bias")?;
        let running_mean = vb.get(num_features, "running_mean")?;
        let running_var = vb.get(num_features, "running_var")?;

        let eps = 1e-5_f64;
        let eps_t = Tensor::new(&[eps as f32], running_var.device())?
            .to_dtype(running_var.dtype())?
            .broadcast_as(running_var.shape())?;
        let std = running_var.broadcast_add(&eps_t)?.sqrt()?;
        let scale = weight.broadcast_div(&std)?;
        let bias = bias.broadcast_sub(&running_mean.broadcast_mul(&scale)?)?;

        // Reshape to [1, C, 1, 1] for broadcasting with [N, C, H, W]
        let c = num_features;
        let scale = scale.reshape((1, c, 1, 1))?;
        let bias = bias.reshape((1, c, 1, 1))?;

        Ok(Self { scale, bias })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.broadcast_mul(&self.scale)?.broadcast_add(&self.bias)
    }
}

/// A single convolution layer (no bias — BN handles bias).
pub struct Conv2d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    padding: usize,
}

impl Conv2d {
    pub fn load(vb: &VarBuilder, stride: usize, padding: usize) -> Result<Self> {
        // Weight shape inferred from the stored tensor
        let weight = vb.get_unchecked("weight")?;
        Ok(Self {
            weight,
            bias: None,
            stride,
            padding,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = x.conv2d(&self.weight, self.padding, self.stride, 1, 1)?;
        match &self.bias {
            Some(b) => out.broadcast_add(b),
            None => Ok(out),
        }
    }

    /// Load with known dimensions (weight only, no bias).
    pub fn load_with_dims(
        vb: &VarBuilder,
        out_c: usize,
        in_c: usize,
        ksize: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let weight = vb.get((out_c, in_c, ksize, ksize), "weight")?;
        Ok(Self {
            weight,
            bias: None,
            stride,
            padding,
        })
    }

    /// Load with known dimensions and a bias term.
    pub fn load_with_bias(
        vb: &VarBuilder,
        out_c: usize,
        in_c: usize,
        ksize: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let weight = vb.get((out_c, in_c, ksize, ksize), "weight")?;
        let bias = vb.get(out_c, "bias")?.reshape((1, out_c, 1, 1))?;
        Ok(Self {
            weight,
            bias: Some(bias),
            stride,
            padding,
        })
    }
}

/// Conv + Frozen BatchNorm block.
///
/// Loads from separate VarBuilder prefixes for conv and bn.
struct ConvBn {
    conv: Conv2d,
    bn: FrozenBatchNorm2d,
}

impl ConvBn {
    /// Load from `{conv_prefix}.weight` and `{bn_prefix}.{weight,bias,...}`.
    fn load(
        conv_vb: &VarBuilder,
        bn_vb: &VarBuilder,
        out_channels: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let conv = Conv2d::load(conv_vb, stride, padding)?;
        let bn = FrozenBatchNorm2d::load(bn_vb, out_channels)?;
        Ok(Self { conv, bn })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        self.bn.forward(&x)
    }
}

/// A basic block for ResNet-18/34.
///
/// Torchvision keys: `conv1`, `bn1`, `conv2`, `bn2`, `downsample.{0,1}`
struct BasicBlock {
    conv_bn1: ConvBn, // 3x3
    conv_bn2: ConvBn, // 3x3
    downsample: Option<ConvBn>,
}

impl BasicBlock {
    fn load(
        vb: &VarBuilder,
        in_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let conv_bn1 = ConvBn::load(&vb.pp("conv1"), &vb.pp("bn1"), out_channels, stride, 1)?;
        let conv_bn2 = ConvBn::load(&vb.pp("conv2"), &vb.pp("bn2"), out_channels, 1, 1)?;

        let downsample = if in_channels != out_channels || stride != 1 {
            Some(ConvBn::load(
                &vb.pp("downsample").pp("0"),
                &vb.pp("downsample").pp("1"),
                out_channels,
                stride,
                0,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv_bn1,
            conv_bn2,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(ref ds) = self.downsample {
            ds.forward(x)?
        } else {
            x.clone()
        };

        let x = self.conv_bn1.forward(x)?;
        let x = x.relu()?;
        let x = self.conv_bn2.forward(&x)?;

        let x = x.broadcast_add(&residual)?;
        x.relu()
    }
}

/// A bottleneck block for ResNet-50/101.
///
/// Three convs: 1x1 (reduce) -> 3x3 -> 1x1 (expand).
/// Torchvision keys: `conv1`, `bn1`, `conv2`, `bn2`, `conv3`, `bn3`, `downsample.{0,1}`
struct Bottleneck {
    conv_bn1: ConvBn, // 1x1 reduce
    conv_bn2: ConvBn, // 3x3
    conv_bn3: ConvBn, // 1x1 expand
    downsample: Option<ConvBn>,
}

impl Bottleneck {
    fn load(
        vb: &VarBuilder,
        in_channels: usize,
        mid_channels: usize,
        out_channels: usize,
        stride: usize,
    ) -> Result<Self> {
        let conv_bn1 = ConvBn::load(&vb.pp("conv1"), &vb.pp("bn1"), mid_channels, 1, 0)?;
        let conv_bn2 = ConvBn::load(&vb.pp("conv2"), &vb.pp("bn2"), mid_channels, stride, 1)?;
        let conv_bn3 = ConvBn::load(&vb.pp("conv3"), &vb.pp("bn3"), out_channels, 1, 0)?;

        let downsample = if in_channels != out_channels || stride != 1 {
            Some(ConvBn::load(
                &vb.pp("downsample").pp("0"),
                &vb.pp("downsample").pp("1"),
                out_channels,
                stride,
                0,
            )?)
        } else {
            None
        };

        Ok(Self {
            conv_bn1,
            conv_bn2,
            conv_bn3,
            downsample,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = if let Some(ref ds) = self.downsample {
            ds.forward(x)?
        } else {
            x.clone()
        };

        let x = self.conv_bn1.forward(x)?;
        let x = x.relu()?;
        let x = self.conv_bn2.forward(&x)?;
        let x = x.relu()?;
        let x = self.conv_bn3.forward(&x)?;

        let x = x.broadcast_add(&residual)?;
        x.relu()
    }
}

/// A stage is a sequence of blocks with the same output channel count.
enum ResNetBlock {
    Basic(BasicBlock),
    Bottleneck(Bottleneck),
}

impl ResNetBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        match self {
            Self::Basic(b) => b.forward(x),
            Self::Bottleneck(b) => b.forward(x),
        }
    }
}

struct ResNetStage {
    blocks: Vec<ResNetBlock>,
}

impl ResNetStage {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

/// Full ResNet backbone (inference-only, frozen BN).
///
/// Returns the output of the last stage (layer4 in torchvision naming).
/// For DETR, only the last feature map is used.
pub struct ResNetBackbone {
    // Initial conv7x7 + BN + ReLU + MaxPool
    embedder: ConvBn,
    // 4 stages (layer1..layer4)
    stages: [ResNetStage; 4],
    /// Output channel count of the last stage
    pub out_channels: usize,
}

impl ResNetBackbone {
    /// Load a ResNet backbone from VarBuilder.
    ///
    /// `vb` should be prefixed to `model.backbone.conv_encoder.model.` already.
    /// `is_resnet18` selects between BasicBlock and Bottleneck.
    pub fn load(vb: &VarBuilder, is_resnet18: bool) -> Result<Self> {
        // Initial: conv1 + bn1
        let embedder = ConvBn::load(&vb.pp("conv1"), &vb.pp("bn1"), 64, 2, 3)?;

        let stages = if is_resnet18 {
            Self::load_basic_stages(vb)?
        } else {
            Self::load_bottleneck_stages(vb)?
        };

        let out_channels = if is_resnet18 { 512 } else { 2048 };

        Ok(Self {
            embedder,
            stages,
            out_channels,
        })
    }

    /// Load ResNet-18/34 stages (BasicBlock).
    /// Layers per stage: [2, 2, 2, 2] for ResNet-18.
    fn load_basic_stages(vb: &VarBuilder) -> Result<[ResNetStage; 4]> {
        let stage_configs: [(usize, usize, usize, usize); 4] = [
            // (num_blocks, in_channels, out_channels, stride)
            (2, 64, 64, 1),
            (2, 64, 128, 2),
            (2, 128, 256, 2),
            (2, 256, 512, 2),
        ];

        let mut stages_arr: [ResNetStage; 4] = [
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
        ];

        for (i, (num_blocks, in_c, out_c, stride)) in stage_configs.iter().enumerate() {
            // torchvision: layer1, layer2, layer3, layer4
            let stage_vb = vb.pp(format!("layer{}", i + 1));
            let mut blocks = Vec::with_capacity(*num_blocks);
            for j in 0..*num_blocks {
                let block_vb = stage_vb.pp(j.to_string());
                let (block_in, block_stride) = if j == 0 {
                    (*in_c, *stride)
                } else {
                    (*out_c, 1)
                };
                blocks.push(ResNetBlock::Basic(BasicBlock::load(
                    &block_vb,
                    block_in,
                    *out_c,
                    block_stride,
                )?));
            }
            stages_arr[i] = ResNetStage { blocks };
        }

        Ok(stages_arr)
    }

    /// Load ResNet-50/101 stages (Bottleneck).
    /// Layers per stage: [3, 4, 6, 3] for ResNet-50.
    fn load_bottleneck_stages(vb: &VarBuilder) -> Result<[ResNetStage; 4]> {
        let stage_configs: [(usize, usize, usize, usize, usize); 4] = [
            // (num_blocks, in_channels, mid_channels, out_channels, stride)
            (3, 64, 64, 256, 1),
            (4, 256, 128, 512, 2),
            (6, 512, 256, 1024, 2),
            (3, 1024, 512, 2048, 2),
        ];

        let mut stages_arr: [ResNetStage; 4] = [
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
            ResNetStage { blocks: Vec::new() },
        ];

        for (i, (num_blocks, in_c, mid_c, out_c, stride)) in stage_configs.iter().enumerate() {
            let stage_vb = vb.pp(format!("layer{}", i + 1));
            let mut blocks = Vec::with_capacity(*num_blocks);
            for j in 0..*num_blocks {
                let block_vb = stage_vb.pp(j.to_string());
                let (block_in, block_stride) = if j == 0 {
                    (*in_c, *stride)
                } else {
                    (*out_c, 1)
                };
                blocks.push(ResNetBlock::Bottleneck(Bottleneck::load(
                    &block_vb,
                    block_in,
                    *mid_c,
                    *out_c,
                    block_stride,
                )?));
            }
            stages_arr[i] = ResNetStage { blocks };
        }

        Ok(stages_arr)
    }

    /// Forward pass: pixel_values [N, 3, H, W] -> feature_map [N, C, H/32, W/32].
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        // Initial conv7x7 + BN + ReLU
        let x = self.embedder.forward(pixel_values)?;
        let x = x.relu()?;
        // MaxPool 3x3 stride 2 pad 1
        // Candle's max_pool2d has no padding param, so we pad manually
        let x = x.pad_with_zeros(2, 1, 1)?; // pad H
        let x = x.pad_with_zeros(3, 1, 1)?; // pad W
        let x = x.max_pool2d_with_stride(3, 2)?;
        // Stages (layer1..layer4)
        let x = self.stages[0].forward(&x)?;
        let x = self.stages[1].forward(&x)?;
        let x = self.stages[2].forward(&x)?;
        self.stages[3].forward(&x)
    }
}
