//! BigVGAN-style decoder blocks for the speech tokenizer.
//!
//! Each block:
//!   SnakeBeta → ConvTranspose1d (upsample by rate) → 3× ResidualUnit (dilations 1, 3, 9)
//!
//! ResidualUnit:
//!   SnakeBeta → dilated CausalConv1d(k=7) → SnakeBeta → CausalConv1d(k=1) + skip
//!
//! Weight prefix: `decoder.decoder.{i}` where i ∈ {1,2,3,4}
//!   - `.block.0.{alpha,beta}` → SnakeBeta
//!   - `.block.1.conv.{weight,bias}` → ConvTranspose1d
//!   - `.block.{2,3,4}` → ResidualUnits

use candle_core::Result;
use candle_core::Tensor;
use candle_nn::VarBuilder;

use super::conv::{CausalConv1d, CausalConvTranspose1d, SnakeBeta};

// ---------------------------------------------------------------------------
// ResidualUnit
// ---------------------------------------------------------------------------

/// Residual unit: `SnakeBeta → dilated conv(k=7) → SnakeBeta → conv(k=1) + residual`.
struct ResidualUnit {
    act1: SnakeBeta,
    conv1: CausalConv1d,
    act2: SnakeBeta,
    conv2: CausalConv1d,
}

impl ResidualUnit {
    fn new(channels: usize, dilation: usize, vb: VarBuilder) -> Result<Self> {
        let act1 = SnakeBeta::new(channels, vb.pp("act1"))?;
        // Dilated causal conv: [channels, channels, 7], dilation, groups=1
        let conv1 = CausalConv1d::new(
            channels,
            channels,
            7,
            1,
            dilation,
            1,
            vb.pp("conv1").pp("conv"),
        )?;
        let act2 = SnakeBeta::new(channels, vb.pp("act2"))?;
        // 1×1 conv
        let conv2 = CausalConv1d::new(channels, channels, 1, 1, 1, 1, vb.pp("conv2").pp("conv"))?;
        Ok(Self {
            act1,
            conv1,
            act2,
            conv2,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.act1.forward(x)?;
        let h = self.conv1.forward(&h)?;
        let h = self.act2.forward(&h)?;
        let h = self.conv2.forward(&h)?;
        x + h
    }
}

// ---------------------------------------------------------------------------
// DecoderBlock
// ---------------------------------------------------------------------------

/// One decoder block: SnakeBeta → ConvTranspose1d(upsample) → 3× ResidualUnit.
///
/// Rates: [8, 5, 4, 3].  Channel progression: 1536→768→384→192→96.
pub(crate) struct DecoderBlock {
    snake: SnakeBeta,
    upsample: CausalConvTranspose1d,
    res_units: Vec<ResidualUnit>,
}

impl DecoderBlock {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        rate: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // Weight prefix: `block.0` → SnakeBeta, `block.1` → ConvTranspose1d, `block.{2,3,4}` → ResidualUnits
        let snake = SnakeBeta::new(in_channels, vb.pp("block").pp("0"))?;

        let kernel_size = rate * 2;
        let upsample = CausalConvTranspose1d::new_strided(
            in_channels,
            out_channels,
            kernel_size,
            rate,
            vb.pp("block").pp("1").pp("conv"),
        )?;

        let dilations = [1, 3, 9];
        let mut res_units = Vec::with_capacity(3);
        for (i, &dil) in dilations.iter().enumerate() {
            res_units.push(ResidualUnit::new(
                out_channels,
                dil,
                vb.pp("block").pp(i + 2),
            )?);
        }

        Ok(Self {
            snake,
            upsample,
            res_units,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.snake.forward(x)?;
        let mut h = self.upsample.forward_strided(&h)?;
        for ru in &self.res_units {
            h = ru.forward(&h)?;
        }
        Ok(h)
    }
}
