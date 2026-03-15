//! ConvNeXt-style upsample block.
//!
//! Each block:  ConvTranspose1d(stride=2) → ConvNeXt residual
//!
//! ConvNeXt residual:
//!   depthwise conv (groups=dim, k=7) → LayerNorm → pwconv1 (dim→4*dim) → GELU → pwconv2 (4*dim→dim) → γ scale → residual add
//!
//! Weight prefix: `decoder.upsample.{i}.0` (transposed conv), `decoder.upsample.{i}.1` (convnext)

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;

use super::conv::CausalConvTranspose1d;

/// Compute `x @ weight.T`, collapsing batch dims so candle's matmul works.
fn linear_fwd(x: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let dims = x.dims();
    let in_dim = *dims.last().unwrap();
    let batch: usize = dims.iter().rev().skip(1).product();
    let flat = x.reshape((batch, in_dim))?;
    let wt = weight.t()?;
    let out = flat.matmul(&wt)?;
    let out_dim = out.dim(1)?;
    let mut shape: Vec<usize> = dims[..dims.len() - 1].to_vec();
    shape.push(out_dim);
    out.reshape(shape)
}

// ---------------------------------------------------------------------------
// LayerNorm (channels-last, like PyTorch nn.LayerNorm)
// ---------------------------------------------------------------------------

struct LayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl LayerNorm {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(dim, "weight")?;
        let bias = vb.get(dim, "bias")?;
        Ok(Self {
            weight,
            bias,
            eps: 1e-6,
        })
    }

    /// x: [B, T, C] → normalized over last dim.
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let dtype = x.dtype();
        let x = x.to_dtype(DType::F32)?;
        let mean = x.mean_keepdim(candle_core::D::Minus1)?;
        let x_centered = x.broadcast_sub(&mean)?;
        let var = (&x_centered * &x_centered)?.mean_keepdim(candle_core::D::Minus1)?;
        let x_norm = x_centered.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        x_norm
            .to_dtype(dtype)?
            .broadcast_mul(&self.weight)?
            .broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------------------------
// Linear (with bias, for pointwise convs applied in channel-last format)
// ---------------------------------------------------------------------------

struct Linear {
    weight: Tensor,
    bias: Tensor,
}

impl Linear {
    fn new(in_dim: usize, out_dim: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_dim, in_dim), "weight")?;
        let bias = vb.get(out_dim, "bias")?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let out = linear_fwd(x, &self.weight)?;
        out.broadcast_add(&self.bias)
    }
}

// ---------------------------------------------------------------------------
// ConvNeXt block (depthwise conv → LayerNorm → pwconv1 → GELU → pwconv2 → γ)
// ---------------------------------------------------------------------------

struct ConvNeXtBlock {
    /// Depthwise causal conv1d (groups = dim, kernel = 7).
    dwconv_weight: Tensor,
    dwconv_bias: Tensor,
    dwconv_kernel_size: usize,
    norm: LayerNorm,
    pwconv1: Linear,
    pwconv2: Linear,
    gamma: Tensor,
    dim: usize,
}

impl ConvNeXtBlock {
    fn new(dim: usize, vb: VarBuilder) -> Result<Self> {
        // Depthwise conv: [dim, 1, 7] with groups=dim
        let dwconv_weight = vb.pp("dwconv").pp("conv").get((dim, 1, 7), "weight")?;
        let dwconv_bias = vb.pp("dwconv").pp("conv").get(dim, "bias")?;
        let norm = LayerNorm::new(dim, vb.pp("norm"))?;
        let pwconv1 = Linear::new(dim, dim * 4, vb.pp("pwconv1"))?;
        let pwconv2 = Linear::new(dim * 4, dim, vb.pp("pwconv2"))?;
        let gamma = vb.get(dim, "gamma")?;

        Ok(Self {
            dwconv_weight,
            dwconv_bias,
            dwconv_kernel_size: 7,
            norm,
            pwconv1,
            pwconv2,
            gamma,
            dim,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = x.clone();

        // Depthwise causal conv: pad left by kernel_size - 1
        let pad_left = self.dwconv_kernel_size - 1;
        let x_padded = x.pad_with_zeros(2, pad_left, 0)?;
        let mut h = x_padded.conv1d(&self.dwconv_weight, 0, 1, 1, self.dim)?;
        let bias = self.dwconv_bias.reshape((1, (), 1))?;
        h = h.broadcast_add(&bias)?;

        // [B, C, T] → [B, T, C] for LayerNorm + pointwise
        let h = h.transpose(1, 2)?;
        let h = self.norm.forward(&h)?;
        let h = self.pwconv1.forward(&h)?;
        let h = h.gelu_erf()?;
        let h = self.pwconv2.forward(&h)?;
        // Gamma scale: [C] broadcast over [B, T, C]
        let h = h.broadcast_mul(&self.gamma)?;
        // [B, T, C] → [B, C, T]
        let h = h.transpose(1, 2)?;

        residual + h
    }
}

// ---------------------------------------------------------------------------
// UpsampleStage: ConvTranspose1d(stride=2) + ConvNeXtBlock
// ---------------------------------------------------------------------------

/// One upsample stage: transposed conv (×2) followed by ConvNeXt residual.
///
/// Weight prefix: `decoder.upsample.{i}`
///   - `.0.conv` → ConvTranspose1d
///   - `.1`      → ConvNeXt block
pub(crate) struct UpsampleStage {
    trans_conv: CausalConvTranspose1d,
    convnext: ConvNeXtBlock,
}

impl UpsampleStage {
    pub fn new(dim: usize, stride: usize, vb: VarBuilder) -> Result<Self> {
        // ConvTranspose1d: [dim, dim, stride] — same in/out channels
        let trans_conv =
            CausalConvTranspose1d::new_strided(dim, dim, stride, stride, vb.pp("0").pp("conv"))?;
        let convnext = ConvNeXtBlock::new(dim, vb.pp("1"))?;
        Ok(Self {
            trans_conv,
            convnext,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let h = self.trans_conv.forward_strided(x)?;
        self.convnext.forward(&h)
    }
}
