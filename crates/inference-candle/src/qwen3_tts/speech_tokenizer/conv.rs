//! Low-level 1D convolution primitives for the speech tokenizer decoder.
//!
//! - `CausalConv1d`: left-padded causal conv (optionally dilated + grouped).
//! - `CausalConvTranspose1d`: transposed conv with right-trim for causal alignment.
//! - `SnakeBeta`: activation `x + (1/exp(β)) * sin²(exp(α) * x)`.

use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

// ---------------------------------------------------------------------------
// SnakeBeta activation
// ---------------------------------------------------------------------------

/// SnakeBeta activation: `x + (1/exp(β)) * sin²(exp(α) * x)`.
///
/// `alpha` and `beta` are learnable per-channel parameters stored as 1-D
/// tensors of shape `[channels]`.  We exponentiate them before use (following
/// the reference implementation).
pub(crate) struct SnakeBeta {
    alpha: Tensor,
    beta: Tensor,
}

impl SnakeBeta {
    /// Load from a `VarBuilder` that contains `alpha` and `beta` tensors.
    pub fn new(channels: usize, vb: VarBuilder) -> Result<Self> {
        let alpha = vb.get(channels, "alpha")?;
        let beta = vb.get(channels, "beta")?;
        Ok(Self { alpha, beta })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: [B, C, T]
        // Reshape params: [C] → [1, C, 1]
        let ndim = self.alpha.dims().len();
        let alpha = if ndim == 1 {
            self.alpha.exp()?.unsqueeze(0)?.unsqueeze(2)?
        } else {
            self.alpha.exp()?
        };
        let beta = if ndim == 1 {
            self.beta.exp()?.unsqueeze(0)?.unsqueeze(2)?
        } else {
            self.beta.exp()?
        };

        let ax = x.broadcast_mul(&alpha)?;
        let sin_ax = ax.sin()?;
        let sin2 = (&sin_ax * &sin_ax)?;
        let inv_beta = (beta + 1e-9)?.recip()?;
        let scaled = sin2.broadcast_mul(&inv_beta)?;
        x + scaled
    }
}

// ---------------------------------------------------------------------------
// CausalConv1d
// ---------------------------------------------------------------------------

/// Causal 1D convolution with left-padding.
///
/// Supports dilation and depthwise (grouped) convolution.
pub(crate) struct CausalConv1d {
    weight: Tensor,
    bias: Option<Tensor>,
    kernel_size: usize,
    stride: usize,
    dilation: usize,
    groups: usize,
}

impl CausalConv1d {
    /// Build from an explicit `VarBuilder` that sits at the conv's prefix.
    ///
    /// Weight key: `conv.weight` or `weight` (we try `conv.weight` first
    /// because the decoder blocks nest convs inside a `conv` sub-prefix).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        dilation: usize,
        groups: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels / groups, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();
        Ok(Self {
            weight,
            bias,
            kernel_size,
            stride,
            dilation,
            groups,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let effective_k = self.dilation * (self.kernel_size - 1) + 1;
        let pad_left = effective_k - self.stride;
        let x = if pad_left > 0 {
            x.pad_with_zeros(2, pad_left, 0)?
        } else {
            x.clone()
        };
        let mut out = x.conv1d(
            &self.weight,
            0, // padding (we already padded manually)
            self.stride,
            self.dilation,
            self.groups,
        )?;
        if let Some(bias) = &self.bias {
            let b = bias.reshape((1, (), 1))?;
            out = out.broadcast_add(&b)?;
        }
        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// CausalConvTranspose1d
// ---------------------------------------------------------------------------

/// Causal transposed 1D convolution (upsampling).
///
/// After the transposed conv we trim `kernel_size - stride` samples from the
/// right to keep exact causal alignment.
pub(crate) struct CausalConvTranspose1d {
    weight: Tensor,
    bias: Option<Tensor>,
    stride: usize,
    trim_right: usize,
}

impl CausalConvTranspose1d {
    pub fn new_strided(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // PyTorch ConvTranspose1d weight shape: [in_channels, out_channels, kernel_size]
        let weight = vb.get((in_channels, out_channels, kernel_size), "weight")?;
        let bias = vb.get(out_channels, "bias").ok();
        let trim_right = kernel_size - stride;
        Ok(Self {
            weight,
            bias,
            stride,
            trim_right,
        })
    }

    /// Forward pass: transposed conv with stride, then trim right for causal alignment.
    pub fn forward_strided(&self, x: &Tensor) -> Result<Tensor> {
        // conv_transpose1d(kernel, padding, output_padding, stride, dilation, groups)
        let mut out = x.conv_transpose1d(&self.weight, 0, 0, self.stride, 1, 1)?;
        if let Some(bias) = &self.bias {
            let b = bias.reshape((1, (), 1))?;
            out = out.broadcast_add(&b)?;
        }
        if self.trim_right > 0 {
            let t = out.dim(2)?;
            let end = t.saturating_sub(self.trim_right);
            out = out.narrow(2, 0, end)?;
        }
        Ok(out)
    }
}
