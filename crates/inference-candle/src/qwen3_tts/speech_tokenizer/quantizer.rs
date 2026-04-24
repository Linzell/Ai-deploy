//! RVQ (Residual Vector Quantization) dequantization.
//!
//! The speech tokenizer stores codebooks as `embedding_sum` / `cluster_usage`
//! (EMA-updated).  At inference we normalize: `codebook = embedding_sum / max(cluster_usage, ε)`.
//!
//! Architecture:
//!   - `rvq_first`: 1 codebook (semantic, CB0).  input_proj [256←512, 1] → VQ → output_proj [512←256, 1]
//!   - `rvq_rest`:  15 codebooks (acoustic, CB1-CB15).  Same proj dims, codes summed before proj.
//!   - Final: `quantized = rvq_first_out + rvq_rest_out`  →  shape [B, 512, T]

use candle_core::{DType, IndexOp, Result, Tensor};
use candle_nn::VarBuilder;

/// A single normalized codebook: `[codebook_size, codebook_dim]`.
struct Codebook {
    /// Pre-normalized embedding table.
    embedding: Tensor,
}

impl Codebook {
    fn from_vb(codebook_size: usize, codebook_dim: usize, vb: VarBuilder) -> Result<Self> {
        let embedding_sum = vb.get((codebook_size, codebook_dim), "embedding_sum")?;
        let cluster_usage = vb.get(codebook_size, "cluster_usage")?;

        // Normalize: embedding = embedding_sum / max(cluster_usage, ε)
        let eps = 1e-7_f64;
        let usage_clamped = cluster_usage.clamp(eps, f64::MAX)?;
        let embedding = embedding_sum.broadcast_div(&usage_clamped.unsqueeze(1)?)?;
        Ok(Self { embedding })
    }

    /// Look up codes → embeddings.  `codes`: [T] (i64/u32).
    /// Returns [T, codebook_dim].
    fn lookup(&self, codes: &Tensor) -> Result<Tensor> {
        self.embedding.index_select(codes, 0)
    }
}

/// Conv1d 1×1 projection (stored as [out, in, 1]).
struct Conv1x1 {
    weight: Tensor,
}

impl Conv1x1 {
    fn new(out_channels: usize, in_channels: usize, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get((out_channels, in_channels, 1), "weight")?;
        Ok(Self { weight })
    }

    /// x: [B, C_in, T] → [B, C_out, T]
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.conv1d(&self.weight, 0, 1, 1, 1)
    }
}

/// One RVQ branch (rvq_first or rvq_rest).
struct RvqBranch {
    codebooks: Vec<Codebook>,
    #[allow(dead_code)]
    input_proj: Conv1x1,
    output_proj: Conv1x1,
    codebook_dim: usize,
}

impl RvqBranch {
    fn new(
        num_codebooks: usize,
        codebook_size: usize,
        codebook_dim: usize,
        proj_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        // input_proj: [codebook_dim, proj_dim, 1]  (encode direction, unused at decode)
        let input_proj = Conv1x1::new(codebook_dim, proj_dim, vb.pp("input_proj"))?;
        // output_proj: [proj_dim, codebook_dim, 1]
        let output_proj = Conv1x1::new(proj_dim, codebook_dim, vb.pp("output_proj"))?;

        let vb_layers = vb.pp("vq").pp("layers");
        let mut codebooks = Vec::with_capacity(num_codebooks);
        for i in 0..num_codebooks {
            codebooks.push(Codebook::from_vb(
                codebook_size,
                codebook_dim,
                vb_layers.pp(i).pp("_codebook"),
            )?);
        }

        Ok(Self {
            codebooks,
            input_proj,
            output_proj,
            codebook_dim,
        })
    }

    /// Decode codes → projected quantized output.
    ///
    /// `codes`: `[num_codebooks_in_branch, T]` (u32).
    /// Returns: `[1, proj_dim, T]` (proj_dim = 512).
    fn decode(&self, codes: &Tensor, device: &candle_core::Device) -> Result<Tensor> {
        let num_cb = codes.dim(0)?;
        let seq_len = codes.dim(1)?;

        // Sum embeddings from all codebooks in this branch
        let mut summed = Tensor::zeros((seq_len, self.codebook_dim), DType::F32, device)?;

        for i in 0..num_cb {
            let cb_codes = codes.i(i)?; // [T]
            let embedded = self.codebooks[i].lookup(&cb_codes)?; // [T, codebook_dim]
            summed = (summed + embedded)?;
        }

        // [T, codebook_dim] → [1, codebook_dim, T] for conv
        let x = summed.transpose(0, 1)?.unsqueeze(0)?;
        // input_proj not used during decode (it's for encode direction).
        // During decode: just output_proj the summed embeddings.
        self.output_proj.forward(&x)
    }
}

/// Full RVQ dequantizer: `rvq_first` (CB0) + `rvq_rest` (CB1-CB15).
pub(crate) struct Quantizer {
    rvq_first: RvqBranch,
    rvq_rest: RvqBranch,
}

impl Quantizer {
    pub fn new(
        codebook_size: usize,
        codebook_dim: usize,
        proj_dim: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let rvq_first =
            RvqBranch::new(1, codebook_size, codebook_dim, proj_dim, vb.pp("rvq_first"))?;
        let rvq_rest =
            RvqBranch::new(15, codebook_size, codebook_dim, proj_dim, vb.pp("rvq_rest"))?;
        Ok(Self {
            rvq_first,
            rvq_rest,
        })
    }

    /// Decode 16-codebook codes to continuous representation.
    ///
    /// `codes`: `[16, T]` (u32 token IDs).
    /// Returns: `[1, 512, T]` (float).
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        let device = codes.device();

        // CB0 → rvq_first
        let first_codes = codes.i(0..1)?; // [1, T]
        let first_out = self.rvq_first.decode(&first_codes, device)?;

        // CB1-CB15 → rvq_rest
        let rest_codes = codes.i(1..)?; // [15, T]
        let rest_out = self.rvq_rest.decode(&rest_codes, device)?;

        // Sum the two projections
        first_out + rest_out
    }
}
