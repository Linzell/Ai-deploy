//! Speech tokenizer — neural audio codec decoder (discrete codes → waveform).
//!
//! Converts 16-codebook discrete tokens into audio waveform at 24kHz.
//!
//! Architecture (`Qwen3TTSTokenizerV2Model`):
//!   1. RVQ dequantize: `rvq_first` (CB0) + `rvq_rest` (CB1-CB15) → [B, 512, T]
//!   2. Pre-conv: Conv1d [512 → 1024, k=3]
//!   3. Pre-transformer: input_proj(1024→512) → 8 transformer layers → norm → output_proj(512→1024)
//!   4. ConvNeXt upsample: 2 stages (each ×2) → total ×4
//!   5. Decoder init conv: Conv1d [1024 → 1536, k=7]
//!   6. Decoder blocks: 4 blocks (rates [8,5,4,3]) → total ×480
//!   7. Final SnakeBeta + Conv1d → 1 channel
//!   8. Clamp to [-1, 1]
//!
//! Total upsample: 4 × 480 = 1920 samples/frame.  At 24kHz ⇒ 12.5 Hz frame rate.
//!
//! The speech tokenizer has its OWN weights in `speech_tokenizer/model.safetensors`
//! and config in `speech_tokenizer/config.json`, separate from the talker model.

#[allow(clippy::needless_pass_by_value)]
mod conv;
#[allow(clippy::needless_pass_by_value)]
mod convnext;
#[allow(clippy::needless_pass_by_value)]
mod decoder_block;
#[allow(clippy::needless_pass_by_value)]
mod quantizer;
#[allow(clippy::needless_pass_by_value)]
mod transformer;

use candle_core::{DType, Result, Tensor};
use candle_nn::VarBuilder;
use serde::Deserialize;

use conv::{CausalConv1d, SnakeBeta};
use convnext::UpsampleStage;
use decoder_block::DecoderBlock;
use quantizer::Quantizer;
use transformer::PreTransformer;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Top-level speech tokenizer config (from `speech_tokenizer/config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct SpeechTokenizerConfig {
    #[serde(default = "default_sample_rate")]
    pub input_sample_rate: u32,
    #[serde(default = "default_sample_rate")]
    pub output_sample_rate: u32,
    #[serde(default)]
    pub decoder_config: DecoderConfig,
}

/// Decoder-specific config nested inside the top-level config.
#[derive(Debug, Clone, Deserialize)]
pub struct DecoderConfig {
    #[serde(default = "default_latent_dim")]
    pub latent_dim: usize,
    #[serde(default = "default_codebook_dim")]
    pub codebook_dim: usize,
    #[serde(default = "default_codebook_size")]
    pub codebook_size: usize,
    #[serde(default = "default_decoder_dim")]
    pub decoder_dim: usize,
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_num_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_num_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_quantizers")]
    pub num_quantizers: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_upsampling_ratios")]
    pub upsampling_ratios: Vec<usize>,
    #[serde(default = "default_upsample_rates")]
    pub upsample_rates: Vec<usize>,
}

impl Default for DecoderConfig {
    fn default() -> Self {
        Self {
            latent_dim: 1024,
            codebook_dim: 512,
            codebook_size: 2048,
            decoder_dim: 1536,
            hidden_size: 512,
            intermediate_size: 1024,
            num_attention_heads: 16,
            head_dim: 64,
            num_hidden_layers: 8,
            num_quantizers: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            max_position_embeddings: 8000,
            upsampling_ratios: vec![2, 2],
            upsample_rates: vec![8, 5, 4, 3],
        }
    }
}

fn default_sample_rate() -> u32 {
    24000
}
fn default_latent_dim() -> usize {
    1024
}
fn default_codebook_dim() -> usize {
    512
}
fn default_codebook_size() -> usize {
    2048
}
fn default_decoder_dim() -> usize {
    1536
}
fn default_hidden_size() -> usize {
    512
}
fn default_intermediate_size() -> usize {
    1024
}
fn default_num_heads() -> usize {
    16
}
fn default_head_dim() -> usize {
    64
}
fn default_num_layers() -> usize {
    8
}
fn default_num_quantizers() -> usize {
    16
}
fn default_rms_norm_eps() -> f64 {
    1e-5
}
fn default_rope_theta() -> f64 {
    10000.0
}
fn default_max_position() -> usize {
    8000
}
fn default_upsampling_ratios() -> Vec<usize> {
    vec![2, 2]
}
fn default_upsample_rates() -> Vec<usize> {
    vec![8, 5, 4, 3]
}

// ---------------------------------------------------------------------------
// SpeechTokenizer
// ---------------------------------------------------------------------------

/// Full speech tokenizer decoder.
///
/// Takes 16-codebook discrete tokens and produces a 24kHz mono audio waveform.
pub struct SpeechTokenizer {
    quantizer: Quantizer,
    pre_conv: CausalConv1d,
    pre_transformer: PreTransformer,
    upsample_stages: Vec<UpsampleStage>,
    decoder_init_conv: CausalConv1d,
    decoder_blocks: Vec<DecoderBlock>,
    final_snake: SnakeBeta,
    final_conv: CausalConv1d,
    /// Audio sample rate (24000).
    pub sample_rate: u32,
}

impl SpeechTokenizer {
    /// Load from a directory containing `config.json` and `model.safetensors`.
    pub fn from_dir(
        dir: &std::path::Path,
        device: &candle_core::Device,
    ) -> std::result::Result<Self, crate::error::TaskError> {
        use crate::error::TaskError;

        let config_path = dir.join("config.json");
        let cfg: SpeechTokenizerConfig = if config_path.exists() {
            let f = std::fs::File::open(&config_path)
                .map_err(|e| TaskError::ModelLoad(format!("speech_tokenizer config: {e}")))?;
            serde_json::from_reader(f)
                .map_err(|e| TaskError::ModelLoad(format!("speech_tokenizer config parse: {e}")))?
        } else {
            tracing::warn!("No speech_tokenizer/config.json found, using defaults");
            SpeechTokenizerConfig {
                input_sample_rate: 24000,
                output_sample_rate: 24000,
                decoder_config: DecoderConfig::default(),
            }
        };

        let weights_path = dir.join("model.safetensors");
        if !weights_path.exists() {
            return Err(TaskError::ModelLoad(
                "speech_tokenizer/model.safetensors not found".into(),
            ));
        }

        let vb = crate::utils::load_safetensors_safe(&[weights_path], DType::F32, device)?;
        Self::new(&cfg, device, vb)
            .map_err(|e| TaskError::ModelLoad(format!("speech_tokenizer: {e}")))
    }

    fn new(
        cfg: &SpeechTokenizerConfig,
        device: &candle_core::Device,
        vb: VarBuilder,
    ) -> Result<Self> {
        let dc = &cfg.decoder_config;
        let vb_dec = vb.pp("decoder");

        // 1. Quantizer (RVQ dequantize)
        // VQ internal dim (256) = codebook_dim(512) / 2, not directly in config.
        let vq_dim = dc.codebook_dim / 2;
        let quantizer = Quantizer::new(
            dc.codebook_size, // 2048 — number of codebook entries
            vq_dim,           // 256  — VQ embedding dimension
            dc.codebook_dim,  // 512  — projection output dimension
            vb_dec.pp("quantizer"),
        )?;

        // 2. Pre-conv: Conv1d [codebook_dim(512) → latent_dim(1024), k=3]
        let pre_conv = CausalConv1d::new(
            dc.codebook_dim,
            dc.latent_dim,
            3,
            1,
            1,
            1,
            vb_dec.pp("pre_conv").pp("conv"),
        )?;

        // 3. Pre-transformer
        let pre_transformer = PreTransformer::new(
            dc.hidden_size,
            dc.num_hidden_layers,
            dc.num_attention_heads,
            dc.head_dim,
            dc.intermediate_size,
            dc.latent_dim,
            dc.rms_norm_eps,
            dc.rope_theta,
            dc.max_position_embeddings,
            device,
            vb_dec.pp("pre_transformer"),
        )?;

        // 4. ConvNeXt upsample stages (2 stages, each ×2, channels stay at latent_dim=1024)
        let mut upsample_stages = Vec::with_capacity(dc.upsampling_ratios.len());
        for (i, &ratio) in dc.upsampling_ratios.iter().enumerate() {
            upsample_stages.push(UpsampleStage::new(
                dc.latent_dim,
                ratio,
                vb_dec.pp("upsample").pp(i),
            )?);
        }

        // 5. Decoder init conv: Conv1d [latent_dim(1024) → decoder_dim(1536), k=7]
        let decoder_init_conv = CausalConv1d::new(
            dc.latent_dim,
            dc.decoder_dim,
            7,
            1,
            1,
            1,
            vb_dec.pp("decoder").pp("0").pp("conv"),
        )?;

        // 6. Decoder blocks (rates [8,5,4,3])
        // Channel progression: 1536 → 768 → 384 → 192 → 96
        let rates = &dc.upsample_rates;
        let mut channels = dc.decoder_dim; // 1536
        let mut decoder_blocks = Vec::with_capacity(rates.len());
        for (i, &rate) in rates.iter().enumerate() {
            let out_channels = channels / 2;
            decoder_blocks.push(DecoderBlock::new(
                channels,
                out_channels,
                rate,
                vb_dec.pp("decoder").pp(i + 1),
            )?);
            channels = out_channels;
        }
        // After 4 blocks: channels = 96

        // 7. Final SnakeBeta + Conv1d → 1 channel
        let final_snake = SnakeBeta::new(channels, vb_dec.pp("decoder").pp("5"))?;
        let final_conv = CausalConv1d::new(
            channels,
            1,
            7,
            1,
            1,
            1,
            vb_dec.pp("decoder").pp("6").pp("conv"),
        )?;

        Ok(Self {
            quantizer,
            pre_conv,
            pre_transformer,
            upsample_stages,
            decoder_init_conv,
            decoder_blocks,
            final_snake,
            final_conv,
            sample_rate: cfg.output_sample_rate,
        })
    }

    /// Decode discrete codebook tokens to audio waveform.
    ///
    /// `codes` shape: `[num_codebooks(16), num_frames]` — each value is a codebook token ID.
    ///
    /// Returns: `[num_samples]` — mono audio waveform in [-1, 1].
    pub fn decode(&self, codes: &Tensor) -> Result<Tensor> {
        // 1. RVQ dequantize → [1, 512, T]
        let x = self.quantizer.decode(codes)?;

        // 2. Pre-conv → [1, 1024, T]
        let x = self.pre_conv.forward(&x)?;

        // 3. Pre-transformer → [1, 1024, T]
        let x = self.pre_transformer.forward(&x)?;

        // 4. ConvNeXt upsample (×2, ×2) → [1, 1024, T×4]
        let mut x = x;
        for stage in &self.upsample_stages {
            x = stage.forward(&x)?;
        }

        // 5. Decoder init conv → [1, 1536, T×4]
        x = self.decoder_init_conv.forward(&x)?;

        // 6. Decoder blocks (×8, ×5, ×4, ×3) → [1, 96, T×1920]
        for block in &self.decoder_blocks {
            x = block.forward(&x)?;
        }

        // 7. Final SnakeBeta + conv → [1, 1, T×1920]
        x = self.final_snake.forward(&x)?;
        x = self.final_conv.forward(&x)?;

        // 8. Squeeze to [T×1920] and clamp to [-1, 1]
        let x = x.squeeze(0)?.squeeze(0)?;
        x.clamp(-1.0_f32, 1.0_f32)
    }
}
