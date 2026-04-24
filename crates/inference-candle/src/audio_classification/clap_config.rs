//! CLAP (Contrastive Language-Audio Pretraining) configuration.
//!
//! Covers `laion/clap-htsat-fused` and similar CLAP models.
//! CLAP uses a dual-encoder architecture:
//! - Audio encoder: HTS-AT (Swin Transformer variant)
//! - Text encoder: RoBERTa

use serde::Deserialize;

/// Top-level CLAP config.json.
#[derive(Debug, Clone, Deserialize)]
pub struct ClapConfig {
    #[serde(default)]
    pub model_type: Option<String>,

    /// Shared projection dimension (audio & text both project to this dim).
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,

    /// Activation for projection heads.
    #[serde(default = "default_projection_hidden_act")]
    pub projection_hidden_act: String,

    /// Audio encoder configuration.
    pub audio_config: ClapAudioConfig,

    /// Text encoder configuration.
    pub text_config: ClapTextConfig,
}

/// Audio encoder config (HTS-AT / Swin Transformer).
#[derive(Debug, Clone, Deserialize)]
pub struct ClapAudioConfig {
    /// Number of mel frequency bins (default 64).
    #[serde(default = "default_num_mel_bins")]
    pub num_mel_bins: usize,

    /// Spectrogram time dimension target (default 256).
    #[serde(default = "default_spec_size")]
    pub spec_size: usize,

    /// Swin Transformer window size (default 8).
    #[serde(default = "default_window_size")]
    pub window_size: usize,

    /// Patch size for patch embedding (default 4).
    #[serde(default = "default_patch_size")]
    pub patch_size: usize,

    /// Patch stride [height, width] (default [4, 4]).
    #[serde(default = "default_patch_stride")]
    pub patch_stride: Vec<usize>,

    /// Initial embedding dimension after patch embed (default 96).
    #[serde(default = "default_patch_embeds_hidden_size")]
    pub patch_embeds_hidden_size: usize,

    /// Number of input channels for patch embedding (default 1).
    #[serde(default = "default_one")]
    pub patch_embed_input_channels: usize,

    /// Swin Transformer stage depths, e.g. [2, 2, 6, 2].
    #[serde(default = "default_depths")]
    pub depths: Vec<usize>,

    /// Per-stage attention heads, e.g. [4, 8, 16, 32].
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: Vec<usize>,

    /// Final hidden size (after all stages, default 768).
    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    /// Enable mel spectrogram fusion (AFF module).
    #[serde(default)]
    pub enable_fusion: bool,

    /// Enable patch-level fusion.
    #[serde(default)]
    pub enable_patch_fusion: bool,

    /// Enable LayerNorm after patch embedding.
    #[serde(default = "default_true")]
    pub enable_patch_layer_norm: bool,

    /// Whether to flatten patch embeddings.
    #[serde(default = "default_true")]
    pub flatten_patch_embeds: bool,

    /// MLP expansion ratio in Swin blocks (default 4.0).
    #[serde(default = "default_mlp_ratio")]
    pub mlp_ratio: f64,

    /// Whether QKV projections have bias (default true).
    #[serde(default = "default_true")]
    pub qkv_bias: bool,

    /// Number of AudioSet classes (default 527).
    #[serde(default = "default_num_classes")]
    pub num_classes: usize,

    /// Activation function (default "gelu").
    #[serde(default = "default_gelu")]
    pub hidden_act: String,

    /// Projection dimension for audio embeddings.
    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,

    /// Hidden size of the projection MLP.
    #[serde(default = "default_hidden_size")]
    pub projection_hidden_size: usize,

    /// AFF block reduction ratio (default 4).
    #[serde(default = "default_aff_block_r")]
    pub aff_block_r: usize,
}

impl ClapAudioConfig {
    /// Number of Swin Transformer stages.
    pub fn num_stages(&self) -> usize {
        self.depths.len()
    }

    /// Hidden dimension at a given stage.
    /// Each stage doubles the channel count via patch merging.
    pub fn stage_dim(&self, stage: usize) -> usize {
        self.patch_embeds_hidden_size * (1 << stage)
    }

    /// Spatial dimensions of the feature map after patch embedding.
    /// Returns (H, W) where:
    ///   H = spec_size / patch_stride[0]
    ///   W = spec_size / patch_stride[1]
    ///
    /// In Python: img_size = (spec_size, spec_size), grid_size = img_size // patch_stride.
    /// The reshape_mel2img step converts the mel to a square (spec_size × spec_size) image.
    pub fn patches_resolution(&self) -> (usize, usize) {
        let h = self.spec_size / self.patch_stride[0];
        let w = self.spec_size / self.patch_stride[1];
        (h, w)
    }
}

/// Text encoder config (RoBERTa-based).
#[derive(Debug, Clone, Deserialize)]
pub struct ClapTextConfig {
    #[serde(default)]
    pub model_type: Option<String>,

    #[serde(default = "default_hidden_size")]
    pub hidden_size: usize,

    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,

    #[serde(default = "default_text_num_hidden_layers")]
    pub num_hidden_layers: usize,

    #[serde(default = "default_text_num_attention_heads")]
    pub num_attention_heads: usize,

    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,

    /// Number of token types (RoBERTa = 1, BERT = 2).
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,

    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    #[serde(default = "default_projection_dim")]
    pub projection_dim: usize,

    #[serde(default = "default_hidden_size")]
    pub projection_hidden_size: usize,

    #[serde(default = "default_projection_hidden_act")]
    pub projection_hidden_act: String,

    #[serde(default)]
    pub pad_token_id: usize,
}

impl ClapTextConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// Defaults
fn default_projection_dim() -> usize {
    512
}
fn default_projection_hidden_act() -> String {
    "relu".into()
}
fn default_num_mel_bins() -> usize {
    64
}
fn default_spec_size() -> usize {
    256
}
fn default_window_size() -> usize {
    8
}
fn default_patch_size() -> usize {
    4
}
fn default_patch_stride() -> Vec<usize> {
    vec![4, 4]
}
fn default_patch_embeds_hidden_size() -> usize {
    96
}
fn default_one() -> usize {
    1
}
fn default_depths() -> Vec<usize> {
    vec![2, 2, 6, 2]
}
fn default_num_attention_heads() -> Vec<usize> {
    vec![4, 8, 16, 32]
}
fn default_hidden_size() -> usize {
    768
}
fn default_mlp_ratio() -> f64 {
    4.0
}
fn default_true() -> bool {
    true
}
fn default_num_classes() -> usize {
    527
}
fn default_gelu() -> String {
    "gelu".into()
}
fn default_aff_block_r() -> usize {
    4
}
fn default_intermediate_size() -> usize {
    3072
}
fn default_text_num_hidden_layers() -> usize {
    12
}
fn default_text_num_attention_heads() -> usize {
    12
}
fn default_vocab_size() -> usize {
    50265
}
fn default_max_position_embeddings() -> usize {
    514
}
fn default_type_vocab_size() -> usize {
    1
}
fn default_layer_norm_eps() -> f64 {
    1e-12
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clap_audio_stage_dim() {
        let cfg = ClapAudioConfig {
            patch_embeds_hidden_size: 96,
            depths: vec![2, 2, 6, 2],
            num_attention_heads: vec![4, 8, 16, 32],
            ..test_audio_config()
        };
        assert_eq!(cfg.stage_dim(0), 96);
        assert_eq!(cfg.stage_dim(1), 192);
        assert_eq!(cfg.stage_dim(2), 384);
        assert_eq!(cfg.stage_dim(3), 768);
        assert_eq!(cfg.num_stages(), 4);
    }

    #[test]
    fn test_clap_audio_patches_resolution() {
        let cfg = ClapAudioConfig {
            num_mel_bins: 64,
            spec_size: 256,
            patch_stride: vec![4, 4],
            ..test_audio_config()
        };
        // img_size = (256, 256), grid = (256/4, 256/4) = (64, 64)
        assert_eq!(cfg.patches_resolution(), (64, 64));
    }

    fn test_audio_config() -> ClapAudioConfig {
        ClapAudioConfig {
            num_mel_bins: 64,
            spec_size: 256,
            window_size: 8,
            patch_size: 4,
            patch_stride: vec![4, 4],
            patch_embeds_hidden_size: 96,
            patch_embed_input_channels: 1,
            depths: vec![2, 2, 6, 2],
            num_attention_heads: vec![4, 8, 16, 32],
            hidden_size: 768,
            enable_fusion: true,
            enable_patch_fusion: true,
            enable_patch_layer_norm: true,
            flatten_patch_embeds: true,
            mlp_ratio: 4.0,
            qkv_bias: true,
            num_classes: 527,
            hidden_act: "gelu".into(),
            projection_dim: 512,
            projection_hidden_size: 768,
            aff_block_r: 4,
        }
    }
}
