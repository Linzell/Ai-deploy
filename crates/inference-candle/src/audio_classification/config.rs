//! Wav2Vec2 / HuBERT configuration.
//!
//! Covers `Wav2Vec2ForSequenceClassification` and `HubertForSequenceClassification`.

use serde::Deserialize;

/// Top-level config.json for wav2vec2 / hubert audio classification models.
#[derive(Debug, Clone, Deserialize)]
pub struct Wav2Vec2Config {
    /// `"wav2vec2"` or `"hubert"`
    #[serde(default)]
    pub model_type: Option<String>,

    // ---- Feature extractor (CNN) ----
    /// Per-layer output channels, e.g. `[512, 512, 512, 512, 512, 512, 512]`
    #[serde(default = "default_conv_dim")]
    pub conv_dim: Vec<usize>,
    /// Per-layer kernel sizes, e.g. `[10, 3, 3, 3, 3, 2, 2]`
    #[serde(default = "default_conv_kernel")]
    pub conv_kernel: Vec<usize>,
    /// Per-layer strides, e.g. `[5, 2, 2, 2, 2, 2, 2]`
    #[serde(default = "default_conv_stride")]
    pub conv_stride: Vec<usize>,
    /// Whether conv layers have bias
    #[serde(default)]
    pub conv_bias: bool,
    /// `"group"` (GroupNorm on first layer only) or `"layer"` (LayerNorm on all layers)
    #[serde(default = "default_feat_extract_norm")]
    pub feat_extract_norm: String,

    // ---- Feature projection ----
    // Projects last conv_dim → hidden_size

    // ---- Positional conv embedding ----
    /// Kernel size for the positional conv, default 128
    #[serde(default = "default_num_conv_pos_embeddings")]
    pub num_conv_pos_embeddings: usize,
    /// Groups for the positional conv, default 16
    #[serde(default = "default_num_conv_pos_embedding_groups")]
    pub num_conv_pos_embedding_groups: usize,

    // ---- Transformer encoder ----
    pub hidden_size: usize,
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
    #[serde(default = "default_num_attention_heads")]
    pub num_attention_heads: usize,
    #[serde(default = "default_intermediate_size")]
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,

    /// If true, LayerNorm is applied *before* attention/FFN (pre-norm).
    /// If false, LayerNorm is applied *after* (post-norm).
    #[serde(default)]
    pub do_stable_layer_norm: bool,

    // ---- Classification head ----
    /// Projection size before classifier, default 256
    #[serde(default = "default_classifier_proj_size")]
    pub classifier_proj_size: usize,
    /// Number of output labels
    #[serde(default = "default_num_labels")]
    pub num_labels: usize,
}

impl Wav2Vec2Config {
    /// Last conv layer's output dimension (input to feature projection).
    pub fn last_conv_dim(&self) -> usize {
        *self.conv_dim.last().unwrap_or(&512)
    }

    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Backbone prefix derived from `model_type`.
    /// `"wav2vec2"` → `"wav2vec2"`, `"hubert"` → `"hubert"`, fallback → `"wav2vec2"`.
    pub fn backbone_prefix(&self) -> &str {
        match self.model_type.as_deref() {
            Some("hubert") => "hubert",
            _ => "wav2vec2",
        }
    }
}

fn default_conv_dim() -> Vec<usize> {
    vec![512; 7]
}
fn default_conv_kernel() -> Vec<usize> {
    vec![10, 3, 3, 3, 3, 2, 2]
}
fn default_conv_stride() -> Vec<usize> {
    vec![5, 2, 2, 2, 2, 2, 2]
}
fn default_feat_extract_norm() -> String {
    "group".into()
}
fn default_num_conv_pos_embeddings() -> usize {
    128
}
fn default_num_conv_pos_embedding_groups() -> usize {
    16
}
fn default_num_hidden_layers() -> usize {
    12
}
fn default_num_attention_heads() -> usize {
    12
}
fn default_intermediate_size() -> usize {
    3072
}
fn default_hidden_act() -> String {
    "gelu".into()
}
fn default_layer_norm_eps() -> f64 {
    1e-5
}
fn default_classifier_proj_size() -> usize {
    256
}
fn default_num_labels() -> usize {
    2
}
