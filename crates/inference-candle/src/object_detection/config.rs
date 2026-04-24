//! DETR configuration.
//!
//! Covers `DetrForObjectDetection` and `TableTransformerForObjectDetection`.

use serde::Deserialize;

/// Top-level config.json for DETR-family object detection models.
#[derive(Debug, Clone, Deserialize)]
pub struct DetrConfig {
    /// `"detr"` or `"table-transformer"`
    #[serde(default)]
    pub model_type: Option<String>,

    // ---- Backbone ----
    /// Backbone name, e.g. `"resnet50"`, `"resnet18"`
    #[serde(default = "default_backbone")]
    pub backbone: String,

    /// Number of input channels (default 3 for RGB)
    #[serde(default = "default_num_channels")]
    pub num_channels: usize,

    // ---- Transformer ----
    /// Hidden dimension for transformer encoder/decoder (default 256)
    #[serde(default = "default_d_model")]
    pub d_model: usize,

    /// Number of encoder layers (default 6)
    #[serde(default = "default_encoder_layers")]
    pub encoder_layers: usize,

    /// Number of decoder layers (default 6)
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,

    /// Number of encoder attention heads (default 8)
    #[serde(default = "default_attention_heads")]
    pub encoder_attention_heads: usize,

    /// Number of decoder attention heads (default 8)
    #[serde(default = "default_attention_heads")]
    pub decoder_attention_heads: usize,

    /// Encoder FFN intermediate dimension (default 2048)
    #[serde(default = "default_ffn_dim")]
    pub encoder_ffn_dim: usize,

    /// Decoder FFN intermediate dimension (default 2048)
    #[serde(default = "default_ffn_dim")]
    pub decoder_ffn_dim: usize,

    /// Activation function (default "relu")
    #[serde(default = "default_activation_function")]
    pub activation_function: String,

    /// Dropout (default 0.1)
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    // ---- Detection ----
    /// Number of object queries (default 100)
    #[serde(default = "default_num_queries")]
    pub num_queries: usize,

    /// Position embedding type: `"sine"` or `"learned"` (default "sine")
    #[serde(default = "default_position_embedding_type")]
    pub position_embedding_type: String,

    /// Use dilation in backbone (default false)
    #[serde(default)]
    pub dilation: bool,

    /// Total number of hidden layers (for ResNet backbone, default 6)
    #[serde(default = "default_num_hidden_layers")]
    pub num_hidden_layers: usize,
}

impl DetrConfig {
    /// Encoder head dimension.
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Decoder head dimension.
    pub fn decoder_head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }

    /// Whether backbone is ResNet-18 (vs ResNet-50).
    pub fn is_resnet18(&self) -> bool {
        self.backbone.contains("18")
    }

    /// ResNet stage output channels.
    /// ResNet-18/34: [64, 128, 256, 512]
    /// ResNet-50/101: [256, 512, 1024, 2048]
    pub fn backbone_channels(&self) -> [usize; 4] {
        if self.is_resnet18() {
            [64, 128, 256, 512]
        } else {
            [256, 512, 1024, 2048]
        }
    }

    /// The channel count of the last backbone stage (input to input_projection).
    pub fn backbone_out_channels(&self) -> usize {
        *self.backbone_channels().last().unwrap()
    }
}

fn default_backbone() -> String {
    "resnet50".into()
}
fn default_num_channels() -> usize {
    3
}
fn default_d_model() -> usize {
    256
}
fn default_encoder_layers() -> usize {
    6
}
fn default_decoder_layers() -> usize {
    6
}
fn default_attention_heads() -> usize {
    8
}
fn default_ffn_dim() -> usize {
    2048
}
fn default_activation_function() -> String {
    "relu".into()
}
fn default_dropout() -> f64 {
    0.1
}
fn default_num_queries() -> usize {
    100
}
fn default_position_embedding_type() -> String {
    "sine".into()
}
fn default_num_hidden_layers() -> usize {
    6
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_detr_config() {
        let json = r#"{
            "model_type": "detr",
            "backbone": "resnet50",
            "d_model": 256,
            "encoder_layers": 6,
            "decoder_layers": 6,
            "num_queries": 100
        }"#;

        let config: DetrConfig = serde_json::from_str(json).unwrap();
        assert_eq!(config.backbone, "resnet50");
        assert_eq!(config.d_model, 256);
        assert_eq!(config.num_queries, 100);
        assert!(!config.is_resnet18());
        assert_eq!(config.backbone_out_channels(), 2048);
    }

    #[test]
    fn test_parse_table_transformer_config() {
        let json = r#"{
            "model_type": "table-transformer",
            "backbone": "resnet18",
            "d_model": 256,
            "num_queries": 15
        }"#;

        let config: DetrConfig = serde_json::from_str(json).unwrap();
        assert!(config.is_resnet18());
        assert_eq!(config.backbone_out_channels(), 512);
        assert_eq!(config.num_queries, 15);
    }
}
