//! Configuration structs for Qwen3 TTS.
//!
//! Deserialized directly from the HuggingFace `config.json`.
//! The top-level config nests a `talker_config` which itself nests a `code_predictor_config`.

use serde::Deserialize;

/// Top-level Qwen3 TTS config (maps to HF `config.json`).
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen3TtsConfig {
    pub model_type: String,
    /// Special token IDs for TTS framing.
    #[serde(default = "default_tts_bos")]
    pub tts_bos_token_id: u32,
    #[serde(default = "default_tts_eos")]
    pub tts_eos_token_id: u32,
    #[serde(default = "default_tts_pad")]
    pub tts_pad_token_id: u32,
    /// Nested talker (transformer) config.
    pub talker_config: TalkerConfig,
}

/// RoPE scaling config — contains MRoPE section sizes when present.
#[derive(Debug, Clone, Deserialize)]
pub struct RopeScalingConfig {
    #[serde(default)]
    pub mrope_section: Option<Vec<usize>>,
    #[serde(default)]
    pub interleaved: Option<bool>,
}

/// Talker transformer config.
#[derive(Debug, Clone, Deserialize)]
pub struct TalkerConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    /// In the HF config, `vocab_size` is the codec vocabulary (3072).
    /// `text_vocab_size` is the text vocabulary (151936).
    pub vocab_size: usize,
    #[serde(default = "default_text_vocab_size")]
    pub text_vocab_size: usize,
    #[serde(default = "default_text_hidden_size")]
    pub text_hidden_size: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_max_position")]
    pub max_position_embeddings: usize,
    /// RoPE scaling config — contains mrope_section for MRoPE.
    #[serde(default)]
    pub rope_scaling: Option<RopeScalingConfig>,
    /// Whether to use sliding window attention.
    #[serde(default)]
    pub use_sliding_window: bool,
    /// Sliding window size. Null in actual config, so optional.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Special codec token IDs used during generation.
    #[serde(default = "default_codec_bos")]
    pub codec_bos_id: u32,
    #[serde(default = "default_codec_eos")]
    pub codec_eos_token_id: u32,
    #[serde(default = "default_codec_pad")]
    pub codec_pad_id: u32,
    #[serde(default = "default_codec_think")]
    pub codec_think_id: u32,
    #[serde(default = "default_codec_nothink")]
    pub codec_nothink_id: u32,
    /// Nested code predictor config.
    pub code_predictor_config: CodePredictorConfig,
}

/// Code predictor sub-model config (generates codebooks 1-15).
#[derive(Debug, Clone, Deserialize)]
pub struct CodePredictorConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_head_dim")]
    pub head_dim: usize,
    #[serde(default = "default_rms_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_cp_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_cp_max_position")]
    pub max_position_embeddings: usize,
    /// Total code groups (including CB0). In HF config this is `num_code_groups: 16`.
    /// The code predictor generates `num_code_groups - 1` additional codebooks.
    #[serde(default = "default_num_code_groups")]
    pub num_code_groups: usize,
    /// Sliding window — null in actual config.
    #[serde(default)]
    pub sliding_window: Option<usize>,
    /// Predictor vocab size (2048 in HF config — different from talker codec vocab).
    #[serde(default = "default_cp_vocab_size")]
    pub vocab_size: usize,
}

impl TalkerConfig {
    /// Number of KV groups for GQA.
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Codec vocabulary size (= `vocab_size` in talker config, typically 3072).
    pub fn codec_vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// MRoPE section sizes, extracted from `rope_scaling.mrope_section`.
    /// Falls back to `[24, 20, 20]` if not present.
    pub fn mrope_section(&self) -> Vec<usize> {
        self.rope_scaling
            .as_ref()
            .and_then(|rs| rs.mrope_section.clone())
            .unwrap_or_else(|| vec![24, 20, 20])
    }
}

impl CodePredictorConfig {
    /// Number of KV groups for GQA.
    pub fn num_kv_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Number of additional codebooks to predict (= `num_code_groups - 1`).
    /// The talker generates CB0; the predictor generates CB1 through CB(n-1).
    pub fn num_codebooks(&self) -> usize {
        self.num_code_groups.saturating_sub(1)
    }

    /// Codec vocab size for the predictor's codebook embeddings and LM heads.
    /// This is the predictor's OWN vocab size (2048 in the default config),
    /// distinct from the talker's codec vocab size (3072).
    pub fn predictor_vocab_size(&self) -> usize {
        self.vocab_size
    }
}

// --- Defaults matching Qwen3-TTS-12Hz-1.7B-CustomVoice ---

fn default_tts_bos() -> u32 {
    151_672
}
fn default_tts_eos() -> u32 {
    151_673
}
fn default_tts_pad() -> u32 {
    151_671
}
fn default_head_dim() -> usize {
    128
}
fn default_text_vocab_size() -> usize {
    151_936
}
fn default_text_hidden_size() -> usize {
    2048
}
fn default_rms_eps() -> f64 {
    1e-6
}
fn default_rope_theta() -> f64 {
    1_000_000.0
}
fn default_max_position() -> usize {
    32768
}
fn default_codec_bos() -> u32 {
    2149
}
fn default_codec_eos() -> u32 {
    2150
}
fn default_codec_pad() -> u32 {
    2148
}
fn default_codec_think() -> u32 {
    2154
}
fn default_codec_nothink() -> u32 {
    2155
}
fn default_cp_rope_theta() -> f64 {
    1_000_000.0
}
fn default_cp_max_position() -> usize {
    65536
}
fn default_num_code_groups() -> usize {
    16
}
fn default_cp_vocab_size() -> usize {
    2048
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_talker_kv_groups() {
        let cfg = TalkerConfig {
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 28,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            head_dim: 128,
            vocab_size: 3072,
            text_vocab_size: 151_936,
            text_hidden_size: 2048,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 32768,
            rope_scaling: Some(RopeScalingConfig {
                mrope_section: Some(vec![24, 20, 20]),
                interleaved: Some(true),
            }),
            use_sliding_window: false,
            sliding_window: None,
            codec_bos_id: 2149,
            codec_eos_token_id: 2150,
            codec_pad_id: 2148,
            codec_think_id: 2154,
            codec_nothink_id: 2155,
            code_predictor_config: CodePredictorConfig {
                hidden_size: 1024,
                intermediate_size: 3072,
                num_hidden_layers: 5,
                num_attention_heads: 16,
                num_key_value_heads: 8,
                head_dim: 128,
                rms_norm_eps: 1e-6,
                rope_theta: 1_000_000.0,
                max_position_embeddings: 65536,
                num_code_groups: 16,
                sliding_window: None,
                vocab_size: 2048,
            },
        };
        assert_eq!(cfg.num_kv_groups(), 2);
        assert_eq!(cfg.code_predictor_config.num_kv_groups(), 2);
        assert_eq!(cfg.codec_vocab_size(), 3072);
        assert_eq!(cfg.mrope_section(), vec![24, 20, 20]);
        assert_eq!(cfg.code_predictor_config.num_codebooks(), 15);
    }

    #[test]
    fn test_parse_actual_config() {
        let json = r#"{
            "model_type": "qwen3_tts",
            "tts_bos_token_id": 151672,
            "tts_eos_token_id": 151673,
            "tts_pad_token_id": 151671,
            "talker_config": {
                "hidden_size": 2048,
                "intermediate_size": 6144,
                "num_hidden_layers": 28,
                "num_attention_heads": 16,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 3072,
                "text_vocab_size": 151936,
                "text_hidden_size": 2048,
                "rms_norm_eps": 1e-6,
                "rope_theta": 1000000,
                "max_position_embeddings": 32768,
                "rope_scaling": {
                    "interleaved": true,
                    "mrope_section": [24, 20, 20],
                    "rope_type": "default",
                    "type": "default"
                },
                "sliding_window": null,
                "use_sliding_window": false,
                "codec_bos_id": 2149,
                "codec_eos_token_id": 2150,
                "codec_pad_id": 2148,
                "codec_think_id": 2154,
                "codec_nothink_id": 2155,
                "code_predictor_config": {
                    "hidden_size": 1024,
                    "intermediate_size": 3072,
                    "num_hidden_layers": 5,
                    "num_attention_heads": 16,
                    "num_key_value_heads": 8,
                    "head_dim": 128,
                    "rms_norm_eps": 1e-6,
                    "rope_theta": 1000000,
                    "max_position_embeddings": 65536,
                    "num_code_groups": 16,
                    "sliding_window": null,
                    "vocab_size": 2048,
                    "attention_bias": false,
                    "attention_dropout": 0,
                    "architectures": null,
                    "bos_token_id": null,
                    "eos_token_id": null,
                    "pad_token_id": null
                }
            }
        }"#;
        let cfg: Qwen3TtsConfig = serde_json::from_str(json).expect("parse config");
        assert_eq!(cfg.model_type, "qwen3_tts");
        assert_eq!(cfg.talker_config.codec_vocab_size(), 3072);
        assert_eq!(cfg.talker_config.text_vocab_size, 151_936);
        assert_eq!(cfg.talker_config.mrope_section(), vec![24, 20, 20]);
        assert_eq!(cfg.talker_config.sliding_window, None);
        assert_eq!(cfg.talker_config.code_predictor_config.num_codebooks(), 15);
        assert_eq!(cfg.talker_config.code_predictor_config.head_dim, 128);
        assert_eq!(
            cfg.talker_config
                .code_predictor_config
                .max_position_embeddings,
            65536
        );
    }
}
