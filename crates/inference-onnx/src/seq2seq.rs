//! Sequence-to-sequence task for encoder-decoder and decoder-only models.
//!
//! Supports autoregressive text generation for:
//! - **Decoder-only** models (GPT2, Llama): Single model with KV-cache
//! - **Encoder-decoder** models (T5, BART, MarianMT, Whisper): Encoder + decoder
//!
//! ## Architecture
//!
//! ```text
//! Decoder-only (text-generation):
//!   input_ids → [Decoder + KV-cache] → logits → sample → next_token
//!                      ↑                                    ↓
//!                      └──────────────────────────────────────┘
//!
//! Encoder-decoder (translation, summarization, ASR):
//!   input → [Encoder] → encoder_hidden_states
//!                              ↓
//!   decoder_input_ids → [Decoder + cross-attention] → logits → sample → next_token
//!                              ↑                                          ↓
//!                              └──────────────────────────────────────────┘
//! ```
//!
//! ## ONNX Model Files
//!
//! Encoder-decoder models from HuggingFace typically export as:
//! - `encoder_model.onnx` - Encoder (run once per input)
//! - `decoder_model.onnx` - Decoder without KV-cache (first token)
//! - `decoder_model_merged.onnx` or `decoder_with_past_model.onnx` - Decoder with KV-cache
//!
//! Decoder-only models export as:
//! - `decoder_model.onnx` - Without KV-cache
//! - `decoder_model_merged.onnx` - With KV-cache

use crate::error::{TaskError, TaskResult};
use crate::session::load_session_for_seq2seq;
use crate::tensor_utils::{json_to_array2_i64, json_to_array_f32};
use async_trait::async_trait;
use inference_core::generation::{GenerationConfig, ModelArchitecture};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use ndarray::{Array2, Array3, Array4, ArrayD};
use ort::session::Session;
use ort::value::{TensorRef, ValueType};
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info};

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

/// Type of KV-cache for encoder-decoder models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KvCacheType {
    /// Decoder self-attention cache - starts empty, grows with each token
    DecoderSelfAttention,
    /// Encoder cross-attention cache - initialized from encoder output, stays constant
    EncoderCrossAttention,
}

/// KV-cache input metadata extracted from model.
#[derive(Debug, Clone)]
struct KvCacheInfo {
    /// Input name (e.g., "past_key_values.0.key" or "past_key_values.0.decoder.key")
    name: String,
    /// Number of attention heads (e.g., 12 for GPT2, 8 for T5-small)
    num_heads: usize,
    /// Head dimension (typically 64)
    head_dim: usize,
    /// Type of cache (decoder self-attention vs encoder cross-attention)
    cache_type: KvCacheType,
    /// Number of dimensions in the tensor (3 or 4)
    /// 4D: [batch, num_heads, seq_len, head_dim]
    /// 3D: [num_heads, seq_len, head_dim] (batch=1 implied)
    ndim: usize,
}

/// Sequence-to-sequence task for autoregressive generation.
///
/// Handles both decoder-only and encoder-decoder architectures with
/// efficient KV-cache support for incremental generation.
///
/// ## Florence-2 Style Models
///
/// Some vision-language models (like Florence-2) use a 3-part architecture:
/// 1. `vision_encoder.onnx`: pixel_values → image_features
/// 2. `embed_tokens.onnx`: input_ids → text_embeddings
/// 3. `decoder_model_merged.onnx`: inputs_embeds (concat) → logits
///
/// For these models, `embed_tokens` is Some and `decoder_uses_embeds` is true.
pub struct Seq2SeqTask {
    name: String,
    architecture: ModelArchitecture,
    /// Encoder session (only for encoder-decoder models)
    encoder: Option<Arc<Mutex<Session>>>,
    /// Decoder session (with KV-cache support if available)
    decoder: Arc<Mutex<Session>>,
    /// Embed tokens session for models like Florence-2 that need separate token embedding
    embed_tokens: Option<Arc<Mutex<Session>>>,
    /// Whether decoder expects inputs_embeds instead of input_ids (Florence-2 style)
    decoder_uses_embeds: bool,
    /// Generation configuration
    gen_config: GenerationConfig,
    /// Preprocessor for input tokenization
    #[cfg(feature = "preprocess")]
    preprocessor: Arc<Preprocessor>,
    /// Whether the decoder model has KV-cache inputs
    has_kv_cache: bool,
    /// Input names for the decoder
    decoder_input_names: Vec<String>,
    /// Output names for the decoder (for KV-cache handling)
    decoder_output_names: Vec<String>,
    /// KV-cache input metadata (name, num_heads, head_dim) for each cache tensor
    kv_cache_inputs: Vec<KvCacheInfo>,
    /// Maximum position embeddings supported by the model.
    /// For Florence-2, this is 1026 (indices 0-1025).
    /// Generation will stop if total sequence length would exceed this limit.
    max_position_embeddings: usize,
}

impl Seq2SeqTask {
    /// Extract KV-cache input metadata from a session.
    /// Returns a list of KvCacheInfo for each past_key_values.* input.
    fn extract_kv_cache_info(session: &Session) -> Vec<KvCacheInfo> {
        let mut kv_cache_inputs = Vec::new();

        for input in session.inputs() {
            let name = input.name();
            if !name.starts_with("past_key_values.") {
                continue;
            }

            // Determine cache type from name:
            // - "past_key_values.X.encoder.key/value" -> encoder cross-attention (T5/BART)
            // - "past_key_values.X.decoder.key/value" -> decoder self-attention (T5/BART)
            // - "past_key_values.X.key/value" -> decoder self-attention (GPT2)
            let cache_type = if name.contains(".encoder.") {
                KvCacheType::EncoderCrossAttention
            } else {
                KvCacheType::DecoderSelfAttention
            };

            // Extract shape from input type
            // KV-cache tensors can be:
            // - 4D: [batch, num_heads, seq_len, head_dim]
            // - 3D: [num_heads, seq_len, head_dim] (batch=1 implied)
            if let ValueType::Tensor { shape, .. } = input.dtype() {
                let dims: Vec<i64> = shape.iter().copied().collect();
                let ndim = dims.len();

                if ndim == 4 {
                    // 4D: dims[1] = num_heads, dims[3] = head_dim
                    // dims[2] is past_sequence_length (dynamic, often -1)
                    let num_heads = usize::try_from(dims[1]).unwrap_or(12);
                    let head_dim = usize::try_from(dims[3]).unwrap_or(64);

                    kv_cache_inputs.push(KvCacheInfo {
                        name: name.to_string(),
                        num_heads,
                        head_dim,
                        cache_type,
                        ndim,
                    });
                } else if ndim == 3 {
                    // 3D: dims[0] = num_heads, dims[2] = head_dim
                    // dims[1] is past_sequence_length (dynamic)
                    let num_heads = usize::try_from(dims[0]).unwrap_or(8);
                    let head_dim = usize::try_from(dims[2]).unwrap_or(64);

                    kv_cache_inputs.push(KvCacheInfo {
                        name: name.to_string(),
                        num_heads,
                        head_dim,
                        cache_type,
                        ndim,
                    });
                }
            }
        }

        kv_cache_inputs
    }

    /// Create a decoder-only task (GPT2, Llama, etc.).
    #[cfg(feature = "preprocess")]
    pub fn decoder_only(
        decoder_path: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let decoder_path = decoder_path.as_ref();
        info!(path = %decoder_path.display(), "Loading decoder-only model");

        let decoder = load_session_for_seq2seq(decoder_path, config)?;
        let decoder_input_names: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let decoder_output_names: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Check if model has KV-cache inputs (past_key_values.*)
        let has_kv_cache = decoder_input_names
            .iter()
            .any(|n| n.contains("past_key_values"));

        // Extract KV-cache metadata from model inputs
        let kv_cache_inputs = Self::extract_kv_cache_info(&decoder);

        info!(
            inputs = ?decoder_input_names,
            outputs = ?decoder_output_names,
            has_kv_cache = has_kv_cache,
            num_kv_cache_inputs = kv_cache_inputs.len(),
            "Decoder model loaded"
        );

        Ok(Self {
            name: name.into(),
            architecture: ModelArchitecture::DecoderOnly,
            encoder: None,
            decoder: Arc::new(Mutex::new(decoder)),
            embed_tokens: None,
            decoder_uses_embeds: false,
            gen_config: GenerationConfig::from_config(config),
            preprocessor,
            has_kv_cache,
            decoder_input_names,
            decoder_output_names,
            kv_cache_inputs,
            // Decoder-only models typically support longer contexts
            max_position_embeddings: 2048,
        })
    }

    /// Create an encoder-decoder task (T5, BART, Whisper, etc.).
    /// This is the legacy method without embed_tokens support.
    #[cfg(feature = "preprocess")]
    pub fn encoder_decoder(
        encoder_path: impl AsRef<Path>,
        decoder_path: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        Self::encoder_decoder_with_embed(
            encoder_path,
            decoder_path,
            None,
            name,
            config,
            preprocessor,
        )
    }

    /// Create an encoder-decoder task with optional embed_tokens (Florence-2 style).
    ///
    /// For Florence-2 style models:
    /// - `embed_tokens_path`: Some(path) to embed_tokens.onnx
    /// - Decoder expects `inputs_embeds` instead of `input_ids`
    /// - Image features + text embeddings are concatenated before decoder
    #[cfg(feature = "preprocess")]
    pub fn encoder_decoder_with_embed(
        encoder_path: impl AsRef<Path>,
        decoder_path: impl AsRef<Path>,
        embed_tokens_path: Option<&Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let encoder_path = encoder_path.as_ref();
        let decoder_path = decoder_path.as_ref();

        info!(
            encoder = %encoder_path.display(),
            decoder = %decoder_path.display(),
            embed_tokens = ?embed_tokens_path,
            "Loading encoder-decoder model"
        );

        let encoder = load_session_for_seq2seq(encoder_path, config)?;
        let decoder = load_session_for_seq2seq(decoder_path, config)?;

        // Load embed_tokens if present (Florence-2 style)
        let embed_tokens = if let Some(path) = embed_tokens_path {
            info!(embed_tokens = %path.display(), "Loading embed_tokens model");
            Some(Arc::new(Mutex::new(load_session_for_seq2seq(
                path, config,
            )?)))
        } else {
            None
        };

        let encoder_inputs: Vec<String> = encoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let encoder_outputs: Vec<String> = encoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let decoder_input_names: Vec<String> = decoder
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let decoder_output_names: Vec<String> = decoder
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        // Check if decoder uses inputs_embeds (Florence-2 style)
        let decoder_uses_embeds = decoder_input_names.contains(&"inputs_embeds".to_string());

        info!(
            encoder_inputs = ?encoder_inputs,
            encoder_outputs = ?encoder_outputs,
            decoder_inputs = ?decoder_input_names,
            decoder_outputs = ?decoder_output_names,
            decoder_uses_embeds = decoder_uses_embeds,
            has_embed_tokens = embed_tokens.is_some(),
            "Encoder-decoder model loaded"
        );

        let has_kv_cache = decoder_input_names
            .iter()
            .any(|n| n.contains("past_key_values"));

        // Extract KV-cache metadata from model inputs
        let kv_cache_inputs = Self::extract_kv_cache_info(&decoder);

        // Determine max position embeddings based on model type
        // Florence-2 style models (with embed_tokens) have max 1024 positions in config,
        // but the model uses an offset of 2 in Florence2LearnedPositionalEmbedding:
        //   positions = torch.arange(past_key_values_length, past_key_values_length + seq_len)
        //   return super().forward(positions + self.offset)  # offset = 2
        // So max valid position index is (max_position_embeddings - 1) = 1023, which maps to
        // embedding index 1023 + 2 = 1025, which is valid in the 1026-size embedding table.
        // If we try position 1024, it becomes 1024 + 2 = 1026, which is out of bounds!
        let max_position_embeddings = if decoder_uses_embeds {
            // Florence-2 style: limited to 1024 positions (0-1023)
            // This is the actual config value from the ONNX community model
            1024
        } else {
            // Standard encoder-decoder (T5, BART, Whisper): typically 512-2048
            2048
        };

        Ok(Self {
            name: name.into(),
            architecture: ModelArchitecture::EncoderDecoder,
            encoder: Some(Arc::new(Mutex::new(encoder))),
            decoder: Arc::new(Mutex::new(decoder)),
            embed_tokens,
            decoder_uses_embeds,
            gen_config: GenerationConfig::from_config(config),
            preprocessor,
            has_kv_cache,
            decoder_input_names,
            decoder_output_names,
            kv_cache_inputs,
            max_position_embeddings,
        })
    }

    /// Load from model directory with auto-detection of architecture.
    #[cfg(feature = "preprocess")]
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let onnx_dir = model_dir.join("onnx");

        // Try to find encoder and decoder files
        // For vision-language models (Florence-2), prefer vision_encoder.onnx over encoder_model.onnx
        // because vision_encoder takes pixel_values directly, while encoder_model expects pre-computed embeddings
        let encoder_path =
            Self::find_model_file(&onnx_dir, &["vision_encoder.onnx", "encoder_model.onnx"]);
        let decoder_path = Self::find_model_file(
            &onnx_dir,
            &[
                "model.onnx",                // transformers.js v3 format (decoder-only with KV-cache)
                "model_q4f16.onnx",          // ONNX community INT4 quantized (GPT-OSS, etc.)
                "model_q4.onnx",             // INT4 quantized variant
                "decoder_model_merged.onnx", // Optimum merged decoder with KV-cache
                "decoder_with_past_model.onnx",
                "decoder_model.onnx",
            ],
        );

        // Check for embed_tokens.onnx (Florence-2 style models)
        let embed_tokens_path = Self::find_model_file(&onnx_dir, &["embed_tokens.onnx"]);

        match (encoder_path, decoder_path) {
            (Some(enc), Some(dec)) => Self::encoder_decoder_with_embed(
                enc,
                dec,
                embed_tokens_path.as_deref(),
                name,
                config,
                preprocessor,
            ),
            (None, Some(dec)) => Self::decoder_only(dec, name, config, preprocessor),
            (Some(_), None) => Err(TaskError::ModelNotFound(
                "Found encoder but no decoder model".to_string(),
            )),
            (None, None) => Err(TaskError::ModelNotFound(format!(
                "No ONNX models found in {}",
                onnx_dir.display()
            ))),
        }
    }

    /// Find the first existing model file from a list of candidates.
    fn find_model_file(dir: &Path, candidates: &[&str]) -> Option<PathBuf> {
        for candidate in candidates {
            let path = dir.join(candidate);
            if path.exists() {
                return Some(path);
            }
        }
        None
    }

    /// Run the encoder and return hidden states.
    ///
    /// For text inputs (translation, summarization):
    ///   - `input_ids`: tokenized text [batch, seq_len]
    ///   - `attention_mask`: attention mask [batch, seq_len]
    async fn run_encoder(
        &self,
        input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
    ) -> TaskResult<ArrayD<f32>> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| TaskError::Config("No encoder for decoder-only model".into()))?;

        let encoder = Arc::clone(encoder);
        let input_ids = input_ids.clone();
        let attention_mask = attention_mask.clone();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let input_ids_dyn = input_ids.clone().into_dyn();
            let attention_mask_dyn = attention_mask.clone().into_dyn();

            let input_tensor = TensorRef::from_array_view(&input_ids_dyn)
                .map_err(|e| TaskError::Inference(format!("Failed to create input tensor: {e}")))?;
            let mask_tensor = TensorRef::from_array_view(&attention_mask_dyn)
                .map_err(|e| TaskError::Inference(format!("Failed to create mask tensor: {e}")))?;

            let inputs: Vec<(&str, SessionInputValue)> = vec![
                ("input_ids", input_tensor.into_dyn().into()),
                ("attention_mask", mask_tensor.into_dyn().into()),
            ];

            let mut session = encoder.blocking_lock();
            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("Encoder failed: {e}")))?;

            // Get encoder hidden states (usually first output)
            let hidden_states = outputs
                .get("last_hidden_state")
                .or_else(|| outputs.get("encoder_hidden_states"))
                .ok_or_else(|| TaskError::Inference("No encoder output found".into()))?;

            hidden_states
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract encoder output: {e}")))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Run the encoder with audio input (mel spectrogram) for ASR models like Whisper.
    ///
    /// For audio inputs (automatic-speech-recognition):
    ///   - `input_features`: mel spectrogram [batch, n_mels, time_frames]
    async fn run_encoder_audio(&self, input_features: &Array3<f32>) -> TaskResult<ArrayD<f32>> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| TaskError::Config("No encoder for decoder-only model".into()))?;

        let encoder = Arc::clone(encoder);
        let input_features = input_features.clone();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let input_features_dyn = input_features.into_dyn();
            let input_tensor = TensorRef::from_array_view(&input_features_dyn).map_err(|e| {
                TaskError::Inference(format!("Failed to create input_features tensor: {e}"))
            })?;

            let inputs: Vec<(&str, SessionInputValue)> =
                vec![("input_features", input_tensor.into_dyn().into())];

            let mut session = encoder.blocking_lock();
            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("Encoder failed: {e}")))?;

            // Get encoder hidden states
            let hidden_states = outputs
                .get("last_hidden_state")
                .or_else(|| outputs.get("encoder_hidden_states"))
                .ok_or_else(|| TaskError::Inference("No encoder output found".into()))?;

            hidden_states
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract encoder output: {e}")))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Run the encoder with image input (pixel values) for image-to-text models like TrOCR.
    ///
    /// For image inputs (image-to-text):
    ///   - `pixel_values`: normalized image tensor [batch, channels, height, width]
    async fn run_encoder_image(&self, pixel_values: &Array4<f32>) -> TaskResult<ArrayD<f32>> {
        let encoder = self
            .encoder
            .as_ref()
            .ok_or_else(|| TaskError::Config("No encoder for decoder-only model".into()))?;

        let encoder = Arc::clone(encoder);
        let pixel_values = pixel_values.clone();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let pixel_values_dyn = pixel_values.into_dyn();
            let input_tensor = TensorRef::from_array_view(&pixel_values_dyn).map_err(|e| {
                TaskError::Inference(format!("Failed to create pixel_values tensor: {e}"))
            })?;

            let inputs: Vec<(&str, SessionInputValue)> =
                vec![("pixel_values", input_tensor.into_dyn().into())];

            let mut session = encoder.blocking_lock();
            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("Encoder failed: {e}")))?;

            // Get encoder hidden states - different models use different output names:
            // - TrOCR: "last_hidden_state"
            // - Florence-2 vision_encoder: "image_features"
            let hidden_states = outputs
                .get("last_hidden_state")
                .or_else(|| outputs.get("encoder_hidden_states"))
                .or_else(|| outputs.get("image_features"))
                .ok_or_else(|| TaskError::Inference("No encoder output found".into()))?;

            hidden_states
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract encoder output: {e}")))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Run embed_tokens model to convert input_ids to embeddings (Florence-2 style).
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs [batch, seq_len]
    ///
    /// # Returns
    /// Token embeddings [batch, seq_len, hidden_dim]
    async fn run_embed_tokens(&self, input_ids: &Array2<i64>) -> TaskResult<ArrayD<f32>> {
        let embed_tokens = self
            .embed_tokens
            .as_ref()
            .ok_or_else(|| TaskError::Config("No embed_tokens model loaded".into()))?;

        let embed_tokens = Arc::clone(embed_tokens);
        let input_ids = input_ids.clone();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let input_ids_dyn = input_ids.into_dyn();
            let input_tensor = TensorRef::from_array_view(&input_ids_dyn)
                .map_err(|e| TaskError::Inference(format!("Failed to create input tensor: {e}")))?;

            let inputs: Vec<(&str, SessionInputValue)> =
                vec![("input_ids", input_tensor.into_dyn().into())];

            let mut session = embed_tokens.blocking_lock();
            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("embed_tokens failed: {e}")))?;

            // Get embeddings - typically named "inputs_embeds"
            // Florence-2's embed_tokens outputs "inputs_embeds"
            let embeddings = outputs.get("inputs_embeds").ok_or_else(|| {
                TaskError::Inference("No inputs_embeds output from embed_tokens".into())
            })?;

            embeddings
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract embeddings: {e}")))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Run decoder step and return logits for next token plus updated KV-cache.
    ///
    /// # Arguments
    /// * `decoder_input_ids` - Token IDs to process. On first step, full sequence; with cache, just new tokens.
    /// * `attention_mask` - Full attention mask (grows with each generated token)
    /// * `encoder_hidden_states` - Encoder output for encoder-decoder models
    /// * `encoder_attention_mask` - Encoder attention mask for encoder-decoder models
    /// * `past_key_values` - KV-cache from previous step (None for first step)
    ///
    /// # Returns
    /// Tuple of (logits, new_kv_cache) where new_kv_cache should be passed to next step.
    /// KV-cache uses dynamic arrays to support both 3D and 4D formats.
    async fn run_decoder_step(
        &self,
        decoder_input_ids: &Array2<i64>,
        attention_mask: &Array2<i64>,
        encoder_hidden_states: Option<&ArrayD<f32>>,
        encoder_attention_mask: Option<&Array2<i64>>,
        past_key_values: Option<&HashMap<String, ArrayD<f32>>>,
    ) -> TaskResult<(ArrayD<f32>, HashMap<String, ArrayD<f32>>)> {
        let decoder = Arc::clone(&self.decoder);
        let decoder_input_ids = decoder_input_ids.clone();
        let attention_mask = attention_mask.clone();
        let encoder_hidden_states = encoder_hidden_states.cloned();
        let encoder_attention_mask = encoder_attention_mask.cloned();
        let decoder_input_names = self.decoder_input_names.clone();
        let decoder_output_names = self.decoder_output_names.clone();
        let has_kv_cache = self.has_kv_cache;
        let kv_cache_inputs = self.kv_cache_inputs.clone();
        let past_key_values = past_key_values.cloned();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let use_cache = past_key_values.is_some();

            let decoder_ids_dyn = decoder_input_ids.clone().into_dyn();
            let decoder_tensor = TensorRef::from_array_view(&decoder_ids_dyn).map_err(|e| {
                TaskError::Inference(format!("Failed to create decoder input: {e}"))
            })?;

            let mut inputs: Vec<(&str, SessionInputValue)> = Vec::new();

            // Add decoder input_ids (may be named "input_ids" or "decoder_input_ids")
            let decoder_ids_name = if decoder_input_names.contains(&"decoder_input_ids".to_string())
            {
                "decoder_input_ids"
            } else {
                "input_ids"
            };
            inputs.push((decoder_ids_name, decoder_tensor.into_dyn().into()));

            // Add attention_mask (required for decoder-only models)
            let attn_mask_dyn = attention_mask.clone().into_dyn();
            let attn_mask_tensor;
            if decoder_input_names.contains(&"attention_mask".to_string()) {
                attn_mask_tensor = TensorRef::from_array_view(&attn_mask_dyn).map_err(|e| {
                    TaskError::Inference(format!("Failed to create attention_mask tensor: {e}"))
                })?;
                inputs.push(("attention_mask", attn_mask_tensor.into_dyn().into()));
            }

            // Add position_ids for models that require it (Qwen, Llama, etc.)
            // position_ids = cumsum(attention_mask) - 1, then slice to current input length
            let position_ids_array;
            let position_ids_tensor;
            if decoder_input_names.contains(&"position_ids".to_string()) {
                let input_seq_len = decoder_input_ids.shape()[1];
                let total_seq_len = attention_mask.shape()[1];

                // Compute position_ids from attention_mask cumsum, then take last input_seq_len positions
                // This handles both first step (full sequence) and subsequent steps (single token with KV-cache)
                #[allow(clippy::cast_possible_wrap)]
                let positions: Vec<i64> = (0..total_seq_len)
                    .map(|i| i as i64)
                    .skip(total_seq_len - input_seq_len)
                    .collect();

                position_ids_array = Array2::from_shape_vec((1, input_seq_len), positions)
                    .map_err(|e| {
                        TaskError::Inference(format!("Failed to create position_ids array: {e}"))
                    })?
                    .into_dyn();
                position_ids_tensor =
                    TensorRef::from_array_view(&position_ids_array).map_err(|e| {
                        TaskError::Inference(format!("Failed to create position_ids tensor: {e}"))
                    })?;
                inputs.push(("position_ids", position_ids_tensor.into_dyn().into()));
            }

            // Add encoder outputs for encoder-decoder models
            let enc_tensor;
            let enc_mask_tensor;
            let enc_mask_dyn;
            if let Some(ref enc_hidden) = encoder_hidden_states {
                enc_tensor = TensorRef::from_array_view(enc_hidden).map_err(|e| {
                    TaskError::Inference(format!("Failed to create encoder hidden tensor: {e}"))
                })?;

                let enc_name = if decoder_input_names.contains(&"encoder_hidden_states".to_string())
                {
                    "encoder_hidden_states"
                } else {
                    "encoder_outputs"
                };
                inputs.push((enc_name, enc_tensor.into_dyn().into()));
            }

            // Add encoder_attention_mask only if the model expects it
            if let Some(ref enc_mask) = encoder_attention_mask {
                if decoder_input_names.contains(&"encoder_attention_mask".to_string()) {
                    enc_mask_dyn = enc_mask.clone().into_dyn();
                    enc_mask_tensor = TensorRef::from_array_view(&enc_mask_dyn).map_err(|e| {
                        TaskError::Inference(format!("Failed to create encoder mask tensor: {e}"))
                    })?;
                    inputs.push(("encoder_attention_mask", enc_mask_tensor.into_dyn().into()));
                }
            }

            // Add use_cache_branch input for merged decoder models
            // Model expects shape [1] (rank 1), not scalar (rank 0)
            let use_cache_array;
            let use_cache_tensor;
            if has_kv_cache && decoder_input_names.contains(&"use_cache_branch".to_string()) {
                use_cache_array = ndarray::arr1(&[use_cache]).into_dyn();
                use_cache_tensor = TensorRef::from_array_view(&use_cache_array).map_err(|e| {
                    TaskError::Inference(format!("Failed to create use_cache_branch tensor: {e}"))
                })?;
                inputs.push(("use_cache_branch", use_cache_tensor.into_dyn().into()));
            }

            // Add KV-cache inputs using pre-extracted metadata
            //
            // IMPORTANT: Following Optimum's behavior for merged decoder models:
            // - On first step (use_cache_branch=false): pass DUMMY tensors with seq_len=0
            //   The model's If node ignores these when use_cache_branch=false
            // - On subsequent steps (use_cache_branch=true): pass actual cache
            //
            // Shape depends on model:
            // - 4D: [batch, num_heads, past_seq_len, head_dim]
            // - 3D: [num_heads, past_seq_len, head_dim] (batch=1 implied)
            let mut kv_cache_arrays: Vec<ArrayD<f32>> = Vec::new();
            let mut kv_cache_names: Vec<String> = Vec::new();

            for kv_info in &kv_cache_inputs {
                let cache_array: ArrayD<f32> = if let Some(ref kv) = past_key_values {
                    // Use cache from previous step
                    kv.get(&kv_info.name).cloned().ok_or_else(|| {
                        TaskError::Inference(format!("Missing KV-cache for {}", kv_info.name))
                    })?
                } else {
                    // First step: create DUMMY cache with seq_len=0 (matching Optimum)
                    // The model's If node will ignore these when use_cache_branch=false.
                    // Using seq_len=0 is correct because Optimum does the same:
                    // https://github.com/huggingface/optimum/blob/main/optimum/onnxruntime/modeling_seq2seq.py#L656-658
                    if kv_info.ndim == 4 {
                        Array4::<f32>::zeros((1, kv_info.num_heads, 0, kv_info.head_dim)).into_dyn()
                    } else {
                        // 3D: [num_heads, seq_len, head_dim]
                        Array3::<f32>::zeros((kv_info.num_heads, 0, kv_info.head_dim)).into_dyn()
                    }
                };

                kv_cache_arrays.push(cache_array);
                kv_cache_names.push(kv_info.name.clone());
            }

            // Create tensors from the stored arrays and add to inputs
            let kv_cache_tensors: Vec<TensorRef<'_, f32>> = kv_cache_arrays
                .iter()
                .map(TensorRef::from_array_view)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    TaskError::Inference(format!("Failed to create KV-cache tensor: {e}"))
                })?;

            // Add KV tensors to inputs
            for (tensor, name) in kv_cache_tensors.into_iter().zip(kv_cache_names.iter()) {
                inputs.push((name.as_str(), tensor.into_dyn().into()));
            }

            // Run the session
            let mut session = decoder.blocking_lock();

            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("Decoder failed: {e}")))?;

            // Get logits (next token probabilities)
            let logits = outputs
                .get("logits")
                .or_else(|| outputs.get("lm_logits"))
                .ok_or_else(|| TaskError::Inference("No decoder logits found".into()))?;

            let logits_array = logits
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract logits: {e}")))?;

            // Extract KV-cache outputs (present.X.key, present.X.value)
            // IMPORTANT: For encoder cross-attention cache, we REUSE the values from the first step
            // because the model outputs empty tensors for encoder PKVs when use_cache_branch=true
            // See: https://github.com/huggingface/optimum/blob/main/optimum/onnxruntime/base.py#L677-L704
            let mut new_kv_cache: HashMap<String, ArrayD<f32>> = HashMap::new();

            for output_name in &decoder_output_names {
                if output_name.starts_with("present.") {
                    // Convert present.X.key -> past_key_values.X.key for next iteration
                    let past_name = output_name.replace("present.", "past_key_values.");
                    let is_encoder_pkv = output_name.contains(".encoder.");

                    // For encoder cross-attention: reuse from previous step if available
                    // (model returns empty tensors for encoder PKVs after first step)
                    if is_encoder_pkv {
                        if let Some(ref kv) = past_key_values {
                            if let Some(prev_cache) = kv.get(&past_name) {
                                // Only reuse if previous cache is valid (non-zero dimensions)
                                if prev_cache.shape().iter().all(|&d| d > 0) {
                                    new_kv_cache.insert(past_name.clone(), prev_cache.clone());
                                    continue;
                                }
                            }
                        }
                    }

                    // For decoder self-attention (or first step encoder): use model output
                    if let Some(present_tensor) = outputs.get(output_name.as_str()) {
                        let present_array = present_tensor
                            .try_extract_array::<f32>()
                            .map(ndarray::ArrayBase::into_owned)
                            .map_err(|e| {
                                TaskError::Inference(format!(
                                    "Failed to extract {output_name}: {e}"
                                ))
                            })?;

                        let shape = present_array.shape();

                        // Skip tensors with zero-length dimensions (empty cache)
                        // These occur when the model doesn't compute certain caches
                        // (e.g., encoder cross-attention on first step with use_cache=false)
                        if shape.contains(&0) {
                            tracing::debug!(
                                output = output_name,
                                shape = ?shape,
                                "Skipping empty KV-cache tensor"
                            );
                            continue;
                        }

                        // Keep original shape (3D or 4D) - don't force conversion
                        // This allows models to use whichever format they prefer
                        if shape.len() == 3 || shape.len() == 4 {
                            new_kv_cache.insert(past_name, present_array);
                        } else {
                            tracing::warn!(
                                output = output_name,
                                shape = ?shape,
                                "Unexpected KV-cache shape (expected 3D or 4D), skipping"
                            );
                        }
                    }
                }
            }

            Ok((logits_array, new_kv_cache))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Run decoder step with inputs_embeds instead of input_ids (Florence-2 style).
    ///
    /// # Arguments
    /// * `inputs_embeds` - Token embeddings [batch, seq_len, hidden_dim]
    /// * `attention_mask` - Full attention mask (grows with each generated token)
    /// * `encoder_hidden_states` - Encoder output for cross-attention
    /// * `encoder_attention_mask` - Attention mask for encoder outputs
    /// * `past_key_values` - KV-cache from previous step (None for first step)
    ///
    /// # Returns
    /// Tuple of (logits, new_kv_cache) where new_kv_cache should be passed to next step.
    async fn run_decoder_step_with_embeds(
        &self,
        inputs_embeds: &ArrayD<f32>,
        attention_mask: &Array2<i64>,
        encoder_hidden_states: Option<&ArrayD<f32>>,
        encoder_attention_mask: Option<&Array2<i64>>,
        past_key_values: Option<&HashMap<String, ArrayD<f32>>>,
    ) -> TaskResult<(ArrayD<f32>, HashMap<String, ArrayD<f32>>)> {
        let decoder = Arc::clone(&self.decoder);
        let inputs_embeds = inputs_embeds.clone();
        let attention_mask = attention_mask.clone();
        let decoder_input_names = self.decoder_input_names.clone();
        let decoder_output_names = self.decoder_output_names.clone();
        let has_kv_cache = self.has_kv_cache;
        let kv_cache_inputs = self.kv_cache_inputs.clone();
        let past_key_values = past_key_values.cloned();
        let encoder_hidden_states = encoder_hidden_states.cloned();
        let encoder_attention_mask = encoder_attention_mask.cloned();

        tokio::task::spawn_blocking(move || {
            use ort::session::SessionInputValue;

            let use_cache = past_key_values.is_some();

            let mut inputs: Vec<(&str, SessionInputValue)> = Vec::new();

            // Add inputs_embeds
            let embeds_tensor = TensorRef::from_array_view(&inputs_embeds).map_err(|e| {
                TaskError::Inference(format!("Failed to create inputs_embeds tensor: {e}"))
            })?;
            inputs.push(("inputs_embeds", embeds_tensor.into_dyn().into()));

            // Add attention_mask
            let attn_mask_dyn = attention_mask.clone().into_dyn();
            let attn_mask_tensor;
            if decoder_input_names.contains(&"attention_mask".to_string()) {
                attn_mask_tensor = TensorRef::from_array_view(&attn_mask_dyn).map_err(|e| {
                    TaskError::Inference(format!("Failed to create attention_mask tensor: {e}"))
                })?;
                inputs.push(("attention_mask", attn_mask_tensor.into_dyn().into()));
            }

            // Add encoder_hidden_states if present (for cross-attention in encoder-decoder models)
            let enc_tensor;
            if let Some(ref enc_hidden) = encoder_hidden_states {
                if decoder_input_names.contains(&"encoder_hidden_states".to_string()) {
                    enc_tensor = TensorRef::from_array_view(enc_hidden).map_err(|e| {
                        TaskError::Inference(format!(
                            "Failed to create encoder_hidden_states tensor: {e}"
                        ))
                    })?;
                    inputs.push(("encoder_hidden_states", enc_tensor.into_dyn().into()));
                }
            }

            // Add encoder_attention_mask if model expects it
            let enc_mask_dyn;
            let enc_mask_tensor;
            if let Some(ref enc_mask) = encoder_attention_mask {
                if decoder_input_names.contains(&"encoder_attention_mask".to_string()) {
                    enc_mask_dyn = enc_mask.clone().into_dyn();
                    enc_mask_tensor = TensorRef::from_array_view(&enc_mask_dyn).map_err(|e| {
                        TaskError::Inference(format!(
                            "Failed to create encoder_attention_mask tensor: {e}"
                        ))
                    })?;
                    inputs.push(("encoder_attention_mask", enc_mask_tensor.into_dyn().into()));
                }
            }

            // Add use_cache_branch input for merged decoder models
            let use_cache_array;
            let use_cache_tensor;
            if has_kv_cache && decoder_input_names.contains(&"use_cache_branch".to_string()) {
                use_cache_array = ndarray::arr1(&[use_cache]).into_dyn();
                use_cache_tensor = TensorRef::from_array_view(&use_cache_array).map_err(|e| {
                    TaskError::Inference(format!("Failed to create use_cache_branch tensor: {e}"))
                })?;
                inputs.push(("use_cache_branch", use_cache_tensor.into_dyn().into()));
            }

            // Add KV-cache inputs (with seq_len=0 matching Optimum)
            let mut kv_cache_arrays: Vec<ArrayD<f32>> = Vec::new();
            let mut kv_cache_names: Vec<String> = Vec::new();

            for kv_info in &kv_cache_inputs {
                let cache_array: ArrayD<f32> = if let Some(ref kv) = past_key_values {
                    // Subsequent steps: use cache from previous step
                    kv.get(&kv_info.name).cloned().ok_or_else(|| {
                        TaskError::Inference(format!("Missing KV-cache for {}", kv_info.name))
                    })?
                } else {
                    // First step: handle differently based on cache type
                    match kv_info.cache_type {
                        KvCacheType::DecoderSelfAttention => {
                            // Decoder self-attention: dummy cache with seq_len=0 (matching Optimum)
                            if kv_info.ndim == 4 {
                                Array4::<f32>::zeros((1, kv_info.num_heads, 0, kv_info.head_dim))
                                    .into_dyn()
                            } else {
                                Array3::<f32>::zeros((kv_info.num_heads, 0, kv_info.head_dim))
                                    .into_dyn()
                            }
                        }
                        KvCacheType::EncoderCrossAttention => {
                            // Encoder cross-attention: also pass empty tensors on first step
                            // The model will compute these from encoder_hidden_states
                            // and output them in present.*.encoder.* for subsequent steps
                            if kv_info.ndim == 4 {
                                Array4::<f32>::zeros((1, kv_info.num_heads, 0, kv_info.head_dim))
                                    .into_dyn()
                            } else {
                                Array3::<f32>::zeros((kv_info.num_heads, 0, kv_info.head_dim))
                                    .into_dyn()
                            }
                        }
                    }
                };

                kv_cache_arrays.push(cache_array);
                kv_cache_names.push(kv_info.name.clone());
            }

            let kv_cache_tensors: Vec<TensorRef<'_, f32>> = kv_cache_arrays
                .iter()
                .map(TensorRef::from_array_view)
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| {
                    TaskError::Inference(format!("Failed to create KV-cache tensor: {e}"))
                })?;

            for (tensor, name) in kv_cache_tensors.into_iter().zip(kv_cache_names.iter()) {
                inputs.push((name.as_str(), tensor.into_dyn().into()));
            }

            // Run the session
            let mut session = decoder.blocking_lock();

            let outputs = session
                .run(inputs)
                .map_err(|e| TaskError::Inference(format!("Decoder failed: {e}")))?;

            // Get logits
            let logits = outputs
                .get("logits")
                .or_else(|| outputs.get("lm_logits"))
                .ok_or_else(|| TaskError::Inference("No decoder logits found".into()))?;

            let logits_array = logits
                .try_extract_array::<f32>()
                .map(ndarray::ArrayBase::into_owned)
                .map_err(|e| TaskError::Inference(format!("Failed to extract logits: {e}")))?;

            // Extract KV-cache outputs (present.X.key, present.X.value)
            // IMPORTANT: For encoder cross-attention cache, we REUSE the values from the first step
            // because the model outputs empty tensors for encoder PKVs when use_cache_branch=true
            let mut new_kv_cache: HashMap<String, ArrayD<f32>> = HashMap::new();

            for output_name in &decoder_output_names {
                if output_name.starts_with("present.") {
                    // Convert present.X.key -> past_key_values.X.key for next iteration
                    let past_name = output_name.replace("present.", "past_key_values.");
                    let is_encoder_pkv = output_name.contains(".encoder.");

                    // For encoder cross-attention: reuse from previous step if available
                    // (model returns empty tensors for encoder PKVs after first step)
                    if is_encoder_pkv {
                        if let Some(ref kv) = past_key_values {
                            if let Some(prev_cache) = kv.get(&past_name) {
                                // Only reuse if previous cache is valid (non-zero dimensions)
                                if prev_cache.shape().iter().all(|&d| d > 0) {
                                    new_kv_cache.insert(past_name.clone(), prev_cache.clone());
                                    continue;
                                }
                            }
                        }
                    }

                    // For decoder self-attention (or first step encoder): use model output
                    if let Some(present_tensor) = outputs.get(output_name.as_str()) {
                        let present_array = present_tensor
                            .try_extract_array::<f32>()
                            .map(ndarray::ArrayBase::into_owned)
                            .map_err(|e| {
                                TaskError::Inference(format!(
                                    "Failed to extract {output_name}: {e}"
                                ))
                            })?;

                        let shape = present_array.shape();

                        // Skip tensors with zero-length dimensions (empty cache)
                        if shape.contains(&0) {
                            tracing::debug!(
                                output = output_name,
                                shape = ?shape,
                                "Skipping empty KV-cache tensor"
                            );
                            continue;
                        }

                        // Keep original shape (3D or 4D)
                        if shape.len() == 3 || shape.len() == 4 {
                            new_kv_cache.insert(past_name, present_array);
                        }
                    }
                }
            }

            Ok((logits_array, new_kv_cache))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))?
    }

    /// Sample next token from logits using temperature and top-p sampling.
    #[allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap
    )]
    fn sample_next_token(&self, logits: &[f32], generated_tokens: &[i64]) -> i64 {
        let mut logits = logits.to_vec();

        // Apply repetition penalty (skip if penalty is essentially 1.0)
        if (self.gen_config.repetition_penalty - 1.0).abs() > f32::EPSILON {
            for &token in generated_tokens {
                // Safe: token IDs from generation are always valid indices
                if let Some(logit) = logits.get_mut(token as usize) {
                    if *logit > 0.0 {
                        *logit /= self.gen_config.repetition_penalty;
                    } else {
                        *logit *= self.gen_config.repetition_penalty;
                    }
                }
            }
        }

        // Apply temperature (skip if temperature is essentially 1.0)
        if (self.gen_config.temperature - 1.0).abs() > f32::EPSILON
            && self.gen_config.temperature > 0.0
        {
            for logit in &mut logits {
                *logit /= self.gen_config.temperature;
            }
        }

        // Apply softmax
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let exp_sum: f32 = logits.iter().map(|&x| (x - max_logit).exp()).sum();
        let probs: Vec<f32> = logits
            .iter()
            .map(|&x| (x - max_logit).exp() / exp_sum)
            .collect();

        // Top-k filtering
        let mut indexed_probs: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        if self.gen_config.top_k > 0 {
            indexed_probs.truncate(self.gen_config.top_k);
        }

        // Top-p (nucleus) sampling
        let mut cumsum = 0.0;
        let mut cutoff_idx = indexed_probs.len();
        for (i, (_, p)) in indexed_probs.iter().enumerate() {
            cumsum += p;
            if cumsum >= self.gen_config.top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        indexed_probs.truncate(cutoff_idx);

        // Renormalize
        let total: f32 = indexed_probs.iter().map(|(_, p)| p).sum();
        for (_, p) in &mut indexed_probs {
            *p /= total;
        }

        // Sample from the distribution
        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen();
        let mut cumsum = 0.0;
        for (idx, p) in &indexed_probs {
            cumsum += p;
            if r < cumsum {
                // Safe: vocab indices fit in i64
                return *idx as i64;
            }
        }

        // Fallback to most likely token
        // Safe: vocab indices fit in i64
        indexed_probs.first().map_or(0, |(idx, _)| *idx as i64)
    }

    /// Generate text autoregressively.
    pub async fn generate(
        &self,
        input_ids: Array2<i64>,
        attention_mask: Array2<i64>,
    ) -> TaskResult<Vec<i64>> {
        let batch_size = input_ids.shape()[0];
        if batch_size != 1 {
            return Err(TaskError::InvalidInput(
                "Batch generation not yet supported".into(),
            ));
        }

        let mut generated_tokens: Vec<i64> = Vec::new();
        let encoder_hidden_states: Option<ArrayD<f32>>;
        let mut decoder_input_ids: Array2<i64>;
        let mut current_attention_mask: Array2<i64>;
        let mut kv_cache: Option<HashMap<String, ArrayD<f32>>> = None;

        match self.architecture {
            ModelArchitecture::EncoderDecoder => {
                // Run encoder once
                encoder_hidden_states = Some(self.run_encoder(&input_ids, &attention_mask).await?);
                // Start decoder with start token
                decoder_input_ids =
                    Array2::from_elem((1, 1), self.gen_config.decoder_start_token_id);
                current_attention_mask = Array2::ones((1, 1));
            }
            ModelArchitecture::DecoderOnly => {
                encoder_hidden_states = None;
                // Start with input tokens
                decoder_input_ids = input_ids.clone();
                current_attention_mask = attention_mask.clone();
            }
        }

        // Autoregressive generation loop
        // Calculate initial sequence length for position tracking
        let initial_seq_len = decoder_input_ids.shape()[1];
        let max_generation_steps = self
            .max_position_embeddings
            .saturating_sub(initial_seq_len)
            .min(self.gen_config.max_new_tokens);

        for step in 0..max_generation_steps {
            // Check if next token would exceed position limit
            let current_position = initial_seq_len + step;
            if current_position >= self.max_position_embeddings {
                debug!(
                    step = step,
                    current_position = current_position,
                    max_position_embeddings = self.max_position_embeddings,
                    "Stopping generation: would exceed max position embeddings"
                );
                break;
            }

            // Run decoder with KV-cache
            let (logits, new_kv_cache) = self
                .run_decoder_step(
                    &decoder_input_ids,
                    &current_attention_mask,
                    encoder_hidden_states.as_ref(),
                    if self.architecture == ModelArchitecture::EncoderDecoder {
                        Some(&attention_mask)
                    } else {
                        None
                    },
                    kv_cache.as_ref(),
                )
                .await?;

            // Get logits for last position
            let logits_shape = logits.shape();
            let seq_len = logits_shape[1];
            let vocab_size = logits_shape[2];

            // Extract last position logits
            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[0, seq_len - 1, v]])
                .collect();

            // Sample next token
            let next_token = self.sample_next_token(&last_logits, &generated_tokens);

            // Check for EOS
            if self.gen_config.eos_token_ids.contains(&next_token) {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_tokens.push(next_token);

            // Update for next step
            if self.has_kv_cache && !new_kv_cache.is_empty() {
                // With KV-cache: only pass the new token
                decoder_input_ids = Array2::from_elem((1, 1), next_token);
                kv_cache = Some(new_kv_cache);

                // Extend attention mask by 1
                let current_len = current_attention_mask.shape()[1];
                let mut new_mask = Array2::ones((1, current_len + 1));
                for i in 0..current_len {
                    new_mask[[0, i]] = current_attention_mask[[0, i]];
                }
                current_attention_mask = new_mask;
            } else {
                // Without KV-cache: pass full sequence
                let current_len = decoder_input_ids.shape()[1];
                let mut new_decoder_input = Array2::zeros((1, current_len + 1));
                for i in 0..current_len {
                    new_decoder_input[[0, i]] = decoder_input_ids[[0, i]];
                }
                new_decoder_input[[0, current_len]] = next_token;
                decoder_input_ids = new_decoder_input;

                // Extend attention mask
                let mask_len = current_attention_mask.shape()[1];
                let mut new_mask = Array2::ones((1, mask_len + 1));
                for i in 0..mask_len {
                    new_mask[[0, i]] = current_attention_mask[[0, i]];
                }
                current_attention_mask = new_mask;
            }

            if step % 50 == 0 {
                debug!(
                    step = step,
                    tokens = generated_tokens.len(),
                    using_kv_cache = kv_cache.is_some(),
                    "Generation progress"
                );
            }
        }

        Ok(generated_tokens)
    }

    /// Generate text from pre-computed encoder hidden states.
    ///
    /// This is used for multimodal inputs (audio, image) where the encoder
    /// processes non-text inputs (mel spectrograms, pixel values) and produces
    /// hidden states that feed into the decoder.
    ///
    /// For encoder-decoder models like Whisper (ASR) or TrOCR (image-to-text):
    /// 1. Encoder processes input_features/pixel_values -> encoder_hidden_states
    /// 2. Decoder generates tokens conditioned on encoder_hidden_states
    pub async fn generate_from_encoder_output(
        &self,
        encoder_hidden_states: ArrayD<f32>,
    ) -> TaskResult<Vec<i64>> {
        if self.architecture != ModelArchitecture::EncoderDecoder {
            return Err(TaskError::Config(
                "generate_from_encoder_output requires encoder-decoder model".into(),
            ));
        }

        let mut generated_tokens: Vec<i64> = Vec::new();
        let mut decoder_input_ids =
            Array2::from_elem((1, 1), self.gen_config.decoder_start_token_id);
        let mut current_attention_mask = Array2::ones((1, 1));
        let mut kv_cache: Option<HashMap<String, ArrayD<f32>>> = None;

        // Create a dummy encoder attention mask based on encoder hidden states shape
        // Shape: [batch, seq_len, hidden_dim] -> mask: [batch, seq_len]
        let encoder_seq_len = encoder_hidden_states.shape()[1];
        let encoder_attention_mask = Array2::ones((1, encoder_seq_len));

        // Autoregressive generation loop
        // Initial sequence length is 1 (decoder_start_token_id)
        let initial_seq_len = 1usize;
        let max_generation_steps = self
            .max_position_embeddings
            .saturating_sub(initial_seq_len)
            .min(self.gen_config.max_new_tokens);

        for step in 0..max_generation_steps {
            // Check if next token would exceed position limit
            let current_position = initial_seq_len + step;
            if current_position >= self.max_position_embeddings {
                debug!(
                    step = step,
                    current_position = current_position,
                    max_position_embeddings = self.max_position_embeddings,
                    "Stopping generation: would exceed max position embeddings"
                );
                break;
            }

            let (logits, new_kv_cache) = self
                .run_decoder_step(
                    &decoder_input_ids,
                    &current_attention_mask,
                    Some(&encoder_hidden_states),
                    Some(&encoder_attention_mask),
                    kv_cache.as_ref(),
                )
                .await?;

            // Get logits for last position
            let logits_shape = logits.shape();
            let seq_len = logits_shape[1];
            let vocab_size = logits_shape[2];

            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[0, seq_len - 1, v]])
                .collect();

            let next_token = self.sample_next_token(&last_logits, &generated_tokens);

            if self.gen_config.eos_token_ids.contains(&next_token) {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_tokens.push(next_token);

            // Update for next step
            if self.has_kv_cache && !new_kv_cache.is_empty() {
                decoder_input_ids = Array2::from_elem((1, 1), next_token);
                kv_cache = Some(new_kv_cache);

                let current_len = current_attention_mask.shape()[1];
                let mut new_mask = Array2::ones((1, current_len + 1));
                for i in 0..current_len {
                    new_mask[[0, i]] = current_attention_mask[[0, i]];
                }
                current_attention_mask = new_mask;
            } else {
                let current_len = decoder_input_ids.shape()[1];
                let mut new_decoder_input = Array2::zeros((1, current_len + 1));
                for i in 0..current_len {
                    new_decoder_input[[0, i]] = decoder_input_ids[[0, i]];
                }
                new_decoder_input[[0, current_len]] = next_token;
                decoder_input_ids = new_decoder_input;

                let mask_len = current_attention_mask.shape()[1];
                let mut new_mask = Array2::ones((1, mask_len + 1));
                for i in 0..mask_len {
                    new_mask[[0, i]] = current_attention_mask[[0, i]];
                }
                current_attention_mask = new_mask;
            }

            if step % 50 == 0 {
                debug!(
                    step = step,
                    tokens = generated_tokens.len(),
                    using_kv_cache = kv_cache.is_some(),
                    "Generation progress (multimodal)"
                );
            }
        }

        Ok(generated_tokens)
    }

    /// Generate text from image features using embed-based decoder (Florence-2 style).
    ///
    /// This is used for vision-language models like Florence-2 where:
    /// 1. Vision encoder produces `image_features`
    /// 2. Text tokens are converted to embeddings via `embed_tokens`
    /// 3. Image features and text embeddings are concatenated for inputs_embeds
    /// 4. Decoder takes `inputs_embeds` AND `encoder_hidden_states` (image_features for cross-attention)
    ///
    /// # Arguments
    /// * `image_features` - Output from vision encoder [batch, num_patches, hidden_dim]
    /// * `prompt_ids` - Optional text prompt token IDs (for conditional generation)
    pub async fn generate_with_embeds(
        &self,
        image_features: ArrayD<f32>,
        prompt_ids: Option<&Array2<i64>>,
    ) -> TaskResult<Vec<i64>> {
        if !self.decoder_uses_embeds {
            return Err(TaskError::Config(
                "generate_with_embeds requires a decoder that uses inputs_embeds".into(),
            ));
        }

        if self.embed_tokens.is_none() {
            return Err(TaskError::Config(
                "generate_with_embeds requires embed_tokens model".into(),
            ));
        }

        let mut generated_tokens: Vec<i64> = Vec::new();

        // For Florence-2 style models:
        // - inputs_embeds: only text token embeddings (NOT image features)
        // - encoder_hidden_states: image features (for cross-attention)
        // This keeps the decoder sequence length manageable (max ~1024 positions)
        let start_token_ids = if let Some(prompt) = prompt_ids {
            // Append decoder_start_token_id to prompt
            let prompt_len = prompt.shape()[1];
            let mut ids = Array2::zeros((1, prompt_len + 1));
            for i in 0..prompt_len {
                ids[[0, i]] = prompt[[0, i]];
            }
            ids[[0, prompt_len]] = self.gen_config.decoder_start_token_id;
            ids
        } else {
            // Just the start token
            Array2::from_elem((1, 1), self.gen_config.decoder_start_token_id)
        };

        // Get text embeddings (this is what goes to inputs_embeds)
        let text_embeds = self.run_embed_tokens(&start_token_ids).await?;

        // Use ONLY text embeddings for inputs_embeds (NOT concatenated with image features)
        // Image features go to encoder_hidden_states for cross-attention
        let initial_embeds = text_embeds;

        // Initial attention mask covers only the text tokens
        let initial_seq_len = initial_embeds.shape()[1];
        let mut current_attention_mask = Array2::ones((1, initial_seq_len));

        // Encoder attention mask covers the image features (for cross-attention)
        // This mask tells the decoder which encoder outputs to attend to
        let encoder_seq_len = image_features.shape()[1];
        let encoder_attention_mask: Array2<i64> = Array2::ones((1, encoder_seq_len));

        // First decoder step with text embeddings only
        // Pass image_features as encoder_hidden_states for cross-attention
        let (logits, new_kv_cache) = self
            .run_decoder_step_with_embeds(
                &initial_embeds,
                &current_attention_mask,
                Some(&image_features),
                Some(&encoder_attention_mask),
                None,
            )
            .await?;

        // Sample first token
        let logits_shape = logits.shape();
        let seq_len = logits_shape[1];
        let vocab_size = logits_shape[2];
        let last_logits: Vec<f32> = (0..vocab_size)
            .map(|v| logits[[0, seq_len - 1, v]])
            .collect();

        let next_token = self.sample_next_token(&last_logits, &generated_tokens);

        if self.gen_config.eos_token_ids.contains(&next_token) {
            debug!("EOS token generated at step 0");
            return Ok(generated_tokens);
        }

        generated_tokens.push(next_token);
        let mut kv_cache = Some(new_kv_cache);

        // Extend attention mask
        let current_len = current_attention_mask.shape()[1];
        let mut new_mask = Array2::ones((1, current_len + 1));
        for i in 0..current_len {
            new_mask[[0, i]] = current_attention_mask[[0, i]];
        }
        current_attention_mask = new_mask;

        // Continue autoregressive generation
        // Limit generation to avoid exceeding max_position_embeddings
        // Current position = initial_seq_len + generated tokens
        let max_generation_steps = self
            .max_position_embeddings
            .saturating_sub(initial_seq_len)
            .min(self.gen_config.max_new_tokens);

        if max_generation_steps == 0 {
            debug!(
                initial_seq_len = initial_seq_len,
                max_position_embeddings = self.max_position_embeddings,
                "Prompt already at max position, cannot generate"
            );
            return Ok(generated_tokens);
        }

        for step in 1..max_generation_steps {
            // Check if next token would exceed position limit
            let next_position = initial_seq_len + step;
            if next_position >= self.max_position_embeddings {
                debug!(
                    step = step,
                    next_position = next_position,
                    max_position_embeddings = self.max_position_embeddings,
                    "Stopping generation: would exceed max position embeddings"
                );
                break;
            }

            // Get embedding for the new token
            let new_token_ids = Array2::from_elem((1, 1), next_token);
            let new_token_embed = self.run_embed_tokens(&new_token_ids).await?;

            // Run decoder step with just the new token embedding (KV-cache handles history)
            // Continue passing image_features as encoder_hidden_states for cross-attention
            let (logits, new_kv_cache) = self
                .run_decoder_step_with_embeds(
                    &new_token_embed,
                    &current_attention_mask,
                    Some(&image_features),
                    Some(&encoder_attention_mask),
                    kv_cache.as_ref(),
                )
                .await?;

            // Sample next token
            let logits_shape = logits.shape();
            let seq_len = logits_shape[1];
            let vocab_size = logits_shape[2];
            let last_logits: Vec<f32> = (0..vocab_size)
                .map(|v| logits[[0, seq_len - 1, v]])
                .collect();

            let next_token = self.sample_next_token(&last_logits, &generated_tokens);

            if self.gen_config.eos_token_ids.contains(&next_token) {
                debug!(step = step, "EOS token generated");
                break;
            }

            generated_tokens.push(next_token);
            kv_cache = Some(new_kv_cache);

            // Extend attention mask
            let current_len = current_attention_mask.shape()[1];
            let mut new_mask = Array2::ones((1, current_len + 1));
            for i in 0..current_len {
                new_mask[[0, i]] = current_attention_mask[[0, i]];
            }
            current_attention_mask = new_mask;

            if step % 50 == 0 {
                debug!(
                    step = step,
                    tokens = generated_tokens.len(),
                    position = initial_seq_len + step,
                    max_positions = self.max_position_embeddings,
                    "Generation progress (Florence-2 style)"
                );
            }
        }

        Ok(generated_tokens)
    }
}

/// Input for seq2seq tasks.
/// Supports text (NLP), audio (ASR), or image (image-to-text) inputs.
#[derive(Debug, Deserialize)]
pub struct Seq2SeqInput {
    /// Input text for NLP tasks (text-generation, translation, summarization)
    pub text: Option<String>,
    /// Input audio for ASR tasks (automatic-speech-recognition)
    pub audio: Option<String>,
    /// Input image for vision tasks (image-to-text)
    pub image: Option<String>,
    /// Optional maximum new tokens to generate (overrides config)
    pub max_new_tokens: Option<usize>,
    /// Optional temperature (overrides config)
    pub temperature: Option<f32>,
}

/// Output from seq2seq tasks.
#[derive(Debug, Serialize)]
pub struct Seq2SeqOutput {
    /// Generated text
    pub text: String,
    /// Number of tokens generated
    pub num_tokens: usize,
}

#[async_trait]
impl Task for Seq2SeqTask {
    fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "preprocess")]
    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "Seq2Seq task executing");

        // Parse input to validate structure (text, audio, or image)
        let input: Seq2SeqInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => {
                return GrpcTaskResult::err(format!("Invalid input: {e}"));
            }
        };

        // Validate that at least one input type is provided
        if input.text.is_none() && input.audio.is_none() && input.image.is_none() {
            return GrpcTaskResult::err(
                "Missing input: provide 'text', 'audio', or 'image' field".into(),
            );
        }

        // Preprocess input to get model inputs
        let preprocessed = match self.preprocessor.process_json(payload).await {
            Ok(p) => p,
            Err(e) => {
                return GrpcTaskResult::err(format!("Preprocessing failed: {e}"));
            }
        };

        // Extract input tensors from preprocessed output
        // For NLP: input_ids + attention_mask
        // For ASR: input_features (mel spectrogram)
        // For image-to-text: pixel_values
        let inputs = &preprocessed.inputs;

        // Route based on input type
        let generated_ids: Vec<i64> = if let Some(v) = inputs.get("input_features") {
            // ASR task: audio input (mel spectrogram)
            debug!("Processing audio input (input_features)");

            let input_features_dyn = match json_to_array_f32(v) {
                Ok(arr) => arr,
                Err(e) => return GrpcTaskResult::err(format!("Invalid input_features: {e}")),
            };

            // Reshape from dynamic to 3D: [batch, n_mels, time_frames]
            let shape = input_features_dyn.shape();
            if shape.len() != 3 {
                return GrpcTaskResult::err(format!(
                    "input_features must be 3D [batch, n_mels, time], got shape: {shape:?}"
                ));
            }
            // Copy shape dimensions before moving the array
            let (d0, d1, d2) = (shape[0], shape[1], shape[2]);
            let input_features = match input_features_dyn.into_shape_with_order((d0, d1, d2)) {
                Ok(arr) => arr,
                Err(e) => {
                    return GrpcTaskResult::err(format!("Failed to reshape input_features: {e}"))
                }
            };

            // Run encoder with audio features
            let encoder_hidden_states = match self.run_encoder_audio(&input_features).await {
                Ok(h) => h,
                Err(e) => return GrpcTaskResult::err(format!("Encoder failed: {e}")),
            };

            // Generate from encoder output
            match self
                .generate_from_encoder_output(encoder_hidden_states)
                .await
            {
                Ok(ids) => ids,
                Err(e) => return GrpcTaskResult::err(format!("Generation failed: {e}")),
            }
        } else if let Some(v) = inputs.get("pixel_values") {
            // Image-to-text task: image input (pixel values)
            debug!("Processing image input (pixel_values)");

            let pixel_values_dyn = match json_to_array_f32(v) {
                Ok(arr) => arr,
                Err(e) => return GrpcTaskResult::err(format!("Invalid pixel_values: {e}")),
            };

            // Reshape from dynamic to 4D: [batch, channels, height, width]
            let shape = pixel_values_dyn.shape();
            if shape.len() != 4 {
                return GrpcTaskResult::err(format!(
                    "pixel_values must be 4D [batch, channels, height, width], got shape: {shape:?}"
                ));
            }
            // Copy shape dimensions before moving the array
            let (d0, d1, d2, d3) = (shape[0], shape[1], shape[2], shape[3]);
            let pixel_values = match pixel_values_dyn.into_shape_with_order((d0, d1, d2, d3)) {
                Ok(arr) => arr,
                Err(e) => {
                    return GrpcTaskResult::err(format!("Failed to reshape pixel_values: {e}"))
                }
            };

            // Run encoder with image features
            let encoder_hidden_states = match self.run_encoder_image(&pixel_values).await {
                Ok(h) => h,
                Err(e) => return GrpcTaskResult::err(format!("Encoder failed: {e}")),
            };

            // Check if this is a Florence-2 style model (decoder uses inputs_embeds)
            if self.decoder_uses_embeds {
                debug!("Using Florence-2 style generation with inputs_embeds");

                // Check if there's an optional text prompt (input_ids) for conditional generation
                let prompt_ids: Option<Array2<i64>> =
                    if let Some(prompt_v) = inputs.get("input_ids") {
                        match json_to_array2_i64(prompt_v) {
                            Ok(arr) => Some(arr),
                            Err(e) => {
                                debug!(error = %e, "Failed to parse prompt input_ids, ignoring");
                                None
                            }
                        }
                    } else {
                        None
                    };

                // Generate using embed-based decoder
                match self
                    .generate_with_embeds(encoder_hidden_states, prompt_ids.as_ref())
                    .await
                {
                    Ok(ids) => ids,
                    Err(e) => return GrpcTaskResult::err(format!("Generation failed: {e}")),
                }
            } else {
                // Standard encoder-decoder generation (TrOCR, etc.)
                match self
                    .generate_from_encoder_output(encoder_hidden_states)
                    .await
                {
                    Ok(ids) => ids,
                    Err(e) => return GrpcTaskResult::err(format!("Generation failed: {e}")),
                }
            }
        } else if let Some(v) = inputs.get("input_ids") {
            // NLP task: text input (tokenized)
            debug!("Processing text input (input_ids)");

            let input_ids = match json_to_array2_i64(v) {
                Ok(arr) => arr,
                Err(e) => return GrpcTaskResult::err(format!("Invalid input_ids: {e}")),
            };
            let attention_mask = match inputs.get("attention_mask") {
                Some(v) => match json_to_array2_i64(v) {
                    Ok(arr) => arr,
                    Err(e) => return GrpcTaskResult::err(format!("Invalid attention_mask: {e}")),
                },
                None => Array2::ones(input_ids.raw_dim()),
            };

            // Generate with text input
            match self.generate(input_ids, attention_mask).await {
                Ok(ids) => ids,
                Err(e) => return GrpcTaskResult::err(format!("Generation failed: {e}")),
            }
        } else {
            return GrpcTaskResult::err(
                "Missing input_ids, input_features, or pixel_values from preprocessing".into(),
            );
        };

        // Decode generated token IDs back to text
        let generated_text = match self.preprocessor.decode(&generated_ids, true) {
            Ok(text) => text,
            Err(e) => {
                // Fall back to returning just token IDs if decoding fails
                debug!(error = %e, "Token decoding failed, returning IDs only");
                String::new()
            }
        };

        let output = serde_json::json!({
            "text": generated_text,
            "generated_ids": generated_ids,
            "num_tokens": generated_ids.len(),
        });

        match serde_json::to_string(&output) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
        }
    }

    fn is_ready(&self) -> bool {
        true
    }
}

impl std::fmt::Debug for Seq2SeqTask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Seq2SeqTask")
            .field("name", &self.name)
            .field("architecture", &self.architecture)
            .field("has_kv_cache", &self.has_kv_cache)
            .finish_non_exhaustive()
    }
}
