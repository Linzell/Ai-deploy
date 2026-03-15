//! Image-to-text (image captioning) via BLIP.
//!
//! Wraps `candle-transformers`' `BlipForConditionalGeneration` to provide
//! native Candle inference for `Salesforce/blip-image-captioning-*` models.
//!
//! ## Input format
//!
//! Base64-encoded image (JPEG/PNG):
//! ```json
//! {"inputs": "<base64-encoded image>"}
//! ```
//!
//! Or with a conditional text prompt:
//! ```json
//! {"inputs": "<base64-encoded image>", "prompt": "a photo of"}
//! ```
//!
//! ## Output format
//!
//! Returns generated caption:
//! ```json
//! [{"generated_text": "a cat sitting on a couch"}]
//! ```

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use candle_transformers::models::blip;
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Mutex;
use tracing::{debug, info};

use crate::error::{TaskError, TaskResult};
use crate::utils;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// BOS token for BLIP text decoder (start of generation).
const BOS_TOKEN_ID: u32 = 30522;

/// SEP token — stop generation when produced.
const SEP_TOKEN_ID: u32 = 102;

/// Maximum number of tokens to generate.
const MAX_GEN_TOKENS: usize = 512;

/// Image normalization constants (OpenAI CLIP-style).
const IMAGE_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
const IMAGE_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

/// Input for image-to-text captioning.
#[derive(Debug, Deserialize)]
pub struct ImageToTextInput {
    /// Base64-encoded image (JPEG/PNG) or data URI.
    pub inputs: Option<serde_json::Value>,
    /// Alternative field name for the image.
    pub image: Option<serde_json::Value>,
    /// Optional conditional prompt (e.g., "a photo of").
    pub prompt: Option<String>,
    /// Maximum tokens to generate (default 512).
    pub max_new_tokens: Option<usize>,
}

/// Single captioning result in HF Inference API format.
#[derive(Debug, Serialize)]
pub struct ImageToTextOutput {
    pub generated_text: String,
}

// ---------------------------------------------------------------------------
// Config parsing helpers
// ---------------------------------------------------------------------------

/// Intermediate struct for parsing the HF config.json into `blip::Config`.
///
/// The HF config doesn't include `encoder_hidden_size` inside `text_config` —
/// it must be inferred from `vision_config.hidden_size`.
#[derive(Debug, Deserialize)]
struct HfBlipConfig {
    #[serde(default)]
    text_config: HfBlipTextConfig,
    #[serde(default)]
    vision_config: HfBlipVisionConfig,
    #[serde(default = "default_projection_dim")]
    projection_dim: usize,
    #[serde(default = "default_image_text_hidden_size")]
    image_text_hidden_size: usize,
}

fn default_projection_dim() -> usize {
    512
}

fn default_image_text_hidden_size() -> usize {
    256
}

#[derive(Debug, Deserialize)]
struct HfBlipVisionConfig {
    #[serde(default = "default_vision_hidden")]
    hidden_size: usize,
    #[serde(default = "default_vision_intermediate")]
    intermediate_size: usize,
    #[serde(default = "default_projection_dim")]
    projection_dim: usize,
    #[serde(default = "default_vision_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_vision_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_image_size")]
    image_size: usize,
    #[serde(default = "default_patch_size")]
    patch_size: usize,
    #[serde(default = "default_layer_norm_eps_vision")]
    layer_norm_eps: f64,
}

impl Default for HfBlipVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 512,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            image_size: 384,
            patch_size: 16,
            layer_norm_eps: 1e-5,
        }
    }
}

fn default_vision_hidden() -> usize {
    768
}
fn default_vision_intermediate() -> usize {
    3072
}
fn default_vision_layers() -> usize {
    12
}
fn default_vision_heads() -> usize {
    12
}
fn default_image_size() -> usize {
    384
}
fn default_patch_size() -> usize {
    16
}
fn default_layer_norm_eps_vision() -> f64 {
    1e-5
}

#[derive(Debug, Deserialize)]
struct HfBlipTextConfig {
    #[serde(default = "default_vocab_size")]
    vocab_size: usize,
    #[serde(default = "default_text_hidden")]
    hidden_size: usize,
    #[serde(default = "default_text_intermediate")]
    intermediate_size: usize,
    #[serde(default = "default_text_projection_dim")]
    projection_dim: usize,
    #[serde(default = "default_text_layers")]
    num_hidden_layers: usize,
    #[serde(default = "default_text_heads")]
    num_attention_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: usize,
    #[serde(default = "default_layer_norm_eps_text")]
    layer_norm_eps: f64,
    #[serde(default = "default_true")]
    is_decoder: bool,
}

impl Default for HfBlipTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30524,
            hidden_size: 768,
            intermediate_size: 3072,
            projection_dim: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            max_position_embeddings: 512,
            layer_norm_eps: 1e-12,
            is_decoder: true,
        }
    }
}

fn default_vocab_size() -> usize {
    30524
}
fn default_text_hidden() -> usize {
    768
}
fn default_text_intermediate() -> usize {
    3072
}
fn default_text_projection_dim() -> usize {
    768
}
fn default_text_layers() -> usize {
    12
}
fn default_text_heads() -> usize {
    12
}
fn default_max_position_embeddings() -> usize {
    512
}
fn default_layer_norm_eps_text() -> f64 {
    1e-12
}
fn default_true() -> bool {
    true
}

impl HfBlipConfig {
    /// Convert to `candle_transformers::models::blip::Config`, setting
    /// `encoder_hidden_size` from the vision config (HF doesn't include it
    /// in `text_config`).
    fn into_candle_config(self) -> blip::Config {
        let text_config = candle_transformers::models::blip_text::Config {
            vocab_size: self.text_config.vocab_size,
            hidden_size: self.text_config.hidden_size,
            encoder_hidden_size: self.vision_config.hidden_size,
            intermediate_size: self.text_config.intermediate_size,
            projection_dim: self.text_config.projection_dim,
            num_hidden_layers: self.text_config.num_hidden_layers,
            num_attention_heads: self.text_config.num_attention_heads,
            max_position_embeddings: self.text_config.max_position_embeddings,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: self.text_config.layer_norm_eps,
            is_decoder: self.text_config.is_decoder,
        };
        let vision_config = blip::VisionConfig {
            hidden_size: self.vision_config.hidden_size,
            intermediate_size: self.vision_config.intermediate_size,
            projection_dim: self.vision_config.projection_dim,
            num_hidden_layers: self.vision_config.num_hidden_layers,
            num_attention_heads: self.vision_config.num_attention_heads,
            image_size: self.vision_config.image_size,
            patch_size: self.vision_config.patch_size,
            hidden_act: candle_nn::Activation::Gelu,
            layer_norm_eps: self.vision_config.layer_norm_eps,
        };
        blip::Config {
            text_config,
            vision_config,
            projection_dim: self.projection_dim,
            image_text_hidden_size: self.image_text_hidden_size,
        }
    }
}

// ---------------------------------------------------------------------------
// Task struct
// ---------------------------------------------------------------------------

/// Candle-based image-to-text task (BLIP image captioning).
pub struct CandleImageToTextTask {
    name: String,
    /// The model is behind a Mutex because `text_decoder().forward()` takes `&mut self`
    /// (it mutates internal KV cache state).
    model: Mutex<blip::BlipForConditionalGeneration>,
    /// Pre-computed image size from vision config.
    image_size: usize,
    device: Device,
    tokenizer: tokenizers::Tokenizer,
}

impl CandleImageToTextTask {
    /// Create from a model directory containing `config.json`, safetensors, and tokenizer.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading candle BLIP image-to-text model"
        );

        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Parse config.json
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;

        // Validate model_type
        let model_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;
        let model_type = model_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        if model_type != "blip" {
            return Err(TaskError::ModelLoad(format!(
                "Unsupported image-to-text architecture: '{model_type}'. \
                 Currently supported: blip."
            )));
        }

        let hf_config: HfBlipConfig = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse BLIP config: {e}")))?;

        let image_size = hf_config.vision_config.image_size;
        let candle_config = hf_config.into_candle_config();

        info!(
            vision_hidden = candle_config.vision_config.hidden_size,
            vision_layers = candle_config.vision_config.num_hidden_layers,
            text_hidden = candle_config.text_config.hidden_size,
            text_layers = candle_config.text_config.num_hidden_layers,
            image_size,
            "BLIP model configuration"
        );

        // Load weights
        let weight_files = utils::find_weight_files(model_dir)?;
        info!(num_files = weight_files.len(), "Found weight files");

        // BLIP uses F32 natively — respect model dtype but default to F32
        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => utils::read_model_dtype(&config_path).unwrap_or(DType::F32),
        };
        info!(dtype = ?dtype, "Compute dtype");

        let vb = if utils::is_pytorch_bin(&weight_files) {
            utils::load_pytorch_bin(&weight_files[0], dtype, &device)?
        } else {
            utils::load_safetensors_safe(&weight_files, dtype, &device)?
        };

        let model = blip::BlipForConditionalGeneration::new(&candle_config, vb)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to build BLIP model: {e}")))?;
        info!("BLIP model loaded");

        // Load tokenizer
        let tokenizer = utils::load_tokenizer(model_dir)?;
        info!("Tokenizer loaded");

        Ok(Self {
            name,
            model: Mutex::new(model),
            image_size,
            device,
            tokenizer,
        })
    }

    /// Decode a base64-encoded image into a normalized pixel tensor `[1, 3, H, W]`.
    fn preprocess_image(&self, input: &ImageToTextInput) -> TaskResult<Tensor> {
        let value = input
            .inputs
            .as_ref()
            .or(input.image.as_ref())
            .ok_or_else(|| {
                TaskError::InvalidInput("Missing 'inputs' or 'image' field in request".into())
            })?;

        let b64_str = match value {
            serde_json::Value::String(s) => {
                if let Some(pos) = s.find(";base64,") {
                    s[pos + 8..].to_string()
                } else {
                    s.clone()
                }
            }
            _ => {
                return Err(TaskError::InvalidInput(
                    "Expected base64-encoded image string".into(),
                ))
            }
        };

        let image_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &b64_str,
        )
        .map_err(|e| TaskError::InvalidInput(format!("Invalid base64: {e}")))?;

        let img = image::load_from_memory(&image_bytes)
            .map_err(|e| TaskError::InvalidInput(format!("Invalid image: {e}")))?;

        let size = self.image_size as u32;
        let img = img.resize_to_fill(size, size, image::imageops::FilterType::Triangle);
        let img = img.to_rgb8();
        let (w, h) = (img.width() as usize, img.height() as usize);
        let pixels = img.as_raw();

        // HWC → CHW, normalize with CLIP mean/std
        let mut data = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                for c in 0..3 {
                    let pixel = f32::from(pixels[idx + c]) / 255.0;
                    data[c * h * w + y * w + x] = (pixel - IMAGE_MEAN[c]) / IMAGE_STD[c];
                }
            }
        }

        let tensor = Tensor::from_vec(data, (3, h, w), &Device::Cpu)
            .map_err(|e| TaskError::Inference(format!("Failed to create image tensor: {e}")))?;

        // Move to device and add batch dim
        tensor
            .unsqueeze(0)
            .and_then(|t| t.to_device(&self.device))
            .map_err(|e| TaskError::Inference(format!("Failed to prepare image tensor: {e}")))
    }

    /// Run autoregressive generation: vision encoder → text decoder loop.
    fn generate(
        &self,
        image: &Tensor,
        _prompt: Option<&str>,
        max_tokens: usize,
    ) -> TaskResult<String> {
        let mut model = self.model.lock().map_err(|e| {
            TaskError::Inference(format!("Model lock poisoned: {e}"))
        })?;

        // Reset KV cache from any prior run
        model.reset_kv_cache();

        // 1. Encode image through ViT
        let image_embeds = image.apply(model.vision_model()).map_err(|e| {
            TaskError::Inference(format!("Vision encoding failed: {e}"))
        })?;

        debug!(shape = ?image_embeds.shape(), "Image embeddings computed");

        // 2. Autoregressive text decoding
        let mut token_ids: Vec<u32> = vec![BOS_TOKEN_ID];

        for index in 0..max_tokens {
            let context_size = if index > 0 { 1 } else { token_ids.len() };
            let start_pos = token_ids.len().saturating_sub(context_size);
            let input_ids =
                Tensor::new(&token_ids[start_pos..], &self.device).and_then(|t| t.unsqueeze(0));

            let input_ids = input_ids.map_err(|e| {
                TaskError::Inference(format!("Failed to create input_ids: {e}"))
            })?;

            let logits = model
                .text_decoder()
                .forward(&input_ids, &image_embeds)
                .map_err(|e| TaskError::Inference(format!("Decoder forward failed: {e}")))?;

            // Get logits for the last token: [1, seq_len, vocab] → [vocab]
            let logits = logits
                .squeeze(0)
                .and_then(|t| {
                    let last = t.dim(0).map(|d| d - 1)?;
                    t.get(last)
                })
                .map_err(|e| TaskError::Inference(format!("Logits extraction failed: {e}")))?;

            // Greedy decoding (argmax)
            let next_token = logits
                .argmax(0)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| TaskError::Inference(format!("Argmax failed: {e}")))?;

            if next_token == SEP_TOKEN_ID {
                break;
            }

            token_ids.push(next_token);
        }

        // 3. Decode tokens (skip the BOS token)
        let output_ids = &token_ids[1..];
        let text = self
            .tokenizer
            .decode(output_ids, true)
            .map_err(|e| TaskError::Inference(format!("Token decode failed: {e}")))?;

        Ok(text.trim().to_string())
    }
}

#[async_trait]
impl Task for CandleImageToTextTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id, "Image-to-text execute");

        let input: ImageToTextInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input JSON: {e}")),
        };

        let max_tokens = input.max_new_tokens.unwrap_or(MAX_GEN_TOKENS);
        let prompt = input.prompt.clone();

        // Preprocess image
        let pixel_values = match self.preprocess_image(&input) {
            Ok(t) => t,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        // Generate caption
        let caption = match self.generate(&pixel_values, prompt.as_deref(), max_tokens) {
            Ok(text) => text,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        debug!(caption = %caption, "Caption generated");

        let output = vec![ImageToTextOutput {
            generated_text: caption,
        }];

        match serde_json::to_string(&output) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("JSON serialization failed: {e}")),
        }
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}
