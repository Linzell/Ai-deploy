//! BART encoder-decoder model for sequence classification (zero-shot NLI).
//!
//! Implements `BartForSequenceClassification` from safetensors, used primarily
//! by `facebook/bart-large-mnli` for zero-shot classification via NLI.
//!
//! Architecture:
//! - Shared token embeddings + learned positional embeddings
//! - Encoder: N transformer layers with self-attention
//! - Decoder: N transformer layers with self-attention + cross-attention
//! - Classification head on decoder's EOS token hidden state
//!
//! Zero-shot approach: for each candidate label, form the hypothesis
//! `"This example is {label}."`, run encoder-decoder, take entailment score.

// VarBuilder is an Arc-based handle — passing by value is the intended Candle API.
#![allow(clippy::needless_pass_by_value)]

use crate::error::{TaskError, TaskResult};
use crate::utils;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{layer_norm, Embedding, LayerNorm, Linear, Module, VarBuilder};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, info};

// ---------------------------------------------------------------------------
// BART Config
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
struct BartConfig {
    vocab_size: usize,
    d_model: usize,
    encoder_layers: usize,
    decoder_layers: usize,
    encoder_attention_heads: usize,
    decoder_attention_heads: usize,
    encoder_ffn_dim: usize,
    decoder_ffn_dim: usize,
    max_position_embeddings: usize,
    #[serde(default = "default_activation")]
    activation_function: String,
    #[serde(default)]
    #[allow(dead_code)]
    scale_embedding: bool,
    pad_token_id: u32,
    eos_token_id: u32,
    #[serde(default = "default_num_labels")]
    _num_labels: usize,
}

fn default_activation() -> String {
    "gelu".into()
}
fn default_num_labels() -> usize {
    3
}

impl BartConfig {
    fn activation(&self) -> candle_nn::Activation {
        match self.activation_function.as_str() {
            "relu" => candle_nn::Activation::Relu,
            "silu" | "swish" => candle_nn::Activation::Silu,
            _ => candle_nn::Activation::Gelu,
        }
    }
}

// ---------------------------------------------------------------------------
// Multi-head attention
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BartAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
}

impl BartAttention {
    fn load(num_heads: usize, d_model: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let head_dim = d_model / num_heads;
        let scaling = (head_dim as f64).powf(-0.5);
        let q_proj = candle_nn::linear(d_model, d_model, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(d_model, d_model, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(d_model, d_model, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(d_model, d_model, vb.pp("out_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scaling,
        })
    }

    fn reshape_for_heads(&self, xs: &Tensor, bsz: usize) -> candle_core::Result<Tensor> {
        xs.reshape((bsz, (), self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()
    }

    /// Forward pass.
    /// `kv_states`: if Some, used as key/value (cross-attention).
    /// `attn_mask`: causal mask for decoder self-attention.
    fn forward(
        &self,
        xs: &Tensor,
        kv_states: Option<&Tensor>,
        attn_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (bsz, tgt_len, _) = xs.dims3()?;
        let query = (xs.apply(&self.q_proj)? * self.scaling)?;

        let (key, value) = match kv_states {
            Some(kv) => (
                self.reshape_for_heads(&kv.apply(&self.k_proj)?, bsz)?,
                self.reshape_for_heads(&kv.apply(&self.v_proj)?, bsz)?,
            ),
            None => (
                self.reshape_for_heads(&xs.apply(&self.k_proj)?, bsz)?,
                self.reshape_for_heads(&xs.apply(&self.v_proj)?, bsz)?,
            ),
        };

        let proj = (bsz * self.num_heads, (), self.head_dim);
        let query = self.reshape_for_heads(&query, bsz)?.reshape(proj)?;
        let key = key.reshape(proj)?;
        let value = value.reshape(proj)?;

        let attn_weights = query.matmul(&key.transpose(1, 2)?)?;
        let attn_weights = match attn_mask {
            Some(m) => attn_weights.broadcast_add(m)?,
            None => attn_weights,
        };
        let attn_probs = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_probs.matmul(&value)?;

        attn_output
            .reshape((bsz, self.num_heads, tgt_len, self.head_dim))?
            .transpose(1, 2)?
            .reshape((bsz, tgt_len, self.num_heads * self.head_dim))?
            .apply(&self.out_proj)
    }
}

// ---------------------------------------------------------------------------
// Encoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BartEncoderLayer {
    self_attn: BartAttention,
    self_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: candle_nn::Activation,
}

impl BartEncoderLayer {
    fn load(cfg: &BartConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attn =
            BartAttention::load(cfg.encoder_attention_heads, cfg.d_model, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let fc1 = candle_nn::linear(cfg.d_model, cfg.encoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.encoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: cfg.activation(),
        })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.self_attn.forward(xs, None, None)?;
        let xs = (xs + residual)?.apply(&self.self_attn_layer_norm)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)?;
        (xs + residual)?.apply(&self.final_layer_norm)
    }
}

// ---------------------------------------------------------------------------
// Decoder layer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BartDecoderLayer {
    self_attn: BartAttention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: BartAttention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    activation: candle_nn::Activation,
}

impl BartDecoderLayer {
    fn load(cfg: &BartConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let self_attn =
            BartAttention::load(cfg.decoder_attention_heads, cfg.d_model, vb.pp("self_attn"))?;
        let self_attn_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;
        let encoder_attn = BartAttention::load(
            cfg.decoder_attention_heads,
            cfg.d_model,
            vb.pp("encoder_attn"),
        )?;
        let encoder_attn_layer_norm =
            layer_norm(cfg.d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?;
        let fc1 = candle_nn::linear(cfg.d_model, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.decoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;
        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            activation: cfg.activation(),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        encoder_hidden: &Tensor,
        causal_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        // Self-attention with causal mask
        let residual = xs;
        let xs = self.self_attn.forward(xs, None, Some(causal_mask))?;
        let xs = (xs + residual)?.apply(&self.self_attn_layer_norm)?;
        // Cross-attention to encoder output
        let residual = &xs;
        let xs = self.encoder_attn.forward(&xs, Some(encoder_hidden), None)?;
        let xs = (xs + residual)?.apply(&self.encoder_attn_layer_norm)?;
        // FFN
        let residual = &xs;
        let xs = xs
            .apply(&self.fc1)?
            .apply(&self.activation)?
            .apply(&self.fc2)?;
        (xs + residual)?.apply(&self.final_layer_norm)
    }
}

// ---------------------------------------------------------------------------
// Encoder & Decoder
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BartEncoder {
    embed_positions: Embedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<BartEncoderLayer>,
}

impl BartEncoder {
    fn load(
        cfg: &BartConfig,
        shared_embed: &Embedding,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        // BART uses learned positional embeddings with offset 2
        // (positions 0,1 are reserved → actual positions start at index 2)
        let embed_positions = Embedding::new(
            vb.pp("embed_positions")
                .get((cfg.max_position_embeddings + 2, cfg.d_model), "weight")?,
            cfg.d_model,
        );
        let layernorm_embedding = layer_norm(cfg.d_model, 1e-5, vb.pp("layernorm_embedding"))?;
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for i in 0..cfg.encoder_layers {
            layers.push(BartEncoderLayer::load(cfg, vb_layers.pp(i))?);
        }
        let _ = shared_embed; // shared embed is used via the caller
        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
        })
    }

    fn forward(&self, token_embeds: &Tensor) -> candle_core::Result<Tensor> {
        let seq_len = token_embeds.dim(1)?;
        // Position ids: [2, 3, ..., seq_len+1] (offset=2 for BART)
        let position_ids: Vec<u32> = (2..seq_len as u32 + 2).collect();
        let position_ids =
            Tensor::new(position_ids.as_slice(), token_embeds.device())?.unsqueeze(0)?;
        let pos_embeds = self.embed_positions.forward(&position_ids)?;

        let mut xs = (token_embeds + pos_embeds)?.apply(&self.layernorm_embedding)?;
        for layer in &self.layers {
            xs = layer.forward(&xs)?;
        }
        Ok(xs)
    }
}

#[derive(Debug, Clone)]
struct BartDecoder {
    embed_positions: Embedding,
    layernorm_embedding: LayerNorm,
    layers: Vec<BartDecoderLayer>,
}

impl BartDecoder {
    fn load(cfg: &BartConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let embed_positions = Embedding::new(
            vb.pp("embed_positions")
                .get((cfg.max_position_embeddings + 2, cfg.d_model), "weight")?,
            cfg.d_model,
        );
        let layernorm_embedding = layer_norm(cfg.d_model, 1e-5, vb.pp("layernorm_embedding"))?;
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.decoder_layers);
        for i in 0..cfg.decoder_layers {
            layers.push(BartDecoderLayer::load(cfg, vb_layers.pp(i))?);
        }
        Ok(Self {
            embed_positions,
            layernorm_embedding,
            layers,
        })
    }

    fn forward(
        &self,
        token_embeds: &Tensor,
        encoder_hidden: &Tensor,
    ) -> candle_core::Result<Tensor> {
        let seq_len = token_embeds.dim(1)?;
        let position_ids: Vec<u32> = (2..seq_len as u32 + 2).collect();
        let position_ids =
            Tensor::new(position_ids.as_slice(), token_embeds.device())?.unsqueeze(0)?;
        let pos_embeds = self.embed_positions.forward(&position_ids)?;

        let mut xs = (token_embeds + pos_embeds)?.apply(&self.layernorm_embedding)?;

        // Causal mask for decoder self-attention
        let mask: Vec<f32> = (0..seq_len)
            .flat_map(|i| (0..seq_len).map(move |j| if j > i { f32::NEG_INFINITY } else { 0f32 }))
            .collect();
        let causal_mask = Tensor::from_vec(mask, (seq_len, seq_len), token_embeds.device())?;

        for layer in &self.layers {
            xs = layer.forward(&xs, encoder_hidden, &causal_mask)?;
        }
        Ok(xs)
    }
}

// ---------------------------------------------------------------------------
// BartForSequenceClassification
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct BartClassificationHead {
    dense: Linear,
    out_proj: Linear,
}

impl BartClassificationHead {
    fn load(d_model: usize, num_labels: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let dense = candle_nn::linear(d_model, d_model, vb.pp("dense"))?;
        let out_proj = candle_nn::linear(d_model, num_labels, vb.pp("out_proj"))?;
        Ok(Self { dense, out_proj })
    }

    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        // xs: [batch, d_model] — the EOS token hidden state
        let xs = xs.apply(&self.dense)?.tanh()?;
        xs.apply(&self.out_proj)
    }
}

/// Full BART model with classification head.
struct BartForSequenceClassification {
    shared: Embedding,
    encoder: BartEncoder,
    decoder: BartDecoder,
    classification_head: BartClassificationHead,
    eos_token_id: u32,
    #[allow(dead_code)]
    pad_token_id: u32,
}

impl BartForSequenceClassification {
    fn load(cfg: &BartConfig, num_labels: usize, vb: VarBuilder) -> TaskResult<Self> {
        let model_vb = vb.pp("model");
        let shared = Embedding::new(
            model_vb
                .pp("shared")
                .get((cfg.vocab_size, cfg.d_model), "weight")
                .map_err(|e| TaskError::ModelLoad(format!("shared embeddings: {e}")))?,
            cfg.d_model,
        );
        let encoder = BartEncoder::load(cfg, &shared, model_vb.pp("encoder"))
            .map_err(|e| TaskError::ModelLoad(format!("encoder: {e}")))?;
        let decoder = BartDecoder::load(cfg, model_vb.pp("decoder"))
            .map_err(|e| TaskError::ModelLoad(format!("decoder: {e}")))?;
        let classification_head =
            BartClassificationHead::load(cfg.d_model, num_labels, vb.pp("classification_head"))
                .map_err(|e| TaskError::ModelLoad(format!("classification_head: {e}")))?;

        Ok(Self {
            shared,
            encoder,
            decoder,
            classification_head,
            eos_token_id: cfg.eos_token_id,
            pad_token_id: cfg.pad_token_id,
        })
    }

    /// Run full encoder-decoder and classification head.
    /// Returns logits [batch, num_labels].
    fn forward(&self, input_ids: &Tensor) -> candle_core::Result<Tensor> {
        // Token embeddings (shared between encoder and decoder)
        let token_embeds = self.shared.forward(input_ids)?;

        // Encoder
        let encoder_hidden = self.encoder.forward(&token_embeds)?;

        // Decoder input: same input_ids (for sequence classification, BART
        // feeds the same tokens into the decoder)
        let decoder_embeds = self.shared.forward(input_ids)?;
        let decoder_hidden = self.decoder.forward(&decoder_embeds, &encoder_hidden)?;

        // Find the EOS token position for each batch element and extract its hidden state.
        // BartForSequenceClassification uses the last EOS token hidden state.
        let (bsz, seq_len, _d) = decoder_hidden.dims3()?;
        let input_flat: Vec<u32> = input_ids.reshape(bsz * seq_len)?.to_vec1()?;

        let mut eos_hidden = Vec::with_capacity(bsz);
        for b in 0..bsz {
            // Find last EOS position in this batch element
            let start = b * seq_len;
            let end = start + seq_len;
            let eos_pos = input_flat[start..end]
                .iter()
                .rposition(|&id| id == self.eos_token_id)
                .unwrap_or(seq_len - 1);
            let hidden = decoder_hidden.i((b, eos_pos))?; // [d_model]
            eos_hidden.push(hidden);
        }
        let eos_hidden = Tensor::stack(&eos_hidden, 0)?; // [batch, d_model]
        self.classification_head.forward(&eos_hidden) // [batch, num_labels]
    }
}

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CandleBartInput {
    /// Primary text (the premise for NLI)
    pub text: Option<String>,
    pub inputs: Option<String>,
    /// Candidate labels for zero-shot classification (comma-separated or array)
    pub candidate_labels: Option<serde_json::Value>,
}

impl CandleBartInput {
    fn text(&self) -> TaskResult<&str> {
        self.text
            .as_deref()
            .or(self.inputs.as_deref())
            .ok_or_else(|| TaskError::InvalidInput("Missing 'text' or 'inputs' field".into()))
    }

    fn labels(&self) -> TaskResult<Vec<String>> {
        match &self.candidate_labels {
            Some(serde_json::Value::Array(arr)) => {
                let labels: Vec<String> = arr
                    .iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect();
                if labels.is_empty() {
                    return Err(TaskError::InvalidInput(
                        "candidate_labels array is empty".into(),
                    ));
                }
                Ok(labels)
            }
            Some(serde_json::Value::String(s)) => {
                let labels: Vec<String> = s.split(',').map(|l| l.trim().to_string()).collect();
                if labels.is_empty() || labels.iter().all(String::is_empty) {
                    return Err(TaskError::InvalidInput(
                        "candidate_labels string is empty".into(),
                    ));
                }
                Ok(labels)
            }
            _ => Err(TaskError::InvalidInput(
                "Missing or invalid 'candidate_labels' — provide an array or comma-separated string"
                    .into(),
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// CandleBartTask
// ---------------------------------------------------------------------------

/// BART-based zero-shot classification task.
///
/// Uses NLI (Natural Language Inference): for each candidate label, constructs
/// hypothesis `"This example is {label}."`, runs through BART, and takes the
/// entailment logit as the label's score.
pub struct CandleBartTask {
    name: String,
    model: BartForSequenceClassification,
    tokenizer: Tokenizer,
    device: Device,
    /// Index of the "entailment" label in the NLI output (usually 2)
    entailment_idx: usize,
}

impl CandleBartTask {
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
            "Loading BART model for zero-shot classification"
        );

        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Parse BART config.json
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;
        let bart_config: BartConfig = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse BartConfig: {e}")))?;
        let model_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        info!(
            d_model = bart_config.d_model,
            encoder_layers = bart_config.encoder_layers,
            decoder_layers = bart_config.decoder_layers,
            vocab_size = bart_config.vocab_size,
            "BART configuration"
        );

        // Determine entailment index from id2label
        let entailment_idx = model_json
            .get("label2id")
            .and_then(|v| v.get("entailment"))
            .and_then(serde_json::Value::as_u64)
            .map_or(2, |n| n as usize);

        let num_labels = model_json
            .get("_num_labels")
            .or_else(|| model_json.get("num_labels"))
            .and_then(serde_json::Value::as_u64)
            .map_or(3, |n| n as usize);

        // Load weights
        let weight_files = utils::find_weight_files(model_dir)?;
        let dtype = DType::F32; // BART is small enough for F32 on any device
        info!(dtype = ?dtype, num_files = weight_files.len(), "Loading weights");

        let vb = utils::load_safetensors_safe(&weight_files, dtype, &device)?;

        // Load model
        let model = BartForSequenceClassification::load(&bart_config, num_labels, vb)?;
        info!("BART model loaded successfully");

        // Load tokenizer
        let tokenizer = utils::load_tokenizer(model_dir)?;
        info!("Tokenizer loaded");

        Ok(Self {
            name,
            model,
            tokenizer,
            device,
            entailment_idx,
        })
    }

    /// Run NLI for a single (premise, hypothesis) pair.
    /// Returns logits [num_labels].
    fn classify_pair(&self, premise: &str, hypothesis: &str) -> TaskResult<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode((premise, hypothesis), true)
            .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;

        let ids = encoding.get_ids();
        let input_ids = Tensor::new(ids, &self.device)
            .map_err(|e| TaskError::Inference(format!("tensor: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?;

        let logits = self
            .model
            .forward(&input_ids)
            .map_err(|e| TaskError::Inference(format!("BART forward failed: {e}")))?;

        // logits: [1, num_labels] → squeeze → [num_labels]
        logits
            .squeeze(0)
            .map_err(|e| TaskError::Inference(format!("squeeze: {e}")))?
            .to_vec1::<f32>()
            .map_err(|e| TaskError::Inference(format!("to_vec1: {e}")))
    }

    /// Zero-shot classification: run NLI for each candidate label and softmax
    /// across entailment scores.
    fn zero_shot(&self, text: &str, candidate_labels: &[String]) -> TaskResult<serde_json::Value> {
        let mut entailment_logits = Vec::with_capacity(candidate_labels.len());

        for label in candidate_labels {
            let hypothesis = format!("This example is {label}.");
            let logits = self.classify_pair(text, &hypothesis)?;
            let entailment_score = logits.get(self.entailment_idx).copied().unwrap_or(0.0);
            entailment_logits.push(entailment_score);
        }

        // Softmax across all labels' entailment scores
        let max_val = entailment_logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);
        let exp: Vec<f32> = entailment_logits
            .iter()
            .map(|&x| (x - max_val).exp())
            .collect();
        let sum: f32 = exp.iter().sum();
        let probs: Vec<f32> = exp.iter().map(|&e| e / sum).collect();

        // Sort by score descending
        let mut scored: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let labels: Vec<&str> = scored
            .iter()
            .map(|&(i, _)| candidate_labels[i].as_str())
            .collect();
        let scores: Vec<f32> = scored.iter().map(|&(_, s)| s).collect();

        debug!(
            top_label = labels[0],
            top_score = scores[0],
            num_labels = candidate_labels.len(),
            "Zero-shot classification complete"
        );

        Ok(serde_json::json!({
            "sequence": text,
            "labels": labels,
            "scores": scores,
        }))
    }
}

// ---------------------------------------------------------------------------
// Task trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Task for CandleBartTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "BART task execute");

        let input: CandleBartInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input JSON: {e}")),
        };

        let text = match input.text() {
            Ok(t) => t,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        let labels = match input.labels() {
            Ok(l) => l,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        match self.zero_shot(text, &labels) {
            Ok(value) => GrpcTaskResult::ok(value.to_string()),
            Err(e) => GrpcTaskResult::err(e.to_string()),
        }
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bart_config_parse() {
        let json = r#"{
            "vocab_size": 50265,
            "d_model": 1024,
            "encoder_layers": 12,
            "decoder_layers": 12,
            "encoder_attention_heads": 16,
            "decoder_attention_heads": 16,
            "encoder_ffn_dim": 4096,
            "decoder_ffn_dim": 4096,
            "max_position_embeddings": 1024,
            "activation_function": "gelu",
            "scale_embedding": false,
            "pad_token_id": 1,
            "eos_token_id": 2,
            "_num_labels": 3
        }"#;
        let cfg: BartConfig = serde_json::from_str(json).unwrap();
        assert_eq!(cfg.d_model, 1024);
        assert_eq!(cfg.encoder_layers, 12);
        assert_eq!(cfg.vocab_size, 50265);
        assert!(!cfg.scale_embedding);
    }

    #[test]
    fn test_bart_input_labels_array() {
        let json = r#"{"text": "hello", "candidate_labels": ["a", "b", "c"]}"#;
        let input: CandleBartInput = serde_json::from_str(json).unwrap();
        let labels = input.labels().unwrap();
        assert_eq!(labels, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_bart_input_labels_csv() {
        let json = r#"{"text": "hello", "candidate_labels": "politics, sports, technology"}"#;
        let input: CandleBartInput = serde_json::from_str(json).unwrap();
        let labels = input.labels().unwrap();
        assert_eq!(labels, vec!["politics", "sports", "technology"]);
    }

    #[test]
    fn test_bart_input_missing_labels() {
        let json = r#"{"text": "hello"}"#;
        let input: CandleBartInput = serde_json::from_str(json).unwrap();
        assert!(input.labels().is_err());
    }
}
