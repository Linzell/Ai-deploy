//! Candle-based encoder-only inference for BERT-family models.
//!
//! Runs encoder-only models (BERT, RoBERTa, DistilBERT, etc.) directly from
//! safetensors for tasks that don't need autoregressive decoding:
//!
//! - **token-classification** (NER, POS tagging)
//! - **text-classification** / sentiment-analysis
//! - **fill-mask** (masked language modeling)
//! - **feature-extraction** (embeddings)
//! - **question-answering** (extractive QA)
//!
//! ## Why?
//!
//! Most popular encoder models on HuggingFace only have safetensors — no ONNX
//! export. Previously these models had no execution path. This module loads them
//! natively via `candle_transformers::models::bert::BertModel`.

use crate::error::{TaskError, TaskResult};
use crate::utils;
use async_trait::async_trait;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tokenizers::Tokenizer;
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Encoder task type
// ---------------------------------------------------------------------------

/// Which encoder sub-task to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EncoderTaskKind {
    TokenClassification,
    TextClassification,
    FillMask,
    FeatureExtraction,
    QuestionAnswering,
}

impl EncoderTaskKind {
    /// Derive from the task-type string in the config.
    pub fn from_task_type(task_type: &str) -> TaskResult<Self> {
        let s = task_type.to_lowercase().replace('-', "_");
        match s.as_str() {
            "token_classification" => Ok(Self::TokenClassification),
            "text_classification" | "sentiment_analysis" | "zero_shot_classification" => {
                Ok(Self::TextClassification)
            }
            "fill_mask" => Ok(Self::FillMask),
            "feature_extraction" => Ok(Self::FeatureExtraction),
            "question_answering" => Ok(Self::QuestionAnswering),
            other => Err(TaskError::Config(format!(
                "Unsupported encoder task kind: {other}"
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct CandleEncoderInput {
    /// Primary text input
    pub text: Option<String>,
    /// Alias: some pipelines send `inputs` instead of `text`
    pub inputs: Option<String>,
    /// Second segment for QA (`question` + `context` pattern)
    pub question: Option<String>,
    pub context: Option<String>,
}

impl CandleEncoderInput {
    fn primary_text(&self) -> TaskResult<&str> {
        self.text
            .as_deref()
            .or(self.inputs.as_deref())
            .or(self.question.as_deref())
            .ok_or_else(|| TaskError::InvalidInput("Missing 'text' or 'inputs' field".into()))
    }

    fn secondary_text(&self) -> Option<&str> {
        self.context.as_deref()
    }
}

#[derive(Debug, Serialize)]
pub struct CandleEncoderOutput {
    /// Raw result — shape/meaning depends on the task kind.
    #[serde(flatten)]
    pub inner: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Classifier head (loaded from weights)
// ---------------------------------------------------------------------------

/// A simple linear classifier on top of encoder hidden states.
struct ClassifierHead {
    linear: Linear,
    #[allow(dead_code)]
    num_labels: usize,
}

impl ClassifierHead {
    /// Try to load a `classifier` linear layer from the weights.
    fn load(vb: &VarBuilder, hidden_size: usize, num_labels: usize) -> TaskResult<Self> {
        let linear = candle_nn::linear(hidden_size, num_labels, vb.pp("classifier"))
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load classifier head: {e}")))?;
        Ok(Self { linear, num_labels })
    }

    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        self.linear.forward(hidden_states)
    }
}

// ---------------------------------------------------------------------------
// MLM prediction head (for fill-mask)
// ---------------------------------------------------------------------------

/// BERT masked language model prediction head.
///
/// Architecture: hidden → dense(hidden→hidden) → GELU → LayerNorm → vocab projection + bias
///
/// The vocab projection typically shares weights with the input word embeddings
/// (weight tying). We load `bert.embeddings.word_embeddings.weight` as the
/// projection matrix and `cls.predictions.bias` as the output bias.
struct MlmHead {
    dense: Linear,
    layer_norm: candle_nn::LayerNorm,
    decoder_weight: Tensor, // word_embeddings.weight [vocab, hidden] used as projection
    decoder_bias: Tensor,   // cls.predictions.bias [vocab]
}

impl MlmHead {
    /// Load from `cls.predictions.*` tensors in the VarBuilder.
    fn load(vb: &VarBuilder, hidden_size: usize) -> TaskResult<Self> {
        let cls_vb = vb.pp("cls").pp("predictions");

        // cls.predictions.transform.dense
        let dense = candle_nn::linear(hidden_size, hidden_size, cls_vb.pp("transform").pp("dense"))
            .map_err(|e| {
                TaskError::ModelLoad(format!("MLM head: failed to load transform.dense: {e}"))
            })?;

        // cls.predictions.transform.LayerNorm
        let layer_norm =
            candle_nn::layer_norm(hidden_size, 1e-12, cls_vb.pp("transform").pp("LayerNorm"))
                .map_err(|e| {
                    TaskError::ModelLoad(format!(
                        "MLM head: failed to load transform.LayerNorm: {e}"
                    ))
                })?;

        // Decoder projection: use word_embeddings.weight (weight-tied)
        // Try multiple prefixes since BertModel may use bert.embeddings or just embeddings
        let decoder_weight = vb
            .pp("bert")
            .pp("embeddings")
            .pp("word_embeddings")
            .get_unchecked("weight")
            .or_else(|_| {
                vb.pp("embeddings")
                    .pp("word_embeddings")
                    .get_unchecked("weight")
            })
            .map_err(|e| {
                TaskError::ModelLoad(format!(
                    "MLM head: failed to load word_embeddings.weight for decoder: {e}"
                ))
            })?;

        // cls.predictions.bias (output bias over vocab)
        let decoder_bias = cls_vb.get_unchecked("bias").map_err(|e| {
            TaskError::ModelLoad(format!(
                "MLM head: failed to load cls.predictions.bias: {e}"
            ))
        })?;

        Ok(Self {
            dense,
            layer_norm,
            decoder_weight,
            decoder_bias,
        })
    }

    /// Forward: hidden_states [batch, seq, hidden] → logits [batch, seq, vocab]
    fn forward(&self, hidden_states: &Tensor) -> candle_core::Result<Tensor> {
        // dense + GELU
        let h = self.dense.forward(hidden_states)?;
        let h = h.gelu_erf()?;
        // LayerNorm
        let h = self.layer_norm.forward(&h)?;
        // Project to vocab: h [batch, seq, hidden] @ decoder_weight^T [hidden, vocab] + bias
        let logits = h.broadcast_matmul(&self.decoder_weight.t()?)?;
        logits.broadcast_add(&self.decoder_bias)
    }
}

// ---------------------------------------------------------------------------
// Main task struct
// ---------------------------------------------------------------------------

/// Candle-based encoder task.
///
/// Wraps `BertModel` (which handles BERT, RoBERTa, etc. via `model_type` prefix
/// routing) plus an optional classifier head.
pub struct CandleEncoderTask {
    name: String,
    model: BertModel,
    classifier: Option<ClassifierHead>,
    mlm_head: Option<MlmHead>,
    tokenizer: Tokenizer,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    kind: EncoderTaskKind,
    id2label: Option<HashMap<usize, String>>,
}

impl CandleEncoderTask {
    /// Create from a model directory containing config.json, tokenizer.json,
    /// and safetensors weights.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();
        let kind = EncoderTaskKind::from_task_type(config.task_type.as_str())?;

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            kind = ?kind,
            "Loading candle encoder model"
        );

        // Resolve device (CPU / Metal / CUDA with fallback)
        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Parse model config.json
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;
        let model_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let bert_config: BertConfig = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse BertConfig: {e}")))?;

        // Extract id2label mapping if present
        let id2label = model_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let idx: usize = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect::<HashMap<usize, String>>()
            });

        let num_labels = model_json
            .get("num_labels")
            .and_then(serde_json::Value::as_u64)
            .map(|n| n as usize)
            .or_else(|| id2label.as_ref().map(HashMap::len))
            .unwrap_or(2); // default for classification

        info!(
            model_type = bert_config.model_type.as_deref().unwrap_or("unknown"),
            hidden_size = bert_config.hidden_size,
            num_labels = num_labels,
            has_id2label = id2label.is_some(),
            "Model configuration"
        );

        // Find and load weights
        let weight_files = utils::find_weight_files(model_dir)?;
        let is_pytorch = utils::is_pytorch_bin(&weight_files);
        info!(
            num_files = weight_files.len(),
            format = if is_pytorch {
                "pytorch_model.bin"
            } else {
                "safetensors"
            },
            "Found weight files"
        );

        // Encoder models are typically small — use F32 on CPU for accuracy,
        // or the model's native dtype on GPU
        let model_dtype = utils::read_model_dtype(&config_path);
        let dtype = match &device {
            Device::Cpu => DType::F32,
            _ => model_dtype.unwrap_or(DType::F32),
        };
        info!(dtype = ?dtype, "Compute dtype");

        let vb = if is_pytorch {
            utils::load_pytorch_bin(&weight_files[0], dtype, &device)?
        } else {
            utils::load_safetensors_safe(&weight_files, dtype, &device)?
        };

        // Load BertModel — it handles prefix routing (bert.*/roberta.*) internally
        let model = BertModel::load(vb.clone(), &bert_config)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to load BertModel: {e}")))?;
        info!("BertModel loaded");

        // Load classifier head if needed
        let classifier = match kind {
            EncoderTaskKind::TokenClassification
            | EncoderTaskKind::TextClassification
            | EncoderTaskKind::QuestionAnswering => {
                match ClassifierHead::load(&vb, bert_config.hidden_size, num_labels) {
                    Ok(head) => {
                        info!(num_labels = num_labels, "Classifier head loaded");
                        Some(head)
                    }
                    Err(e) => {
                        warn!(
                            error = %e,
                            "No classifier head found in weights — will return raw hidden states"
                        );
                        None
                    }
                }
            }
            EncoderTaskKind::FillMask | EncoderTaskKind::FeatureExtraction => None,
        };

        // Load MLM prediction head for fill-mask
        let mlm_head = if kind == EncoderTaskKind::FillMask {
            match MlmHead::load(&vb, bert_config.hidden_size) {
                Ok(head) => {
                    info!("MLM prediction head loaded");
                    Some(head)
                }
                Err(e) => {
                    warn!(
                        error = %e,
                        "No MLM head found in weights — fill-mask will return hidden states only"
                    );
                    None
                }
            }
        } else {
            None
        };

        // Load tokenizer — handles tokenizer.json, vocab.txt, vocab.json+merges.txt
        let tokenizer = utils::load_tokenizer(model_dir)?;
        info!("Tokenizer loaded");

        Ok(Self {
            name,
            model,
            classifier,
            mlm_head,
            tokenizer,
            device,
            dtype,
            kind,
            id2label,
        })
    }

    // -----------------------------------------------------------------------
    // Inference helpers
    // -----------------------------------------------------------------------

    /// Tokenize input text, returning (input_ids, token_type_ids, attention_mask, tokens).
    fn tokenize(
        &self,
        text: &str,
        text_pair: Option<&str>,
    ) -> TaskResult<(Tensor, Tensor, Tensor, Vec<String>)> {
        let encoding = if let Some(pair) = text_pair {
            self.tokenizer.encode((text, pair), true)
        } else {
            self.tokenizer.encode(text, true)
        }
        .map_err(|e| TaskError::InvalidInput(format!("Tokenization failed: {e}")))?;

        let ids = encoding.get_ids();
        let type_ids = encoding.get_type_ids();
        let attention = encoding.get_attention_mask();
        let tokens: Vec<String> = encoding.get_tokens().to_vec();

        let input_ids = Tensor::new(ids, &self.device)
            .map_err(|e| TaskError::Inference(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("unsqueeze failed: {e}")))?;
        let token_type_ids = Tensor::new(type_ids, &self.device)
            .map_err(|e| TaskError::Inference(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("unsqueeze failed: {e}")))?;
        let attention_mask = Tensor::new(attention, &self.device)
            .map_err(|e| TaskError::Inference(format!("Tensor creation failed: {e}")))?
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("unsqueeze failed: {e}")))?;

        Ok((input_ids, token_type_ids, attention_mask, tokens))
    }

    /// Run BertModel forward and optional classifier.
    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> TaskResult<Tensor> {
        let hidden = self
            .model
            .forward(input_ids, token_type_ids, Some(attention_mask))
            .map_err(|e| TaskError::Inference(format!("BertModel forward failed: {e}")))?;

        if let Some(ref head) = self.classifier {
            head.forward(&hidden)
                .map_err(|e| TaskError::Inference(format!("Classifier forward failed: {e}")))
        } else {
            Ok(hidden)
        }
    }

    // -----------------------------------------------------------------------
    // Task-specific output formatting
    // -----------------------------------------------------------------------

    fn process_token_classification(
        &self,
        logits: &Tensor,
        tokens: &[String],
        text: &str,
    ) -> TaskResult<serde_json::Value> {
        // logits: [1, seq_len, num_labels]
        let logits = logits
            .squeeze(0)
            .map_err(|e| TaskError::Inference(format!("squeeze failed: {e}")))?;
        // [seq_len, num_labels]

        let seq_len = logits
            .dim(0)
            .map_err(|e| TaskError::Inference(format!("dim failed: {e}")))?;

        let mut entities = Vec::new();

        for (i, token) in tokens.iter().enumerate().take(seq_len) {
            let token_logits = logits
                .i(i)
                .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?;
            let probs = softmax_1d(&token_logits)?;
            let (label_idx, score) = argmax_1d(&probs)?;

            let label = self
                .id2label
                .as_ref()
                .and_then(|m| m.get(&label_idx))
                .cloned()
                .unwrap_or_else(|| format!("LABEL_{label_idx}"));

            // Skip special tokens and O labels
            if token.starts_with('[') && token.ends_with(']') {
                continue;
            }
            if token == "<s>" || token == "</s>" || token == "<pad>" {
                continue;
            }
            if label == "O" {
                continue;
            }

            entities.push(serde_json::json!({
                "entity": label,
                "score": score,
                "word": token,
                "index": i,
            }));
        }

        debug!(
            num_entities = entities.len(),
            text = text,
            "Token classification complete"
        );

        Ok(serde_json::json!(entities))
    }

    fn process_text_classification(&self, logits: &Tensor) -> TaskResult<serde_json::Value> {
        // logits: [1, num_labels] — take first token (CLS) if [1, seq, num_labels]
        let logits = match logits.rank() {
            3 => {
                // [1, seq_len, num_labels] → take CLS token (index 0)
                logits
                    .i((0, 0))
                    .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?
            }
            2 => {
                // [1, num_labels] → squeeze batch dim
                logits
                    .squeeze(0)
                    .map_err(|e| TaskError::Inference(format!("squeeze failed: {e}")))?
            }
            r => {
                return Err(TaskError::Inference(format!("Unexpected logits rank: {r}")));
            }
        };

        let probs = softmax_1d(&logits)?;
        let num_labels = probs.len();

        let mut results: Vec<serde_json::Value> = Vec::new();
        for (i, &prob) in probs.iter().enumerate().take(num_labels) {
            let label = self
                .id2label
                .as_ref()
                .and_then(|m| m.get(&i))
                .cloned()
                .unwrap_or_else(|| format!("LABEL_{i}"));
            results.push(serde_json::json!({
                "label": label,
                "score": prob,
            }));
        }

        // Sort by score descending
        results.sort_by(|a, b| {
            b.get("score")
                .and_then(serde_json::Value::as_f64)
                .unwrap_or(0.0)
                .partial_cmp(
                    &a.get("score")
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0),
                )
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(serde_json::json!(results))
    }

    fn process_fill_mask(
        &self,
        hidden_states: &Tensor,
        tokens: &[String],
        text: &str,
    ) -> TaskResult<serde_json::Value> {
        // Find the [MASK] token position
        let mask_idx = tokens
            .iter()
            .position(|t| t == "[MASK]" || t == "<mask>")
            .ok_or_else(|| TaskError::InvalidInput("No [MASK] token found in input".into()))?;

        let Some(mlm_head) = &self.mlm_head else {
            // No MLM head — return hidden state as fallback
            warn!(
                "fill-mask without MLM head — returning hidden state at mask position. \
                 For proper fill-mask predictions, use a model that includes cls.predictions weights."
            );
            let mask_hidden = hidden_states
                .i((0, mask_idx))
                .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?;
            let values: Vec<f32> = mask_hidden
                .to_vec1()
                .map_err(|e| TaskError::Inference(format!("to_vec1 failed: {e}")))?;
            return Ok(serde_json::json!({
                "mask_position": mask_idx,
                "hidden_state_dim": values.len(),
                "note": "fill-mask without MLM head returns hidden state, not vocab predictions"
            }));
        };

        // Run MLM head: hidden_states → logits [1, seq_len, vocab_size]
        let logits = mlm_head
            .forward(hidden_states)
            .map_err(|e| TaskError::Inference(format!("MLM head forward failed: {e}")))?;

        // Extract logits at mask position: [vocab_size]
        let mask_logits = logits
            .i((0, mask_idx))
            .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?;

        // Softmax to get probabilities
        let probs = softmax_1d(&mask_logits)?;

        // Top-5 predictions
        let top_k = 5;
        let mut scored: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        let predictions: Vec<serde_json::Value> = scored
            .iter()
            .map(|&(token_id, score)| {
                let token_str = self
                    .tokenizer
                    .id_to_token(token_id as u32)
                    .unwrap_or_else(|| format!("[UNK:{token_id}]"));
                // Build the filled sentence
                let filled = text
                    .replace("[MASK]", &token_str)
                    .replace("<mask>", &token_str);
                serde_json::json!({
                    "score": score,
                    "token": token_id,
                    "token_str": token_str,
                    "sequence": filled,
                })
            })
            .collect();

        debug!(
            mask_position = mask_idx,
            top_token = %predictions[0]["token_str"],
            top_score = %predictions[0]["score"],
            "Fill-mask predictions"
        );

        Ok(serde_json::json!(predictions))
    }

    #[allow(clippy::unused_self)]
    fn process_feature_extraction(&self, hidden_states: &Tensor) -> TaskResult<serde_json::Value> {
        // hidden_states: [1, seq_len, hidden_size]
        // Mean pooling over non-padding tokens
        let hidden = hidden_states
            .squeeze(0)
            .map_err(|e| TaskError::Inference(format!("squeeze failed: {e}")))?;
        // [seq_len, hidden_size]
        let embedding = hidden
            .mean(0)
            .map_err(|e| TaskError::Inference(format!("mean failed: {e}")))?;
        // [hidden_size]
        let values: Vec<f32> = embedding
            .to_vec1()
            .map_err(|e| TaskError::Inference(format!("to_vec1 failed: {e}")))?;

        Ok(serde_json::json!({
            "embedding": values,
            "dimensions": values.len(),
        }))
    }

    #[allow(clippy::unused_self)]
    fn process_question_answering(
        &self,
        logits: &Tensor,
        tokens: &[String],
        _text: &str,
        _context: Option<&str>,
    ) -> TaskResult<serde_json::Value> {
        // For QA with a classifier head (num_labels=2): logits are start/end logits
        // logits: [1, seq_len, 2]
        let logits = logits
            .squeeze(0)
            .map_err(|e| TaskError::Inference(format!("squeeze failed: {e}")))?;
        // [seq_len, 2]

        let seq_len = logits
            .dim(0)
            .map_err(|e| TaskError::Inference(format!("dim failed: {e}")))?;

        let start_logits: Vec<f32> = logits
            .i((.., 0))
            .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?
            .to_vec1()
            .map_err(|e| TaskError::Inference(format!("to_vec1 failed: {e}")))?;
        let end_logits: Vec<f32> = logits
            .i((.., 1))
            .map_err(|e| TaskError::Inference(format!("index failed: {e}")))?
            .to_vec1()
            .map_err(|e| TaskError::Inference(format!("to_vec1 failed: {e}")))?;

        // Find best start/end positions
        let (start_idx, start_score) = start_logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));
        let (end_idx, end_score) = end_logits
            .iter()
            .enumerate()
            .skip(start_idx)
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, &0.0));

        // Extract answer text from tokens
        let answer_tokens: Vec<&str> = tokens[start_idx..=end_idx.min(seq_len - 1)]
            .iter()
            .map(String::as_str)
            .collect();
        let answer = answer_tokens.join(" ").replace(" ##", "").replace("##", "");

        let score = (start_score + end_score) / 2.0;

        Ok(serde_json::json!({
            "answer": answer,
            "score": score,
            "start": start_idx,
            "end": end_idx,
        }))
    }
}

// ---------------------------------------------------------------------------
// Task trait implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Task for CandleEncoderTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "Encoder task execute");

        // Parse input
        let input: CandleEncoderInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => {
                return GrpcTaskResult::err(format!("Invalid input JSON: {e}"));
            }
        };

        let text = match input.primary_text() {
            Ok(t) => t,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };
        let text_pair = input.secondary_text();

        // Tokenize
        let (input_ids, token_type_ids, attention_mask, tokens) =
            match self.tokenize(text, text_pair) {
                Ok(t) => t,
                Err(e) => return GrpcTaskResult::err(e.to_string()),
            };

        debug!(
            num_tokens = tokens.len(),
            kind = ?self.kind,
            "Running encoder forward pass"
        );

        // Forward pass
        let output = match self.forward(&input_ids, &token_type_ids, &attention_mask) {
            Ok(o) => o,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        // Post-process based on task kind
        let result = match self.kind {
            EncoderTaskKind::TokenClassification => {
                self.process_token_classification(&output, &tokens, text)
            }
            EncoderTaskKind::TextClassification => self.process_text_classification(&output),
            EncoderTaskKind::FillMask => self.process_fill_mask(&output, &tokens, text),
            EncoderTaskKind::FeatureExtraction => self.process_feature_extraction(&output),
            EncoderTaskKind::QuestionAnswering => {
                self.process_question_answering(&output, &tokens, text, text_pair)
            }
        };

        match result {
            Ok(value) => GrpcTaskResult::ok(value.to_string()),
            Err(e) => GrpcTaskResult::err(e.to_string()),
        }
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Utility functions
// ---------------------------------------------------------------------------

/// Softmax over a 1D tensor, returns Vec<f32>.
fn softmax_1d(tensor: &Tensor) -> TaskResult<Vec<f32>> {
    let max_val = tensor
        .max(0)
        .map_err(|e| TaskError::Inference(format!("max failed: {e}")))?;
    let shifted = tensor
        .broadcast_sub(&max_val)
        .map_err(|e| TaskError::Inference(format!("sub failed: {e}")))?;
    let exp = shifted
        .exp()
        .map_err(|e| TaskError::Inference(format!("exp failed: {e}")))?;
    let sum = exp
        .sum_all()
        .map_err(|e| TaskError::Inference(format!("sum failed: {e}")))?;
    let probs = exp
        .broadcast_div(&sum)
        .map_err(|e| TaskError::Inference(format!("div failed: {e}")))?;
    probs
        .to_dtype(DType::F32)
        .map_err(|e| TaskError::Inference(format!("to_dtype failed: {e}")))?
        .to_vec1()
        .map_err(|e| TaskError::Inference(format!("to_vec1 failed: {e}")))
}

/// Argmax over a Vec<f32>, returns (index, value).
fn argmax_1d(values: &[f32]) -> TaskResult<(usize, f32)> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, &v)| (i, v))
        .ok_or_else(|| TaskError::Inference("Empty tensor".into()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encoder_task_kind_from_task_type() {
        assert_eq!(
            EncoderTaskKind::from_task_type("token-classification").unwrap(),
            EncoderTaskKind::TokenClassification
        );
        assert_eq!(
            EncoderTaskKind::from_task_type("text-classification").unwrap(),
            EncoderTaskKind::TextClassification
        );
        assert_eq!(
            EncoderTaskKind::from_task_type("sentiment-analysis").unwrap(),
            EncoderTaskKind::TextClassification
        );
        assert_eq!(
            EncoderTaskKind::from_task_type("fill-mask").unwrap(),
            EncoderTaskKind::FillMask
        );
        assert_eq!(
            EncoderTaskKind::from_task_type("feature-extraction").unwrap(),
            EncoderTaskKind::FeatureExtraction
        );
        assert_eq!(
            EncoderTaskKind::from_task_type("question-answering").unwrap(),
            EncoderTaskKind::QuestionAnswering
        );
        assert!(EncoderTaskKind::from_task_type("text-generation").is_err());
    }

    #[test]
    fn test_softmax_1d() {
        let device = Device::Cpu;
        let tensor = Tensor::new(&[1.0f32, 2.0, 3.0], &device).unwrap();
        let probs = softmax_1d(&tensor).unwrap();
        assert_eq!(probs.len(), 3);
        // Sum should be ~1.0
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        // Last element should be largest
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
    }

    #[test]
    fn test_argmax_1d() {
        let values = vec![0.1, 0.7, 0.2];
        let (idx, val) = argmax_1d(&values).unwrap();
        assert_eq!(idx, 1);
        assert!((val - 0.7).abs() < 1e-5);
    }
}
