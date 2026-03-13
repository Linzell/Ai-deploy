//! HuggingFace API client for model discovery and metadata.
//!
//! Queries the HuggingFace Hub API to:
//! - List popular models for a given task (pipeline_tag)
//! - Fetch model metadata (pipeline_tag, files, library) for auto-config

use serde::Deserialize;
use std::path::Path;

const HF_API_BASE: &str = "https://huggingface.co/api/models";

/// Helper: check if a filename has a given extension (case-insensitive).
fn has_extension(filename: &str, ext: &str) -> bool {
    Path::new(filename)
        .extension()
        .is_some_and(|e| e.eq_ignore_ascii_case(ext))
}

/// Summary info for a model returned by the HF search API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HfModelSummary {
    /// Model ID (e.g., `Qwen/Qwen2.5-0.5B-Instruct`)
    #[serde(alias = "modelId")]
    pub id: String,
    /// Number of downloads
    #[serde(default)]
    pub downloads: u64,
    /// Number of likes
    #[serde(default)]
    pub likes: u64,
    /// Pipeline tag (e.g., `text-generation`)
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    /// Tags (e.g., `transformers`, `safetensors`, `qwen2`)
    #[serde(default)]
    pub tags: Vec<String>,
}

/// Detailed model metadata from the HF model card API.
#[derive(Debug, Deserialize)]
#[allow(dead_code)]
pub struct HfModelInfo {
    /// Model ID
    #[serde(alias = "modelId")]
    pub id: String,
    /// Pipeline tag (task type)
    #[serde(default)]
    pub pipeline_tag: Option<String>,
    /// Tags
    #[serde(default)]
    pub tags: Vec<String>,
    /// Siblings (files in the repo)
    #[serde(default)]
    pub siblings: Vec<HfFileSibling>,
    /// Library name (e.g., `transformers`, `onnx`)
    #[serde(default)]
    pub library_name: Option<String>,
}

/// A file in a HuggingFace repo.
#[derive(Debug, Deserialize)]
pub struct HfFileSibling {
    /// Relative file path (e.g., `onnx/model.onnx`, `model.safetensors`)
    #[serde(rename = "rfilename")]
    pub filename: String,
}

impl HfModelInfo {
    /// Check if the repo contains ONNX files.
    pub fn has_onnx(&self) -> bool {
        self.siblings
            .iter()
            .any(|f| has_extension(&f.filename, "onnx"))
    }

    /// Check if the repo contains safetensors files.
    pub fn has_safetensors(&self) -> bool {
        self.siblings
            .iter()
            .any(|f| has_extension(&f.filename, "safetensors"))
    }

    /// Check if the repo contains GGUF files.
    pub fn has_gguf(&self) -> bool {
        self.siblings
            .iter()
            .any(|f| has_extension(&f.filename, "gguf"))
    }

    /// Find the first ONNX model file in the repo.
    /// Prefers `onnx/model.onnx`, then any `.onnx` file.
    pub fn find_onnx_file(&self) -> Option<&str> {
        // Priority order
        let preferred = [
            "onnx/model.onnx",
            "onnx/model_quantized.onnx",
            "model.onnx",
            "model_quantized.onnx",
        ];
        for p in &preferred {
            if self.siblings.iter().any(|f| f.filename == *p) {
                return Some(p);
            }
        }
        // Fall back to first .onnx file
        self.siblings
            .iter()
            .find(|f| has_extension(&f.filename, "onnx"))
            .map(|f| f.filename.as_str())
    }

    /// Find the first GGUF file in the repo.
    pub fn find_gguf_file(&self) -> Option<&str> {
        self.siblings
            .iter()
            .find(|f| has_extension(&f.filename, "gguf"))
            .map(|f| f.filename.as_str())
    }

    /// Infer the best backend based on available files and pipeline_tag.
    ///
    /// Returns (backend_str, onnx_file, gguf_file).
    pub fn infer_backend(&self) -> (&str, Option<&str>, Option<&str>) {
        // GGUF files -> llama backend
        if self.has_gguf() {
            return ("llama", None, self.find_gguf_file());
        }

        let is_text_gen = self
            .pipeline_tag
            .as_deref()
            .is_some_and(|t| t == "text-generation");

        // Safetensors + text-generation -> candle
        if is_text_gen && self.has_safetensors() {
            return ("candle", None, None);
        }

        // ONNX files -> onnx backend
        if self.has_onnx() {
            return ("onnx", self.find_onnx_file(), None);
        }

        // Safetensors for non-text-gen -> still try candle
        if self.has_safetensors() {
            return ("candle", None, None);
        }

        // Default to auto
        ("auto", None, None)
    }
}

/// Task families for the no-args browsing mode.
pub struct TaskFamily {
    pub name: &'static str,
    pub description: &'static str,
    pub tasks: &'static [(&'static str, &'static str)],
}

/// All supported task families for browsing.
pub static TASK_FAMILIES: &[TaskFamily] = &[
    TaskFamily {
        name: "NLP",
        description: "Natural Language Processing",
        tasks: &[
            ("text-generation", "Text generation (LLMs, chatbots)"),
            ("text-classification", "Sentiment, topic classification"),
            ("token-classification", "NER, POS tagging"),
            ("feature-extraction", "Embeddings (semantic search)"),
            ("question-answering", "Extractive QA"),
            ("summarization", "Text summarization"),
            ("translation", "Machine translation"),
            ("fill-mask", "Masked language modeling"),
            ("zero-shot-classification", "Zero-shot text classification"),
        ],
    },
    TaskFamily {
        name: "Audio",
        description: "Audio & Speech",
        tasks: &[
            (
                "automatic-speech-recognition",
                "Speech-to-text (Whisper, etc.)",
            ),
            ("text-to-speech", "Text-to-speech synthesis"),
            ("audio-classification", "Audio event detection"),
        ],
    },
    TaskFamily {
        name: "Vision",
        description: "Computer Vision",
        tasks: &[
            ("image-classification", "Image classification"),
            ("object-detection", "Object detection"),
            (
                "zero-shot-image-classification",
                "CLIP zero-shot classification",
            ),
            ("image-to-text", "Image captioning, OCR"),
        ],
    },
    TaskFamily {
        name: "Multimodal",
        description: "Multimodal models",
        tasks: &[
            (
                "visual-question-answering",
                "VQA (Florence-2, etc.)",
            ),
            ("image-text-to-text", "Vision-language models"),
            (
                "document-question-answering",
                "Document understanding",
            ),
        ],
    },
];

/// Query HuggingFace API for popular models with a given pipeline_tag.
pub async fn search_models(
    task: &str,
    limit: usize,
) -> anyhow::Result<Vec<HfModelSummary>> {
    let url = format!(
        "{HF_API_BASE}?pipeline_tag={task}&sort=downloads&direction=-1&limit={limit}"
    );

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "maiia-inference-service")
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "HuggingFace API returned {}: {}",
            resp.status(),
            resp.text().await.unwrap_or_default()
        );
    }

    let models: Vec<HfModelSummary> = resp.json().await?;
    Ok(models)
}

/// Fetch detailed model info from HuggingFace API.
pub async fn get_model_info(model_id: &str) -> anyhow::Result<HfModelInfo> {
    let url = format!("{HF_API_BASE}/{model_id}");

    let client = reqwest::Client::new();
    let resp = client
        .get(&url)
        .header("User-Agent", "maiia-inference-service")
        .send()
        .await?;

    if !resp.status().is_success() {
        anyhow::bail!(
            "HuggingFace API returned {} for model '{}': {}",
            resp.status(),
            model_id,
            resp.text().await.unwrap_or_default()
        );
    }

    let info: HfModelInfo = resp.json().await?;
    Ok(info)
}

/// Format download count for display (e.g., 23551240 -> "23.6M").
pub fn format_downloads(n: u64) -> String {
    if n >= 1_000_000_000 {
        format!("{:.1}B", n as f64 / 1_000_000_000.0)
    } else if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        n.to_string()
    }
}
