//! HuggingFace API client for model discovery and metadata.
//!
//! Queries the HuggingFace Hub API to:
//! - List popular models for a given task (pipeline_tag)
//! - Fetch model metadata (pipeline_tag, files, library) for auto-config

use serde::Deserialize;
use std::path::Path;
use std::time::Duration;

const HF_API_BASE: &str = "https://huggingface.co/api/models";
const HF_API_TIMEOUT: Duration = Duration::from_secs(30);

/// Create an HTTP client with a bounded timeout for HuggingFace API calls.
fn hf_client() -> reqwest::Client {
    reqwest::Client::builder()
        .timeout(HF_API_TIMEOUT)
        .build()
        .unwrap_or_else(|_| reqwest::Client::new())
}

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
    /// File size in bytes (available when fetched with `blobs=true`).
    #[serde(default)]
    pub size: Option<u64>,
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

    /// Check if the repo contains PyTorch bin files (legacy format).
    pub fn has_pytorch_bin(&self) -> bool {
        self.siblings
            .iter()
            .any(|f| f.filename == "pytorch_model.bin" || has_extension(&f.filename, "bin"))
    }

    /// Returns true if the model has at least one supported weight format.
    pub fn has_supported_files(&self) -> bool {
        self.has_safetensors() || self.has_onnx() || self.has_gguf() || self.has_pytorch_bin()
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
    /// Prefers common quantization levels in order: Q4_K_M, Q5_K_M, Q4_K_S, Q5_K_S, Q8_0.
    /// Falls back to the first GGUF file if none of the preferred patterns match.
    pub fn find_gguf_file(&self) -> Option<&str> {
        // Preferred quantization patterns in priority order (good quality/size tradeoff for CPU)
        let preferred = ["Q4_K_M", "Q5_K_M", "Q4_K_S", "Q5_K_S", "Q8_0", "Q6_K"];
        let gguf_files: Vec<&HfFileSibling> = self
            .siblings
            .iter()
            .filter(|f| has_extension(&f.filename, "gguf"))
            .collect();

        for pattern in &preferred {
            if let Some(f) = gguf_files
                .iter()
                .find(|f| f.filename.to_uppercase().contains(pattern))
            {
                return Some(f.filename.as_str());
            }
        }

        // Fall back to first GGUF file
        gguf_files.first().map(|f| f.filename.as_str())
    }

    /// Estimate total safetensors weight size in bytes.
    /// Returns 0 if no size info is available.
    pub fn safetensors_size(&self) -> u64 {
        self.siblings
            .iter()
            .filter(|f| has_extension(&f.filename, "safetensors"))
            .filter_map(|f| f.size)
            .sum()
    }

    /// Returns true if the model's safetensors weights are larger than the given threshold.
    /// Useful for routing large models to quantized backends (GGUF/llama.cpp).
    pub fn is_large_model(&self, threshold_bytes: u64) -> bool {
        let size = self.safetensors_size();
        // If we have size data and it exceeds the threshold
        size > threshold_bytes
    }

    /// Check if the model's tags indicate a seq2seq (encoder-decoder) architecture.
    ///
    /// Some tasks like `zero-shot-classification` can be implemented by either
    /// encoder-only models (BERT + classifier) or encoder-decoder models (BART NLI).
    /// This check prevents routing seq2seq models into the encoder-only pipeline.
    pub fn is_seq2seq_architecture(&self) -> bool {
        const SEQ2SEQ_ARCHS: &[&str] = &[
            "bart",
            "mbart",
            "t5",
            "mt5",
            "pegasus",
            "marian",
            "led",
            "longt5",
            "bigbird_pegasus",
            "blenderbot",
            "fsmt",
            "prophetnet",
            "plbart",
            "mvp",
            "nllb",
        ];
        self.tags.iter().any(|t| {
            let lower = t.to_lowercase();
            SEQ2SEQ_ARCHS.iter().any(|arch| lower == *arch)
        })
    }

    /// Check if this model uses an audio classification architecture supported by candle.
    pub fn is_candle_audio_architecture(&self) -> bool {
        const AUDIO_ARCHS: &[&str] = &[
            "wav2vec2",
            "hubert",
            "wav2vec2-conformer",
            "unispeech",
            "unispeech-sat",
            "clap",
        ];
        self.tags.iter().any(|t| {
            let lower = t.to_lowercase();
            AUDIO_ARCHS.iter().any(|arch| lower == *arch)
        })
    }

    /// Check if this model uses a DETR-family object detection architecture supported by candle.
    pub fn is_candle_object_detection_architecture(&self) -> bool {
        const OD_ARCHS: &[&str] = &["detr", "table-transformer"];
        self.tags.iter().any(|t| {
            let lower = t.to_lowercase();
            OD_ARCHS.iter().any(|arch| lower == *arch)
        })
    }

    /// Check if this model uses a BLIP-family image-to-text architecture supported by candle.
    pub fn is_candle_image_to_text_architecture(&self) -> bool {
        const ITT_ARCHS: &[&str] = &["blip"];
        self.tags.iter().any(|t| {
            let lower = t.to_lowercase();
            ITT_ARCHS.iter().any(|arch| lower == *arch)
        })
    }

    /// Infer the best backend based on available files and pipeline_tag.
    ///
    /// Returns (backend_str, onnx_file, gguf_file).
    ///
    /// Candle is returned for tasks it supports natively:
    /// - `text-generation` (decoder-only)
    /// - `automatic-speech-recognition` (encoder-decoder, Whisper)
    /// - `text-to-speech` (encoder-decoder, Parler TTS)
    /// - `image-to-text` (BLIP image captioning)
    /// - Encoder-only tasks: `token-classification`, `text-classification`,
    ///   `fill-mask`, `feature-extraction`, `question-answering`, etc.
    ///
    /// For all other tasks, ONNX is preferred. If the model has safetensors
    /// but no ONNX files for a non-Candle task, we return `"onnx"` with no
    /// file — the caller (`run_with_model`) is responsible for searching for
    /// an ONNX variant on HuggingFace.
    pub fn infer_backend(&self) -> (&str, Option<&str>, Option<&str>) {
        let tag = self.pipeline_tag.as_deref().unwrap_or("");

        // Tasks Candle supports via decoder/seq2seq backends
        let candle_gen_supported = matches!(
            tag,
            "text-generation"
                | "automatic-speech-recognition"
                | "text-to-speech"
                | "translation"
                | "summarization"
        );

        // For text-generation with safetensors, prefer Candle over llama/GGUF.
        // Candle loads safetensors directly and is more compatible — GGUF models
        // may use formats not yet supported by the bundled llama.cpp version
        // (e.g., qwen35.rope.dimension_sections length mismatches).
        if candle_gen_supported && self.has_safetensors() {
            return ("candle", None, None);
        }

        // GGUF files without safetensors -> llama backend
        if self.has_gguf() {
            return ("llama", None, self.find_gguf_file());
        }

        // Tasks Candle supports via encoder backend (BERT-family)
        let task_type = inference_core::TaskType::new(tag);
        let candle_encoder_supported =
            task_type.is_encoder_only() && !self.is_seq2seq_architecture();

        // Seq2seq models (BART/T5) on encoder-only tasks (e.g. zero-shot-classification)
        // → route to candle BART task instead of blocking
        let candle_bart_supported = task_type.is_encoder_only() && self.is_seq2seq_architecture();

        // Audio classification (Wav2Vec2, HuBERT, CLAP) → candle audio classifier
        let candle_audio_supported =
            task_type.is_audio_classification() && self.is_candle_audio_architecture();

        // Object detection (DETR, Table-Transformer) → candle object detection
        let candle_object_detection_supported =
            task_type.is_object_detection() && self.is_candle_object_detection_architecture();

        // Image-to-text (BLIP captioning) → candle image-to-text
        let candle_image_to_text_supported =
            task_type.is_image_to_text() && self.is_candle_image_to_text_architecture();

        // Safetensors + Candle-supported task -> candle
        if (candle_gen_supported
            || candle_encoder_supported
            || candle_bart_supported
            || candle_audio_supported
            || candle_object_detection_supported
            || candle_image_to_text_supported)
            && self.has_safetensors()
        {
            return ("candle", None, None);
        }

        // PyTorch bin + encoder task -> candle (VarBuilder::from_pth can load .bin)
        if (candle_encoder_supported || candle_audio_supported) && self.has_pytorch_bin() {
            return ("candle", None, None);
        }

        // ONNX files -> onnx backend (works for any task)
        if self.has_onnx() {
            return ("onnx", self.find_onnx_file(), None);
        }

        // Safetensors but task not supported by Candle -> need ONNX
        // Return "onnx" so the caller knows to search for an ONNX variant.
        if self.has_safetensors() {
            return ("onnx", None, None);
        }

        // PyTorch bin only, task not supported by Candle -> need ONNX
        // (e.g. openai/clip-vit-base-patch32 has only pytorch_model.bin)
        if self.has_pytorch_bin() {
            return ("onnx", None, None);
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
            ("visual-question-answering", "VQA (Florence-2, etc.)"),
            ("image-text-to-text", "Vision-language models"),
            ("document-question-answering", "Document understanding"),
        ],
    },
];

/// Tags that indicate a model has usable weight files.
const SUPPORTED_TAGS: &[&str] = &["transformers", "safetensors", "onnx", "gguf"];

/// Query HuggingFace API for popular models with a given pipeline_tag.
///
/// Results are post-filtered to exclude models without supported weight formats
/// (e.g. pyannote pipeline-only models that have no .safetensors/.onnx/.gguf files).
pub async fn search_models(task: &str, limit: usize) -> anyhow::Result<Vec<HfModelSummary>> {
    // Over-fetch to compensate for filtering out unsupported models.
    let fetch_limit = (limit * 2).max(30);
    let url = format!(
        "{HF_API_BASE}?pipeline_tag={task}&sort=downloads&direction=-1&limit={fetch_limit}"
    );

    let client = hf_client();
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

    let all: Vec<HfModelSummary> = resp.json().await?;

    // Keep only models that have at least one supported weight-format tag.
    let models: Vec<HfModelSummary> = all
        .into_iter()
        .filter(|m| m.tags.iter().any(|t| SUPPORTED_TAGS.contains(&t.as_str())))
        .take(limit)
        .collect();

    Ok(models)
}

/// Search HuggingFace API for models matching a text query, optionally filtered by task.
///
/// Unlike `search_models` which returns the most-downloaded models for a task,
/// this performs a free-text search across all repos (including community uploads,
/// fine-tunes, quantized variants, etc.).
pub async fn search_models_by_query(
    query: &str,
    task: Option<&str>,
    limit: usize,
) -> anyhow::Result<Vec<HfModelSummary>> {
    let fetch_limit = (limit * 2).max(30);
    let task_filter = task.map_or(String::new(), |t| format!("&pipeline_tag={t}"));
    let url = format!(
        "{HF_API_BASE}?search={query}{task_filter}&sort=downloads&direction=-1&limit={fetch_limit}"
    );

    let client = hf_client();
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

    let all: Vec<HfModelSummary> = resp.json().await?;

    // Keep only models with supported weight formats.
    let models: Vec<HfModelSummary> = all
        .into_iter()
        .filter(|m| m.tags.iter().any(|t| SUPPORTED_TAGS.contains(&t.as_str())))
        .take(limit)
        .collect();

    Ok(models)
}

/// Fetch detailed model info from HuggingFace API.
pub async fn get_model_info(model_id: &str) -> anyhow::Result<HfModelInfo> {
    let url = format!("{HF_API_BASE}/{model_id}?blobs=true");

    let client = hf_client();
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

/// Search for a GGUF variant of a model on HuggingFace.
///
/// Similar to [`find_compatible_variant`] but specifically looks for models
/// with GGUF files, which are preferred for CPU inference of large models.
pub async fn find_gguf_variant(model_id: &str) -> anyhow::Result<Option<HfModelInfo>> {
    let model_name = urlencoded_model_name(model_id);
    let base_tag = format!("base_model:{model_id}");
    let client = hf_client();

    // Search for GGUF conversions — these often include "GGUF" in their name
    let url =
        format!("{HF_API_BASE}?search={model_name}+GGUF&sort=downloads&direction=-1&limit=10");

    let resp = client
        .get(&url)
        .header("User-Agent", "maiia-inference-service")
        .send()
        .await?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let candidates: Vec<HfModelSummary> = resp.json().await?;

    // Find the first candidate derived from this model that has GGUF files
    for candidate in &candidates {
        if candidate.id == model_id {
            continue;
        }
        if candidate.tags.iter().any(|t| t == &base_tag) {
            if let Ok(info) = get_model_info(&candidate.id).await {
                if info.has_gguf() {
                    return Ok(Some(info));
                }
            }
        }
    }

    Ok(None)
}

/// Search for a compatible variant of a model on HuggingFace.
///
/// Looks for models derived from `model_id` (via `base_model` tag) that have
/// supported weight files (.onnx, .safetensors, .gguf), sorted by downloads.
/// Returns the best match (if any).
pub async fn find_compatible_variant(model_id: &str) -> anyhow::Result<Option<HfModelInfo>> {
    let model_name = urlencoded_model_name(model_id);
    let base_tag = format!("base_model:{model_id}");
    let client = hf_client();

    // Search broadly — no tag filter so we catch ONNX, GGUF, and safetensors variants
    let url = format!("{HF_API_BASE}?search={model_name}&sort=downloads&direction=-1&limit=10");

    let resp = client
        .get(&url)
        .header("User-Agent", "maiia-inference-service")
        .send()
        .await?;

    if !resp.status().is_success() {
        return Ok(None);
    }

    let candidates: Vec<HfModelSummary> = resp.json().await?;

    // Find the first candidate derived from this model that has supported files
    for candidate in &candidates {
        // Skip the original model itself
        if candidate.id == model_id {
            continue;
        }
        if candidate.tags.iter().any(|t| t == &base_tag) {
            if let Ok(info) = get_model_info(&candidate.id).await {
                if info.has_supported_files() {
                    return Ok(Some(info));
                }
            }
        }
    }

    // Fallback: try well-known ONNX export communities that use the naming
    // convention `{community}/{org}_{model}` (e.g. `Xenova/openai_clip-vit-base-patch32`,
    // `onnx-community/Salesforce_blip-image-captioning-base`).
    let well_known_prefixes = ["Xenova", "onnx-community"];
    let variant_suffix = model_id.replace('/', "_");

    for prefix in &well_known_prefixes {
        let candidate_id = format!("{prefix}/{variant_suffix}");
        if let Ok(info) = get_model_info(&candidate_id).await {
            if info.has_supported_files() {
                tracing::info!(
                    original = model_id,
                    variant = %candidate_id,
                    "Found variant via well-known ONNX community"
                );
                return Ok(Some(info));
            }
        }
    }

    Ok(None)
}

/// URL-encode only the slash in a model ID for search queries.
fn urlencoded_model_name(model_id: &str) -> String {
    // Extract just the model name part (after the org/) for better search results
    model_id
        .rsplit_once('/')
        .map_or(model_id.to_string(), |(_, name)| name.to_string())
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
