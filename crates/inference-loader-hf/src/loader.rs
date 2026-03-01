//! HuggingFace Hub loader implementation.

use async_trait::async_trait;
use bytes::Bytes;
use hf_hub::api::tokio::{Api, ApiBuilder, ApiRepo};
use hf_hub::{Repo, RepoType};
use inference_core::{DataLoader, Error, Result};
use std::path::PathBuf;
use tracing::{debug, info, instrument, warn};

/// HuggingFace Hub loader.
///
/// Downloads and caches model files from HuggingFace Hub.
///
/// # Example
///
/// ```rust,ignore
/// use inference_loader_hf::HfLoader;
/// use inference_core::DataLoader;
///
/// let loader = HfLoader::new("meta-llama/Llama-2-7b-hf", None, None).await?;
/// let model_path = loader.get("model.safetensors").await?;
/// ```
pub struct HfLoader {
    #[allow(dead_code)]
    api: Api,
    repo: ApiRepo,
    model_id: String,
}

impl HfLoader {
    /// Create a new HuggingFace loader.
    ///
    /// # Arguments
    ///
    /// * `model_id` - HuggingFace model ID (e.g., "meta-llama/Llama-2-7b-hf")
    /// * `token` - Optional API token for gated models
    /// * `revision` - Optional revision (branch, tag, or commit hash)
    #[instrument(skip(token))]
    pub async fn new(
        model_id: &str,
        token: Option<String>,
        revision: Option<String>,
    ) -> Result<Self> {
        info!(model_id = %model_id, "Initializing HuggingFace loader");

        let mut builder = ApiBuilder::new().with_progress(true);

        if let Some(token) = token {
            builder = builder.with_token(Some(token));
        }

        let api = builder
            .build()
            .map_err(|e| Error::Loader(format!("Failed to build HF API: {e}")))?;

        let repo = match revision {
            Some(rev) => {
                let repo = Repo::with_revision(model_id.to_string(), RepoType::Model, rev);
                api.repo(repo)
            }
            None => api.model(model_id.to_string()),
        };

        Ok(Self {
            api,
            repo,
            model_id: model_id.to_string(),
        })
    }

    /// Create a loader from environment config.
    pub async fn from_config(config: &inference_core::Config) -> Result<Self> {
        let model_id = config.model_path.as_ref().ok_or_else(|| {
            Error::Config("MAIIA_AI_MODEL_PATH is required for HuggingFace".to_string())
        })?;

        Self::new(
            model_id,
            config.hf_token.clone(),
            config.model_revision.clone(),
        )
        .await
    }

    /// Get an ONNX model file, also downloading external data files if present.
    ///
    /// ONNX models can store weights in external data files (e.g., `model.onnx_data`)
    /// for large models. This method automatically detects and downloads these files
    /// so ONNX Runtime can load the model correctly.
    ///
    /// # Arguments
    ///
    /// * `onnx_path` - Path to the ONNX file (e.g., "onnx/model.onnx")
    ///
    /// # Returns
    ///
    /// Path to the downloaded ONNX file. External data files will be in the same directory.
    #[instrument(skip(self), fields(model_id = %self.model_id))]
    pub async fn get_onnx(&self, onnx_path: &str) -> Result<PathBuf> {
        // Download the main ONNX file
        let model_path = self.get(onnx_path).await?;

        // Check for external data file (convention: model.onnx -> model.onnx_data)
        let data_path = format!("{onnx_path}_data");
        match self.repo.get(&data_path).await {
            Ok(path) => {
                info!(
                    data_path = %path.display(),
                    "Downloaded ONNX external data file"
                );
            }
            Err(_) => {
                debug!(
                    data_path = %data_path,
                    "No external data file found (model may be self-contained)"
                );
            }
        }

        // Also check for .onnx.data variant (some exporters use this)
        let alt_data_path = format!("{}.data", onnx_path.trim_end_matches(".onnx"));
        if alt_data_path != data_path {
            match self.repo.get(&alt_data_path).await {
                Ok(path) => {
                    info!(
                        data_path = %path.display(),
                        "Downloaded ONNX external data file (alt format)"
                    );
                }
                Err(_) => {
                    debug!(
                        data_path = %alt_data_path,
                        "No alternate external data file"
                    );
                }
            }
        }

        Ok(model_path)
    }

    /// Try to get an ONNX model, falling back to quantized variants if external data is missing.
    ///
    /// This is useful when you want to prefer full-precision models but fall back to
    /// quantized versions that don't require external data files.
    ///
    /// Fallback order: original -> model_int8.onnx -> model_quantized.onnx
    #[instrument(skip(self), fields(model_id = %self.model_id))]
    pub async fn get_onnx_with_fallback(&self, onnx_path: &str) -> Result<PathBuf> {
        // Try to get the requested model with its external data
        let model_path = self.get_onnx(onnx_path).await?;

        // Check if external data file exists and was downloaded
        let data_path = format!("{onnx_path}_data");
        let needs_external_data = self.repo.get(&data_path).await.is_ok();

        if needs_external_data {
            // External data file exists, check if it was downloaded properly
            let parent = model_path.parent().ok_or_else(|| {
                Error::Loader("Cannot determine ONNX model directory".to_string())
            })?;
            let data_filename = PathBuf::from(&data_path).file_name().map_or_else(
                || format!("{}_data", model_path.file_name().unwrap().to_string_lossy()),
                |f| f.to_string_lossy().to_string(),
            );
            let local_data_path = parent.join(&data_filename);

            if !local_data_path.exists() {
                warn!(
                    model_path = %onnx_path,
                    "External data file missing, trying quantized fallbacks"
                );
                return self.try_quantized_fallbacks(onnx_path).await;
            }
        }

        Ok(model_path)
    }

    /// Try quantized model variants as fallbacks.
    async fn try_quantized_fallbacks(&self, original_path: &str) -> Result<PathBuf> {
        let base_dir = PathBuf::from(original_path)
            .parent()
            .map(|p| p.to_string_lossy().to_string())
            .unwrap_or_default();

        let fallbacks = [
            "model_int8.onnx",
            "model_quantized.onnx",
            "model_uint8.onnx",
        ];

        for fallback in fallbacks {
            let fallback_path = if base_dir.is_empty() {
                fallback.to_string()
            } else {
                format!("{base_dir}/{fallback}")
            };

            match self.get(&fallback_path).await {
                Ok(path) => {
                    info!(
                        fallback = %fallback_path,
                        "Using quantized model fallback"
                    );
                    return Ok(path);
                }
                Err(_) => {
                    debug!(fallback = %fallback_path, "Fallback not available");
                }
            }
        }

        Err(Error::Loader(format!(
            "No suitable ONNX model found for '{original_path}' (external data missing and no quantized fallbacks available)"
        )))
    }
}

#[async_trait]
impl DataLoader for HfLoader {
    #[instrument(skip(self), fields(model_id = %self.model_id))]
    async fn get(&self, path: &str) -> Result<PathBuf> {
        debug!(path = %path, "Fetching file from HuggingFace");

        self.repo
            .get(path)
            .await
            .map_err(|e| Error::Loader(format!("Failed to get '{path}': {e}")))
    }

    #[instrument(skip(self), fields(model_id = %self.model_id))]
    async fn get_bytes(&self, path: &str) -> Result<Bytes> {
        let file_path = self.get(path).await?;
        let contents = tokio::fs::read(&file_path).await.map_err(Error::Io)?;
        Ok(Bytes::from(contents))
    }

    async fn exists(&self, path: &str) -> Result<bool> {
        // HuggingFace Hub doesn't have a direct "exists" check,
        // so we try to get the file and catch the error
        match self.repo.get(path).await {
            Ok(_) => Ok(true),
            Err(_) => Ok(false),
        }
    }

    async fn list(&self, _prefix: &str) -> Result<Vec<String>> {
        // HuggingFace Hub doesn't support listing files directly via hf-hub crate
        // This would require using the HTTP API
        Err(Error::Loader(
            "Listing files is not supported for HuggingFace loader".to_string(),
        ))
    }

    fn name(&self) -> &'static str {
        "huggingface"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_loader_creation() {
        // This test uses a small public model
        let loader = HfLoader::new("hf-internal-testing/tiny-random-gpt2", None, None).await;
        assert!(loader.is_ok());
    }
}
