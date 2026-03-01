//! Data loader trait for fetching model weights and data files.

use async_trait::async_trait;
use bytes::Bytes;
use std::path::PathBuf;

use crate::Result;

/// Trait for loading data from various sources (HuggingFace, S3, local).
///
/// All loaders implement this trait, allowing the inference service
/// to be agnostic of the underlying data source.
#[async_trait]
pub trait DataLoader: Send + Sync {
    /// Get a file and return its local path (downloads if necessary).
    ///
    /// This method should cache files locally to avoid repeated downloads.
    async fn get(&self, path: &str) -> Result<PathBuf>;

    /// Get file contents as bytes (useful for small files).
    async fn get_bytes(&self, path: &str) -> Result<Bytes>;

    /// Check if a file exists in the source.
    async fn exists(&self, path: &str) -> Result<bool>;

    /// List files matching a pattern (if supported by the source).
    async fn list(&self, prefix: &str) -> Result<Vec<String>>;

    /// Get the name of this loader (for logging).
    fn name(&self) -> &'static str;
}

/// A loader that combines multiple loaders (e.g., HF for models, S3 for data).
pub struct CompositeLoader {
    model_loader: Box<dyn DataLoader>,
    data_loader: Box<dyn DataLoader>,
}

impl CompositeLoader {
    pub fn new(model_loader: Box<dyn DataLoader>, data_loader: Box<dyn DataLoader>) -> Self {
        Self {
            model_loader,
            data_loader,
        }
    }

    /// Get a model file.
    pub async fn get_model(&self, path: &str) -> Result<PathBuf> {
        self.model_loader.get(path).await
    }

    /// Get model file contents.
    pub async fn get_model_bytes(&self, path: &str) -> Result<Bytes> {
        self.model_loader.get_bytes(path).await
    }

    /// Get a data file.
    pub async fn get_data(&self, path: &str) -> Result<PathBuf> {
        self.data_loader.get(path).await
    }

    /// Get data file contents.
    pub async fn get_data_bytes(&self, path: &str) -> Result<Bytes> {
        self.data_loader.get_bytes(path).await
    }
}
