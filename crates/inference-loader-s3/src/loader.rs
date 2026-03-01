//! AWS S3 / MinIO loader implementation.

use async_trait::async_trait;
use aws_config::BehaviorVersion;
use aws_sdk_s3::config::Builder as S3ConfigBuilder;
use aws_sdk_s3::Client;
use bytes::Bytes;
use inference_core::{DataLoader, Error, Result};
use std::env;
use std::path::PathBuf;
use tokio::io::AsyncWriteExt;
use tracing::{debug, info, instrument};

/// AWS S3 / MinIO loader.
///
/// Downloads and caches files from S3-compatible storage (AWS S3, MinIO, etc.).
///
/// # Environment Variables
///
/// - `AWS_ENDPOINT_URL`: Custom endpoint for MinIO (e.g., "http://localhost:9000")
/// - `AWS_ACCESS_KEY_ID`: Access key
/// - `AWS_SECRET_ACCESS_KEY`: Secret key
/// - `AWS_REGION`: Region (default: "us-east-1")
///
/// # Example
///
/// ```rust,ignore
/// use inference_loader_s3::S3Loader;
/// use inference_core::DataLoader;
///
/// // AWS S3
/// let loader = S3Loader::new("my-bucket", "models/whisper", "/tmp/cache").await?;
///
/// // MinIO (set AWS_ENDPOINT_URL=http://localhost:9000)
/// let loader = S3Loader::new("my-bucket", "models/whisper", "/tmp/cache").await?;
///
/// let audio_bytes = loader.get_bytes("recording.wav").await?;
/// ```
pub struct S3Loader {
    client: Client,
    bucket: String,
    prefix: String,
    cache_dir: PathBuf,
    enable_cache: bool,
}

impl S3Loader {
    /// Create a new S3/MinIO loader.
    ///
    /// Automatically detects MinIO via `AWS_ENDPOINT_URL` environment variable.
    ///
    /// # Arguments
    ///
    /// * `bucket` - S3 bucket name
    /// * `prefix` - Key prefix for files (e.g., "models/whisper")
    /// * `cache_dir` - Local directory for caching downloaded files
    #[instrument]
    pub async fn new(bucket: &str, prefix: &str, cache_dir: &str) -> Result<Self> {
        let endpoint_url = env::var("AWS_ENDPOINT_URL").ok();
        let region = env::var("AWS_REGION").unwrap_or_else(|_| "us-east-1".to_string());

        if let Some(ref endpoint) = endpoint_url {
            info!(bucket = %bucket, prefix = %prefix, endpoint = %endpoint, "Initializing MinIO loader");
        } else {
            info!(bucket = %bucket, prefix = %prefix, "Initializing S3 loader");
        }

        let client = Self::build_client(endpoint_url.as_deref(), &region).await?;

        let cache_path = PathBuf::from(cache_dir);
        tokio::fs::create_dir_all(&cache_path)
            .await
            .map_err(Error::Io)?;

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            cache_dir: cache_path,
            enable_cache: true,
        })
    }

    /// Create a new S3 loader with explicit endpoint (for MinIO).
    ///
    /// # Arguments
    ///
    /// * `bucket` - S3 bucket name
    /// * `prefix` - Key prefix for files
    /// * `cache_dir` - Local directory for caching
    /// * `endpoint_url` - Custom endpoint URL (e.g., "http://localhost:9000")
    /// * `region` - AWS region
    #[instrument]
    pub async fn with_endpoint(
        bucket: &str,
        prefix: &str,
        cache_dir: &str,
        endpoint_url: &str,
        region: &str,
    ) -> Result<Self> {
        info!(
            bucket = %bucket,
            prefix = %prefix,
            endpoint = %endpoint_url,
            region = %region,
            "Initializing S3 loader with custom endpoint"
        );

        let client = Self::build_client(Some(endpoint_url), region).await?;

        let cache_path = PathBuf::from(cache_dir);
        tokio::fs::create_dir_all(&cache_path)
            .await
            .map_err(Error::Io)?;

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            cache_dir: cache_path,
            enable_cache: true,
        })
    }

    /// Create a new S3 loader with region only (for AWS S3).
    #[instrument]
    pub async fn with_region(
        bucket: &str,
        prefix: &str,
        cache_dir: &str,
        region: &str,
    ) -> Result<Self> {
        info!(bucket = %bucket, prefix = %prefix, region = %region, "Initializing S3 loader with region");

        let client = Self::build_client(None, region).await?;

        let cache_path = PathBuf::from(cache_dir);
        tokio::fs::create_dir_all(&cache_path)
            .await
            .map_err(Error::Io)?;

        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.to_string(),
            cache_dir: cache_path,
            enable_cache: true,
        })
    }

    /// Build S3 client with optional custom endpoint (for MinIO).
    async fn build_client(endpoint_url: Option<&str>, region: &str) -> Result<Client> {
        let sdk_config = aws_config::defaults(BehaviorVersion::latest())
            .region(aws_config::Region::new(region.to_string()))
            .load()
            .await;

        let mut s3_config_builder = S3ConfigBuilder::from(&sdk_config);

        // Configure for MinIO if endpoint is provided
        if let Some(endpoint) = endpoint_url {
            s3_config_builder = s3_config_builder
                .endpoint_url(endpoint)
                .force_path_style(true); // Required for MinIO
        }

        let client = Client::from_conf(s3_config_builder.build());
        Ok(client)
    }

    /// Create a loader from environment config.
    ///
    /// Reads configuration from environment variables and creates the appropriate loader.
    pub async fn from_config(config: &inference_core::Config) -> Result<Self> {
        let bucket = config
            .s3_bucket
            .as_ref()
            .ok_or_else(|| Error::Config("MAIIA_AI_S3_BUCKET is required".to_string()))?;

        let prefix = config.s3_data_prefix.as_deref().unwrap_or("");

        // Check for custom endpoint (MinIO)
        let endpoint_url = config
            .s3_endpoint
            .clone()
            .or_else(|| env::var("AWS_ENDPOINT_URL").ok());
        let env_region = env::var("AWS_REGION").ok();
        let region = config
            .s3_region
            .as_deref()
            .or(env_region.as_deref())
            .unwrap_or("us-east-1");

        match endpoint_url {
            Some(endpoint) => {
                Self::with_endpoint(bucket, prefix, &config.cache_dir, &endpoint, region).await
            }
            None => Self::with_region(bucket, prefix, &config.cache_dir, region).await,
        }
    }

    /// Disable caching (always download from S3).
    #[must_use]
    pub fn disable_cache(mut self) -> Self {
        self.enable_cache = false;
        self
    }

    /// Get the full S3 key for a path.
    fn full_key(&self, path: &str) -> String {
        if self.prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", self.prefix.trim_end_matches('/'), path)
        }
    }

    /// Get the cache path for a file.
    fn cache_path(&self, path: &str) -> PathBuf {
        self.cache_dir.join(path)
    }
}

#[async_trait]
impl DataLoader for S3Loader {
    #[instrument(skip(self), fields(bucket = %self.bucket, prefix = %self.prefix))]
    async fn get(&self, path: &str) -> Result<PathBuf> {
        let cache_path = self.cache_path(path);

        // Check cache first
        if self.enable_cache && cache_path.exists() {
            debug!(path = %path, "Using cached file");
            return Ok(cache_path);
        }

        // Download from S3
        let key = self.full_key(path);
        debug!(key = %key, "Downloading from S3");

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| Error::Loader(format!("S3 get_object failed for '{key}': {e}")))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| Error::Loader(format!("Failed to read S3 response: {e}")))?
            .into_bytes();

        // Write to cache
        if let Some(parent) = cache_path.parent() {
            tokio::fs::create_dir_all(parent).await.map_err(Error::Io)?;
        }

        let mut file = tokio::fs::File::create(&cache_path)
            .await
            .map_err(Error::Io)?;
        file.write_all(&bytes).await.map_err(Error::Io)?;

        info!(path = %path, size = bytes.len(), "Downloaded and cached file from S3");
        Ok(cache_path)
    }

    #[instrument(skip(self), fields(bucket = %self.bucket, prefix = %self.prefix))]
    async fn get_bytes(&self, path: &str) -> Result<Bytes> {
        // For bytes, we can skip the cache and stream directly
        let key = self.full_key(path);
        debug!(key = %key, "Streaming bytes from S3");

        let response = self
            .client
            .get_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
            .map_err(|e| Error::Loader(format!("S3 get_object failed for '{key}': {e}")))?;

        let bytes = response
            .body
            .collect()
            .await
            .map_err(|e| Error::Loader(format!("Failed to read S3 response: {e}")))?
            .into_bytes();

        Ok(bytes)
    }

    #[instrument(skip(self), fields(bucket = %self.bucket))]
    async fn exists(&self, path: &str) -> Result<bool> {
        let key = self.full_key(path);

        match self
            .client
            .head_object()
            .bucket(&self.bucket)
            .key(&key)
            .send()
            .await
        {
            Ok(_) => Ok(true),
            Err(e) => {
                // Check if it's a "not found" error
                if e.to_string().contains("404") || e.to_string().contains("NotFound") {
                    Ok(false)
                } else {
                    Err(Error::Loader(format!("S3 head_object failed: {e}")))
                }
            }
        }
    }

    #[instrument(skip(self), fields(bucket = %self.bucket))]
    async fn list(&self, prefix: &str) -> Result<Vec<String>> {
        let full_prefix = self.full_key(prefix);
        debug!(prefix = %full_prefix, "Listing S3 objects");

        let mut files = Vec::new();
        let mut continuation_token: Option<String> = None;

        loop {
            let mut request = self
                .client
                .list_objects_v2()
                .bucket(&self.bucket)
                .prefix(&full_prefix);

            if let Some(token) = continuation_token {
                request = request.continuation_token(token);
            }

            let response = request
                .send()
                .await
                .map_err(|e| Error::Loader(format!("S3 list_objects failed: {e}")))?;

            if let Some(contents) = response.contents {
                for object in contents {
                    if let Some(key) = object.key {
                        // Remove the prefix to get relative path
                        let relative = key
                            .strip_prefix(&self.prefix)
                            .unwrap_or(&key)
                            .trim_start_matches('/');
                        files.push(relative.to_string());
                    }
                }
            }

            if response.is_truncated.unwrap_or(false) {
                continuation_token = response.next_continuation_token;
            } else {
                break;
            }
        }

        Ok(files)
    }

    fn name(&self) -> &'static str {
        "s3"
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_full_key_generation() {
        // Test key generation without S3 client
        let prefix = "models/whisper";
        let path = "config.json";
        let expected = "models/whisper/config.json";

        let result = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", prefix.trim_end_matches('/'), path)
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_key_empty_prefix() {
        let prefix = "";
        let path = "config.json";
        let expected = "config.json";

        let result = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", prefix.trim_end_matches('/'), path)
        };

        assert_eq!(result, expected);
    }

    #[test]
    fn test_full_key_trailing_slash() {
        let prefix = "models/whisper/";
        let path = "config.json";
        let expected = "models/whisper/config.json";

        let result = if prefix.is_empty() {
            path.to_string()
        } else {
            format!("{}/{}", prefix.trim_end_matches('/'), path)
        };

        assert_eq!(result, expected);
    }
}
