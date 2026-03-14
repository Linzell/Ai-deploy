//! Shared utilities for Candle-based tasks.
//!
//! This module provides common functionality used across all Candle backends:
//! - Device resolution (CPU, Metal, CUDA)
//! - Safetensors loading (safe, non-mmap)
//! - Weight file discovery (single and sharded models)
//!
//! ## Why Safe Loading?
//!
//! Candle provides `VarBuilder::from_mmaped_safetensors` for memory-mapped loading,
//! but this requires `unsafe` blocks. For serverless and security-focused deployments,
//! we prefer the safe approach of reading files into memory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use inference_core::{CacheDType, DeviceType, KvCacheConfig};
use tracing::{debug, info, warn};

use crate::error::{TaskError, TaskResult};

/// Resolve a device type to a Candle device.
///
/// Metal is auto-enabled on macOS via target-specific dependencies (no feature flag needed).
/// CUDA requires explicit `--features cuda` on Linux/Windows.
///
/// If the feature is compiled in but hardware probe fails at runtime
/// (e.g., running in a VM), falls back to CPU with a warning.
///
/// # Arguments
///
/// * `device_type` - The requested device type from configuration
///
/// # Returns
///
/// A Candle `Device` ready for tensor operations.
///
/// # Example
///
/// ```rust,ignore
/// use inference_core::DeviceType;
/// use inference_candle::resolve_device;
///
/// let device = resolve_device(&DeviceType::Metal)?;
/// ```
/// Resolve a device type to a Candle device.
///
/// Metal is auto-enabled on macOS via target-specific dependencies (no feature flag needed).
/// CUDA requires explicit `--features cuda` on Linux/Windows.
///
/// If hardware probe fails at runtime (e.g., running in a VM), falls back to CPU with a warning.
///
/// # Arguments
///
/// * `device_type` - The requested device type from configuration
///
/// # Returns
///
/// A Candle `Device` ready for tensor operations.
///
/// # Example
///
/// ```rust,ignore
/// use inference_core::DeviceType;
/// use inference_candle::resolve_device;
///
/// let device = resolve_device(&DeviceType::Metal)?;
/// ```
pub fn resolve_device(device_type: &DeviceType) -> TaskResult<Device> {
    let resolved = device_type.resolve();

    match resolved {
        DeviceType::Cpu | DeviceType::Auto => Ok(Device::Cpu),
        DeviceType::Metal => {
            // Metal is auto-compiled on macOS via target-specific deps in Cargo.toml
            #[cfg(target_os = "macos")]
            {
                match Device::new_metal(0) {
                    Ok(dev) => {
                        info!("Using Metal GPU device");
                        Ok(dev)
                    }
                    Err(e) => {
                        warn!("Metal device probe failed ({e}), falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(target_os = "macos"))]
            {
                debug!("Metal not available (macOS only)");
                Ok(Device::Cpu)
            }
        }
        DeviceType::Cuda => {
            #[cfg(all(not(target_os = "macos"), feature = "cuda"))]
            {
                match Device::new_cuda(0) {
                    Ok(dev) => {
                        info!("Using CUDA GPU device");
                        Ok(dev)
                    }
                    Err(e) => {
                        warn!("CUDA device probe failed ({e}), falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
            #[cfg(not(all(not(target_os = "macos"), feature = "cuda")))]
            {
                debug!("CUDA not available (build with --features all-cuda on Linux for GPU)");
                Ok(Device::Cpu)
            }
        }
        DeviceType::Gpu => {
            // This shouldn't happen after resolve(), but handle it anyway
            warn!("Generic GPU resolved to CPU");
            Ok(Device::Cpu)
        }
    }
}

/// Returns `true` if GPU acceleration is compiled into this build of `inference-candle`.
///
/// - macOS → `true` (Metal is auto-enabled via target-specific deps)
/// - non-macOS + `cuda` feature → `true`
/// - Otherwise → `false`
///
/// Use this to decide whether the effective runtime device is truly GPU or
/// will silently fall back to CPU.
#[must_use]
pub fn is_gpu_compiled() -> bool {
    cfg!(target_os = "macos") || cfg!(feature = "cuda")
}

/// Read the model's preferred dtype from `config.json` (`torch_dtype` field).
///
/// Returns `None` if the file can't be read or the field is missing/unrecognized.
pub fn read_model_dtype(config_path: &Path) -> Option<DType> {
    let content = std::fs::read_to_string(config_path).ok()?;
    let json: serde_json::Value = serde_json::from_str(&content).ok()?;
    let torch_dtype = json.get("torch_dtype")?.as_str()?;

    match torch_dtype {
        "bfloat16" => Some(DType::BF16),
        "float16" => Some(DType::F16),
        "float32" => Some(DType::F32),
        _ => None,
    }
}

/// Find all weight files in a model directory.
///
/// Supports multiple formats in order of preference:
/// 1. Single `model.safetensors` file
/// 2. GGUF quantized models (`*.gguf`)
/// 3. Sharded safetensors (`model-00001-of-*.safetensors`)
///
/// # Arguments
///
/// * `model_dir` - Directory containing model weights
///
/// # Returns
///
/// A vector of paths to weight files, sorted for consistent loading order.
///
/// # Errors
///
/// Returns `TaskError::ModelNotFound` if no weight files are found.
pub fn find_weight_files(model_dir: &Path) -> TaskResult<Vec<PathBuf>> {
    // Check for single safetensors
    let safetensors = model_dir.join("model.safetensors");
    if safetensors.exists() {
        return Ok(vec![safetensors]);
    }

    // Check for GGUF (quantized)
    for entry in std::fs::read_dir(model_dir)
        .map_err(|e| TaskError::ModelLoad(format!("Cannot read directory: {e}")))?
    {
        let entry =
            entry.map_err(|e| TaskError::ModelLoad(format!("Directory entry error: {e}")))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "gguf") {
            return Ok(vec![path]);
        }
    }

    // Check for sharded safetensors
    let mut shards = Vec::new();
    for entry in std::fs::read_dir(model_dir)
        .map_err(|e| TaskError::ModelLoad(format!("Cannot read directory: {e}")))?
    {
        let entry =
            entry.map_err(|e| TaskError::ModelLoad(format!("Directory entry error: {e}")))?;
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "safetensors") {
            shards.push(path);
        }
    }

    if !shards.is_empty() {
        shards.sort();
        return Ok(shards);
    }

    Err(TaskError::ModelNotFound(
        "No weight files found (safetensors or gguf)".into(),
    ))
}

/// Check if the weight files are GGUF format (quantized).
///
/// # Arguments
///
/// * `paths` - Vector of weight file paths
///
/// # Returns
///
/// `true` if the first file has a `.gguf` extension.
pub fn is_gguf(paths: &[PathBuf]) -> bool {
    paths.len() == 1 && paths[0].extension().is_some_and(|ext| ext == "gguf")
}

/// Load safetensors files without using memory-mapped I/O.
///
/// This is a safe alternative to `VarBuilder::from_mmaped_safetensors` which requires
/// an unsafe block. Instead, we read the files into memory and construct the VarBuilder
/// from the in-memory tensors.
///
/// # Arguments
///
/// * `paths` - Paths to safetensors files (supports sharded models)
/// * `dtype` - Target data type for tensors (e.g., `DType::F32`, `DType::BF16`)
/// * `device` - Device to load tensors onto
///
/// # Returns
///
/// A `VarBuilder` containing all tensors from the safetensors files.
///
/// # Example
///
/// ```rust,ignore
/// let paths = find_weight_files(&model_dir)?;
/// let vb = load_safetensors_safe(&paths, DType::F32, &device)?;
/// let model = MyModel::load(vb)?;
/// ```
pub fn load_safetensors_safe(
    paths: &[PathBuf],
    dtype: DType,
    device: &Device,
) -> TaskResult<VarBuilder<'static>> {
    let mut all_tensors: HashMap<String, Tensor> = HashMap::new();

    for path in paths {
        debug!(file = %path.display(), "Loading safetensors file");

        // Read the file into memory (safe, no mmap)
        let file_data = std::fs::read(path).map_err(|e| {
            TaskError::ModelLoad(format!(
                "Failed to read safetensors file {}: {e}",
                path.display()
            ))
        })?;

        // Parse the safetensors format
        let tensors = safetensors::SafeTensors::deserialize(&file_data).map_err(|e| {
            TaskError::ModelLoad(format!(
                "Failed to parse safetensors file {}: {e}",
                path.display()
            ))
        })?;

        // Load each tensor, converting dtype and moving to device
        for (name, view) in tensors.tensors() {
            let shape: Vec<usize> = view.shape().to_vec();

            // Convert safetensors dtype to candle dtype
            let view_dtype = convert_safetensors_dtype(view.dtype(), &name)?;

            // Get raw data and create tensor
            let raw_data = view.data();
            let tensor =
                Tensor::from_raw_buffer(raw_data, view_dtype, &shape, device).map_err(|e| {
                    TaskError::ModelLoad(format!("Failed to create tensor {name} on device: {e}"))
                })?;

            // Convert to target dtype if needed
            let tensor = if tensor.dtype() == dtype {
                tensor
            } else {
                tensor.to_dtype(dtype).map_err(|e| {
                    TaskError::ModelLoad(format!(
                        "Failed to convert tensor {name} to {dtype:?}: {e}"
                    ))
                })?
            };

            all_tensors.insert(name, tensor);
        }
    }

    info!(
        num_tensors = all_tensors.len(),
        num_shards = paths.len(),
        "Loaded all tensors into memory"
    );

    Ok(VarBuilder::from_tensors(all_tensors, dtype, device))
}

/// Convert safetensors dtype to candle dtype.
fn convert_safetensors_dtype(st_dtype: safetensors::Dtype, tensor_name: &str) -> TaskResult<DType> {
    match st_dtype {
        safetensors::Dtype::F16 => Ok(DType::F16),
        safetensors::Dtype::BF16 => Ok(DType::BF16),
        safetensors::Dtype::F32 => Ok(DType::F32),
        safetensors::Dtype::F64 => Ok(DType::F64),
        safetensors::Dtype::I32 | safetensors::Dtype::I64 => Ok(DType::I64),
        safetensors::Dtype::U8 => Ok(DType::U8),
        safetensors::Dtype::U32 => Ok(DType::U32),
        other => Err(TaskError::ModelLoad(format!(
            "Unsupported dtype {other:?} for tensor {tensor_name}"
        ))),
    }
}

/// Resolve compute dtype from KV cache configuration.
///
/// In Candle, the KV cache dtype is tied to the model compute dtype — you
/// can't independently quantize the cache. So if the user requests F16 or BF16
/// for the cache, we change the entire model's compute dtype accordingly.
///
/// - F16/BF16: supported, halves memory for weights + KV cache
/// - Q8_0/Q4_0: not natively supported in Candle (requires GGUF quantized models),
///   falls back to F16 with a warning
/// - F32 or None: uses default F32
///
/// If `cache_dtype_k` and `cache_dtype_v` differ, we pick the higher precision
/// of the two to avoid dtype mismatches during attention computation.
///
/// `weight_size_bytes`: total size of weight files on disk. Used on CPU to decide
/// whether F16 is safe (small models) or must be promoted to F32 (large models
/// where F16 activations overflow, producing NaN).
pub fn resolve_compute_dtype(
    kv_cache: &KvCacheConfig,
    model_dtype: Option<DType>,
    device: &Device,
    weight_size_bytes: u64,
) -> DType {
    let raw = resolve_compute_dtype_raw(kv_cache, model_dtype);

    if matches!(device, Device::Cpu) {
        // BF16 matmul is not supported on CPU at all
        if raw == DType::BF16 {
            // Large models: BF16 → F32 (F16 would overflow activations)
            // Small models: BF16 → F16 (saves memory, activations stay in range)
            const LARGE_WEIGHT_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024; // 2 GB
            if weight_size_bytes > LARGE_WEIGHT_THRESHOLD {
                warn!(
                    weight_mb = weight_size_bytes / (1024 * 1024),
                    "Large model on CPU: using F32 instead of BF16/F16 \
                     (F16 causes NaN overflow in large model activations). \
                     For faster inference, use a GGUF model with the llama backend."
                );
                return DType::F32;
            }
            info!("BF16 not supported on CPU, using F16 instead (still half the memory of F32)");
            return DType::F16;
        }

        // F16 on CPU: safe for small models, NaN-prone for large ones
        if raw == DType::F16 {
            const LARGE_WEIGHT_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;
            if weight_size_bytes > LARGE_WEIGHT_THRESHOLD {
                warn!(
                    weight_mb = weight_size_bytes / (1024 * 1024),
                    "Large model on CPU: using F32 instead of F16 \
                     (F16 causes NaN overflow in large model activations). \
                     For faster inference, use a GGUF model with the llama backend."
                );
                return DType::F32;
            }
        }
    }

    raw
}

/// Inner dtype resolution without device constraints.
fn resolve_compute_dtype_raw(kv_cache: &KvCacheConfig, model_dtype: Option<DType>) -> DType {
    // Determine the target from both K and V preferences.
    // If they differ, pick the higher-precision one since Candle requires
    // uniform dtype across model + KV cache.
    let target = match (&kv_cache.cache_dtype_k, &kv_cache.cache_dtype_v) {
        (None, None) => {
            // No explicit KV cache dtype — use model's native dtype if available,
            // otherwise F32. This respects the model's torch_dtype from config.json
            // (e.g., bfloat16 for Qwen2.5, Llama 3.x) instead of always upscaling to F32.
            let dtype = model_dtype.unwrap_or(DType::F32);
            if dtype != DType::F32 {
                info!(
                    dtype = ?dtype,
                    "Using model's native dtype (from config.json torch_dtype)"
                );
            }
            return dtype;
        }
        (Some(k), None) => *k,
        (None, Some(v)) => *v,
        (Some(k), Some(v)) if k == v => *k,
        (Some(k), Some(v)) => {
            // Pick higher precision when K and V differ
            let k_rank = cache_dtype_precision_rank(*k);
            let v_rank = cache_dtype_precision_rank(*v);
            if k_rank >= v_rank {
                *k
            } else {
                *v
            }
        }
    };

    match target {
        CacheDType::F32 => DType::F32,
        CacheDType::F16 => DType::F16,
        CacheDType::BF16 => DType::BF16,
        CacheDType::Q8_0 | CacheDType::Q4_0 => {
            warn!(
                requested = %target,
                "Candle does not support quantized KV cache ({target}). \
                 Use GGUF models via llama.cpp for quantized caching. \
                 Falling back to F16 for reduced memory."
            );
            DType::F16
        }
    }
}

/// Precision rank for `CacheDType` (higher = more precision).
fn cache_dtype_precision_rank(dt: CacheDType) -> u8 {
    match dt {
        CacheDType::Q4_0 => 0,
        CacheDType::Q8_0 => 1,
        CacheDType::F16 | CacheDType::BF16 => 2,
        CacheDType::F32 => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_resolve_device_cpu() {
        let device = resolve_device(&DeviceType::Cpu).unwrap();
        assert!(matches!(device, Device::Cpu));
    }

    #[test]
    fn test_find_weight_files_single() {
        let dir = tempdir().unwrap();
        let model_path = dir.path().join("model.safetensors");
        fs::write(&model_path, b"dummy").unwrap();

        let files = find_weight_files(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0], model_path);
    }

    #[test]
    fn test_find_weight_files_sharded() {
        let dir = tempdir().unwrap();
        fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"a").unwrap();
        fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"b").unwrap();

        let files = find_weight_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
        // Should be sorted
        assert!(files[0].to_string_lossy().contains("00001"));
        assert!(files[1].to_string_lossy().contains("00002"));
    }

    #[test]
    fn test_find_weight_files_gguf() {
        let dir = tempdir().unwrap();
        let gguf_path = dir.path().join("model.gguf");
        fs::write(&gguf_path, b"dummy").unwrap();

        let files = find_weight_files(dir.path()).unwrap();
        assert_eq!(files.len(), 1);
        assert!(is_gguf(&files));
    }

    #[test]
    fn test_find_weight_files_not_found() {
        let dir = tempdir().unwrap();
        let result = find_weight_files(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn test_is_gguf() {
        assert!(is_gguf(&[PathBuf::from("model.gguf")]));
        assert!(!is_gguf(&[PathBuf::from("model.safetensors")]));
        assert!(!is_gguf(&[
            PathBuf::from("a.safetensors"),
            PathBuf::from("b.safetensors")
        ]));
    }

    #[test]
    fn test_resolve_compute_dtype_defaults_to_f32() {
        let kv = KvCacheConfig::default();
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F32
        );
    }

    #[test]
    fn test_resolve_compute_dtype_uses_model_dtype() {
        let kv = KvCacheConfig::default();
        // BF16 model dtype gets downgraded to F16 on CPU (small model)
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::BF16), &Device::Cpu, 0),
            DType::F16
        );
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::F16), &Device::Cpu, 0),
            DType::F16
        );
    }

    #[test]
    fn test_resolve_compute_dtype_large_model_forces_f32_on_cpu() {
        let kv = KvCacheConfig::default();
        let large = 3 * 1024 * 1024 * 1024; // 3 GB
                                            // BF16 model + large weights on CPU → F32
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::BF16), &Device::Cpu, large),
            DType::F32
        );
        // F16 model + large weights on CPU → F32
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::F16), &Device::Cpu, large),
            DType::F32
        );
        // F32 model + large weights on CPU → F32 (already F32, no change)
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::F32), &Device::Cpu, large),
            DType::F32
        );
    }

    #[test]
    fn test_resolve_compute_dtype_kv_overrides_model_dtype() {
        // Explicit KV cache dtype takes precedence over model dtype
        let kv = KvCacheConfig::default().with_cache_dtype(CacheDType::F16);
        assert_eq!(
            resolve_compute_dtype(&kv, Some(DType::BF16), &Device::Cpu, 0),
            DType::F16
        );
    }

    #[test]
    fn test_resolve_compute_dtype_f16() {
        let kv = KvCacheConfig::default().with_cache_dtype(CacheDType::F16);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );
    }

    #[test]
    fn test_resolve_compute_dtype_bf16_on_cpu_downgrades_to_f16() {
        // Candle CPU backend doesn't support BF16 matmul (small model → F16)
        let kv = KvCacheConfig::default().with_cache_dtype(CacheDType::BF16);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );
    }

    #[test]
    fn test_resolve_compute_dtype_quantized_falls_back_to_f16() {
        let kv = KvCacheConfig::default().with_cache_dtype(CacheDType::Q8_0);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );

        let kv = KvCacheConfig::default().with_cache_dtype(CacheDType::Q4_0);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );
    }

    #[test]
    fn test_resolve_compute_dtype_picks_higher_precision() {
        // K=F32, V=F16 -> should pick F32
        let kv = KvCacheConfig::default()
            .with_cache_dtype_k(CacheDType::F32)
            .with_cache_dtype_v(CacheDType::F16);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F32
        );

        // K=F16, V=F32 -> should also pick F32
        let kv = KvCacheConfig::default()
            .with_cache_dtype_k(CacheDType::F16)
            .with_cache_dtype_v(CacheDType::F32);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F32
        );
    }

    #[test]
    fn test_resolve_compute_dtype_single_side() {
        // Only K set
        let kv = KvCacheConfig::default().with_cache_dtype_k(CacheDType::F16);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );

        // Only V set — BF16 on CPU downgrades to F16 (small model)
        let kv = KvCacheConfig::default().with_cache_dtype_v(CacheDType::BF16);
        assert_eq!(
            resolve_compute_dtype(&kv, None, &Device::Cpu, 0),
            DType::F16
        );
    }
}
