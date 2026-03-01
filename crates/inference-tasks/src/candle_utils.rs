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
use inference_core::DeviceType;
use tracing::{debug, info, warn};

use crate::error::{TaskError, TaskResult};

/// Resolve a device type to a Candle device.
///
/// Handles platform-specific device initialization with automatic CPU fallback
/// when the requested device feature is not enabled.
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
/// use inference_tasks::candle_utils::resolve_device;
///
/// let device = resolve_device(&DeviceType::Metal)?;
/// ```
pub fn resolve_device(device_type: &DeviceType) -> TaskResult<Device> {
    let resolved = device_type.resolve();

    match resolved {
        DeviceType::Cpu => Ok(Device::Cpu),
        DeviceType::Metal => {
            #[cfg(feature = "candle-metal")]
            {
                Device::new_metal(0)
                    .map_err(|e| TaskError::Config(format!("Metal device error: {e}")))
            }
            #[cfg(not(feature = "candle-metal"))]
            {
                warn!("Metal requested but candle-metal feature not enabled, falling back to CPU");
                Ok(Device::Cpu)
            }
        }
        DeviceType::Cuda => {
            #[cfg(feature = "candle-cuda")]
            {
                Device::new_cuda(0)
                    .map_err(|e| TaskError::Config(format!("CUDA device error: {e}")))
            }
            #[cfg(not(feature = "candle-cuda"))]
            {
                warn!("CUDA requested but candle-cuda feature not enabled, falling back to CPU");
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
}
