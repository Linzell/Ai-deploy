//! ONNX Session builder utilities.
//!
//! This module provides shared session building functionality used by both
//! `OnnxTask` and `Seq2SeqTask`. It handles:
//! - Session configuration with optimization level
//! - Device-specific execution providers (CPU, CoreML, CUDA)
//! - Thread configuration for CPU inference
//! - Automatic CPU fallback for seq2seq tasks on CoreML (see `load_session_for_seq2seq`)
//! - Handling of models with external data files (symlink resolution)

use std::path::{Path, PathBuf};

use inference_core::{Config, DeviceType};
use ort::session::{builder::GraphOptimizationLevel, Session};

use crate::error::{TaskError, TaskResult};

#[cfg(any(feature = "coreml", feature = "cuda"))]
use tracing::info;
use tracing::{debug, warn};

/// Load an ONNX session from a file path with the given configuration.
///
/// This configures the session with:
/// - Optimization level (Level3 for maximum optimization)
/// - Execution provider based on device type (CPU, CoreML, CUDA)
/// - Thread count for CPU inference
///
/// # Arguments
///
/// * `path` - Path to the ONNX model file
/// * `config` - Configuration with device and thread settings
///
/// # Errors
///
/// Returns `TaskError::ModelNotFound` if the file doesn't exist,
/// or `TaskError::OnnxLoad` if loading fails.
///
/// # Example
///
/// ```rust,ignore
/// let session = load_session_from_file(Path::new("model.onnx"), &config)?;
/// ```
pub fn load_session_from_file(path: &Path, config: &Config) -> TaskResult<Session> {
    if !path.exists() {
        return Err(TaskError::ModelNotFound(path.display().to_string()));
    }

    // Handle symlinked models with external data files (e.g., HuggingFace Hub cache)
    // ONNX Runtime looks for external data relative to the resolved (canonical) path,
    // not the symlink path. We need to ensure external data files are accessible.
    let load_path = prepare_model_path_for_external_data(path)?;

    let mut builder = Session::builder().map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    // Resolve Gpu to platform-specific backend (Metal on macOS, CUDA elsewhere)
    let device = config.device.resolve();

    // Configure execution provider based on device type
    match device {
        DeviceType::Cpu => {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;
        }
        DeviceType::Metal => {
            // macOS: Use CoreML execution provider (requires coreml feature)
            #[cfg(feature = "coreml")]
            {
                use ort::ep::coreml::{CoreML, ModelFormat};
                info!("Configuring CoreML execution provider for Metal");
                // Use MLProgram format for better dynamic shape support (requires macOS 12+/iOS 15+)
                builder = builder
                    .with_execution_providers([CoreML::default()
                        .with_model_format(ModelFormat::MLProgram)
                        .build()])
                    .map_err(|e| TaskError::OnnxLoad(format!("CoreML setup failed: {e}")))?;
            }
            #[cfg(not(feature = "coreml"))]
            {
                tracing::warn!(
                    "Metal requested but coreml feature not enabled, falling back to CPU"
                );
            }
        }
        DeviceType::Cuda => {
            // Linux/Windows: Use CUDA execution provider (requires cuda feature)
            #[cfg(feature = "cuda")]
            {
                use ort::ep::CUDA;
                info!("Configuring CUDA execution provider");
                builder = builder
                    .with_execution_providers([CUDA::default().build()])
                    .map_err(|e| TaskError::OnnxLoad(format!("CUDA setup failed: {e}")))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA requested but cuda feature not enabled, falling back to CPU");
            }
        }
        DeviceType::Gpu => {
            // Should never reach here after resolve(), but handle gracefully
            unreachable!("DeviceType::Gpu should be resolved before reaching session builder");
        }
    }

    builder
        .commit_from_file(&load_path)
        .map_err(|e| TaskError::OnnxLoad(format!("{}: {}", load_path.display(), e)))
}

/// Load an ONNX session from bytes with the given configuration.
///
/// This is similar to `load_session_from_file` but loads from memory instead.
///
/// # Arguments
///
/// * `bytes` - ONNX model bytes
/// * `config` - Configuration with device and thread settings
///
/// # Errors
///
/// Returns `TaskError::OnnxLoad` if loading fails.
pub fn load_session_from_bytes(bytes: &[u8], config: &Config) -> TaskResult<Session> {
    let mut builder = Session::builder().map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    // Resolve Gpu to platform-specific backend (Metal on macOS, CUDA elsewhere)
    let device = config.device.resolve();

    // Configure execution provider based on device type
    match device {
        DeviceType::Cpu => {
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;
        }
        DeviceType::Metal => {
            #[cfg(feature = "coreml")]
            {
                use ort::ep::coreml::{CoreML, ModelFormat};
                info!("Configuring CoreML execution provider for Metal");
                // Use MLProgram format for better dynamic shape support (requires macOS 12+/iOS 15+)
                builder = builder
                    .with_execution_providers([CoreML::default()
                        .with_model_format(ModelFormat::MLProgram)
                        .build()])
                    .map_err(|e| TaskError::OnnxLoad(format!("CoreML setup failed: {e}")))?;
            }
            #[cfg(not(feature = "coreml"))]
            {
                tracing::warn!(
                    "Metal requested but coreml feature not enabled, falling back to CPU"
                );
            }
        }
        DeviceType::Cuda => {
            #[cfg(feature = "cuda")]
            {
                use ort::ep::CUDA;
                info!("Configuring CUDA execution provider");
                builder = builder
                    .with_execution_providers([CUDA::default().build()])
                    .map_err(|e| TaskError::OnnxLoad(format!("CUDA setup failed: {e}")))?;
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA requested but cuda feature not enabled, falling back to CPU");
            }
        }
        DeviceType::Gpu => {
            unreachable!("DeviceType::Gpu should be resolved before reaching session builder");
        }
    }

    builder
        .commit_from_memory(bytes)
        .map_err(|e| TaskError::OnnxLoad(e.to_string()))
}

/// Load an ONNX session for seq2seq/autoregressive tasks.
///
/// This is a wrapper around `load_session_from_file` that automatically forces
/// CPU execution on macOS/CoreML. CoreML cannot handle autoregressive text
/// generation because it cannot dynamically resize sequence lengths during
/// inference (KV-cache grows each step).
///
/// On Linux/Windows with CUDA, seq2seq tasks still get GPU acceleration.
///
/// # Arguments
///
/// * `path` - Path to the ONNX model file
/// * `config` - Configuration with device and thread settings
///
/// # Note
///
/// If the configured device would resolve to Metal/CoreML on macOS, this
/// function logs a warning and forces CPU execution instead.
pub fn load_session_for_seq2seq(path: &Path, config: &Config) -> TaskResult<Session> {
    let resolved_device = config.device.resolve();

    // CoreML/Metal cannot handle autoregressive generation with dynamic KV-cache
    // Force CPU on macOS for seq2seq tasks
    if resolved_device == DeviceType::Metal {
        warn!(
            "Seq2seq/text-generation tasks require CPU on macOS (CoreML cannot handle \
             dynamic KV-cache sequence lengths). Forcing CPU execution."
        );
        let cpu_config = Config {
            device: DeviceType::Cpu,
            ..config.clone()
        };
        return load_session_from_file(path, &cpu_config);
    }

    // CUDA and CPU work fine for seq2seq
    load_session_from_file(path, config)
}

/// Load an ONNX session for seq2seq from bytes.
///
/// Same as `load_session_for_seq2seq` but loads from memory instead of file.
/// See that function for details on the CoreML limitation and CPU fallback.
pub fn load_session_for_seq2seq_from_bytes(bytes: &[u8], config: &Config) -> TaskResult<Session> {
    let resolved_device = config.device.resolve();

    if resolved_device == DeviceType::Metal {
        warn!(
            "Seq2seq/text-generation tasks require CPU on macOS (CoreML cannot handle \
             dynamic KV-cache sequence lengths). Forcing CPU execution."
        );
        let cpu_config = Config {
            device: DeviceType::Cpu,
            ..config.clone()
        };
        return load_session_from_bytes(bytes, &cpu_config);
    }

    load_session_from_bytes(bytes, config)
}

/// Prepare the model path for loading, handling external data files for symlinked models.
///
/// When loading ONNX models from HuggingFace Hub cache, the model files are symlinks
/// pointing to blob files. ONNX Runtime resolves symlinks and looks for external data
/// files relative to the resolved (blob) path, not the symlink directory.
///
/// This function detects such situations and creates hardlinks in the blob directory
/// so ONNX Runtime can find the external data files.
///
/// # Arguments
///
/// * `path` - Path to the ONNX model file (may be a symlink)
///
/// # Returns
///
/// The path to use for loading (either original or with external data prepared)
fn prepare_model_path_for_external_data(path: &Path) -> TaskResult<PathBuf> {
    // Check if the path is a symlink
    let metadata = std::fs::symlink_metadata(path).map_err(|e| {
        TaskError::ModelLoad(format!(
            "Failed to get metadata for {}: {e}",
            path.display()
        ))
    })?;

    if !metadata.file_type().is_symlink() {
        // Not a symlink, use as-is
        return Ok(path.to_path_buf());
    }

    // Get the symlink directory (where external data files should be)
    let symlink_dir = path
        .parent()
        .ok_or_else(|| TaskError::ModelLoad("Model path has no parent directory".into()))?;

    // Get the model filename (e.g., "model_q4f16.onnx")
    let model_filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| TaskError::ModelLoad("Invalid model filename".into()))?;

    // Check for external data files in the symlink directory
    // External data files follow patterns like: model_q4f16.onnx_data, model_q4f16.onnx_data_1, etc.
    let external_data_pattern = format!("{model_filename}_data");
    let mut external_data_files: Vec<PathBuf> = Vec::new();

    if let Ok(entries) = std::fs::read_dir(symlink_dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            if name_str.starts_with(&external_data_pattern) {
                external_data_files.push(entry.path());
            }
        }
    }

    if external_data_files.is_empty() {
        // No external data files, use the original path
        debug!(
            path = %path.display(),
            "Model is symlink but has no external data files"
        );
        return Ok(path.to_path_buf());
    }

    // Resolve the symlink to get the blob path
    let resolved_path = std::fs::canonicalize(path).map_err(|e| {
        TaskError::ModelLoad(format!("Failed to resolve symlink {}: {e}", path.display()))
    })?;

    let blob_dir = resolved_path
        .parent()
        .ok_or_else(|| TaskError::ModelLoad("Resolved path has no parent directory".into()))?;

    debug!(
        symlink_dir = %symlink_dir.display(),
        blob_dir = %blob_dir.display(),
        num_external_files = external_data_files.len(),
        "Preparing external data files for symlinked model"
    );

    // Create hardlinks in the blob directory for each external data file
    for ext_file in &external_data_files {
        let ext_filename = ext_file
            .file_name()
            .ok_or_else(|| TaskError::ModelLoad("External data file has no name".into()))?;

        let target_path = blob_dir.join(ext_filename);

        // Skip if hardlink already exists
        if target_path.exists() {
            debug!(
                path = %target_path.display(),
                "External data hardlink already exists"
            );
            continue;
        }

        // Resolve the external data symlink to get the actual blob
        let ext_resolved = if ext_file.is_symlink() {
            std::fs::canonicalize(ext_file).map_err(|e| {
                TaskError::ModelLoad(format!(
                    "Failed to resolve external data symlink {}: {e}",
                    ext_file.display()
                ))
            })?
        } else {
            ext_file.clone()
        };

        // Create hardlink in blob directory
        std::fs::hard_link(&ext_resolved, &target_path).map_err(|e| {
            TaskError::ModelLoad(format!(
                "Failed to create hardlink from {} to {}: {e}",
                ext_resolved.display(),
                target_path.display()
            ))
        })?;

        debug!(
            source = %ext_resolved.display(),
            target = %target_path.display(),
            "Created external data hardlink"
        );
    }

    // Return the resolved (canonical) path since external data is now accessible there
    Ok(resolved_path)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_session_file_not_found() {
        let config = Config::default();
        let result = load_session_from_file(Path::new("/nonexistent/model.onnx"), &config);
        assert!(result.is_err());
        assert!(matches!(result, Err(TaskError::ModelNotFound(_))));
    }

    #[test]
    fn test_seq2seq_file_not_found() {
        let config = Config::default();
        let result = load_session_for_seq2seq(Path::new("/nonexistent/model.onnx"), &config);
        assert!(result.is_err());
        assert!(matches!(result, Err(TaskError::ModelNotFound(_))));
    }

    /// Test that seq2seq session loader handles Metal device correctly.
    /// On macOS, Metal should be detected and converted to CPU.
    /// On other platforms, this test just verifies the function doesn't panic.
    #[test]
    fn test_seq2seq_metal_fallback_logic() {
        // Test with Metal device - should work without panicking
        // (actual session loading will fail due to missing file, but that's OK)
        let config = Config {
            device: DeviceType::Metal,
            ..Config::default()
        };
        let result = load_session_for_seq2seq(Path::new("/nonexistent/model.onnx"), &config);
        // Should fail with ModelNotFound (means it got past the device resolution)
        assert!(matches!(result, Err(TaskError::ModelNotFound(_))));
    }

    /// Test that seq2seq session loader handles GPU device correctly.
    /// GPU should resolve to Metal on macOS (then fallback to CPU),
    /// or CUDA on Linux.
    #[test]
    fn test_seq2seq_gpu_fallback_logic() {
        let config = Config {
            device: DeviceType::Gpu,
            ..Config::default()
        };
        let result = load_session_for_seq2seq(Path::new("/nonexistent/model.onnx"), &config);
        // Should fail with ModelNotFound (means it got past the device resolution)
        assert!(matches!(result, Err(TaskError::ModelNotFound(_))));
    }
}
