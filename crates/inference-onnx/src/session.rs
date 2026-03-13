//! ONNX Session builder utilities.
//!
//! This module provides shared session building functionality used by both
//! `OnnxTask` and `Seq2SeqTask`. It handles:
//! - Session configuration with optimization level
//! - Device-specific execution providers (CPU, CUDA)
//! - Thread configuration for CPU inference
//! - Pre-loading external data files into memory to work around ONNX Runtime path bugs
//!
//! ## macOS / Metal
//!
//! ONNX sessions always use CPU on macOS. The `coreml` feature has been removed
//! because CoreML is unreliable for ONNX models (MLProgram crashes, NeuralNetwork
//! runtime failures, context leaks). For GPU on macOS, use the Candle backend
//! (direct Metal) instead.

use std::borrow::Cow;
use std::path::Path;

use inference_core::{Config, DeviceType};
use ort::session::{builder::GraphOptimizationLevel, Session};

use crate::error::{TaskError, TaskResult};

#[cfg(feature = "cuda")]
use tracing::info;
use tracing::{debug, warn};

/// Configure the session builder with the appropriate execution provider.
///
/// On macOS (Metal/Auto), ONNX sessions always use CPU — the `coreml` feature
/// has been removed entirely because CoreML is not production-ready for ONNX.
/// For GPU on macOS, use the Candle backend (direct Metal).
///
/// On Linux/Windows, CUDA acceleration works correctly when the `cuda` feature
/// is enabled.
fn configure_execution_provider(
    mut builder: ort::session::builder::SessionBuilder,
    config: &Config,
) -> TaskResult<ort::session::builder::SessionBuilder> {
    let device = config.device.resolve();

    match device {
        DeviceType::Cpu | DeviceType::Auto | DeviceType::Metal => {
            // Metal falls through to CPU for ONNX — CoreML is unreliable.
            // Candle backend handles Metal GPU acceleration separately.
            builder = builder
                .with_intra_threads(config.num_threads)
                .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;
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
                warn!("CUDA requested but cuda feature not enabled, falling back to CPU");
            }
        }
        DeviceType::Gpu => {
            unreachable!("DeviceType::Gpu should be resolved before reaching session builder");
        }
    }

    Ok(builder)
}

/// Load an ONNX session from a file path with the given configuration.
///
/// This configures the session with:
/// - Optimization level (Level3 for maximum optimization)
/// - Execution provider based on device type (CPU on macOS, CUDA on Linux)
/// - Thread count for CPU inference
/// - External data files pre-loaded into memory (workaround for ONNX Runtime bug)
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

    let mut builder = Session::builder().map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| TaskError::OnnxLoad(e.to_string()))?;

    // Pre-load external data files into memory. The bundled ONNX Runtime C library
    // has a bug where it appends external data filenames to the model *file* path
    // instead of the parent *directory* path (e.g., it tries to open
    // "model.onnx/model.onnx_data" instead of "./model.onnx_data"). By providing
    // the external data as in-memory buffers via AddExternalInitializersFromMemory,
    // we bypass file path resolution entirely.
    let external_data = find_external_data_files(path)?;
    for (filename, buffer) in external_data {
        debug!(file = %filename, "Pre-loading external data file into memory");
        builder = builder
            .with_external_initializer_file_in_memory(&filename, buffer)
            .map_err(|e| {
                TaskError::OnnxLoad(format!("Failed to load external data '{filename}': {e}"))
            })?;
    }

    builder = configure_execution_provider(builder, config)?;

    builder
        .commit_from_file(path)
        .map_err(|e| TaskError::OnnxLoad(format!("{}: {}", path.display(), e)))
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

    builder = configure_execution_provider(builder, config)?;

    builder
        .commit_from_memory(bytes)
        .map_err(|e| TaskError::OnnxLoad(e.to_string()))
}

/// Find and load external data files associated with an ONNX model.
///
/// ONNX models can store large tensors in external data files (e.g., `model.onnx_data`
/// or `model.data`). These files must be in the same directory as the model file.
///
/// We pre-load these files into memory and return them so they can be provided to
/// the session builder via `with_external_initializer_file_in_memory`. This bypasses
/// a bug in the bundled ONNX Runtime C library that incorrectly resolves external
/// data file paths (appends the data filename to the model file path instead of the
/// parent directory path).
///
/// # Arguments
///
/// * `model_path` - Path to the ONNX model file
///
/// # Returns
///
/// A list of `(filename, data)` pairs for any external data files found.
/// Returns an empty list if no external data files exist.
fn find_external_data_files(model_path: &Path) -> TaskResult<Vec<(String, Cow<'static, [u8]>)>> {
    let Some(parent) = model_path.parent() else {
        return Ok(Vec::new());
    };

    let Some(model_name) = model_path.file_name() else {
        return Ok(Vec::new());
    };
    let model_name = model_name.to_string_lossy();

    // Common external data file naming conventions:
    // - model.onnx_data  (most common, e.g., HuggingFace models)
    // - model.data        (some ONNX exporters)
    let candidates = [
        format!("{model_name}_data"),
        format!(
            "{}.data",
            model_name.strip_suffix(".onnx").unwrap_or(&model_name)
        ),
    ];

    let mut result = Vec::new();
    for candidate in &candidates {
        let data_path = parent.join(candidate);
        if data_path.exists() {
            let data = std::fs::read(&data_path).map_err(|e| {
                TaskError::ModelLoad(format!(
                    "Failed to read external data file {}: {e}",
                    data_path.display()
                ))
            })?;
            debug!(
                file = %candidate,
                size_mb = data.len() / (1024 * 1024),
                "Found external data file"
            );
            result.push((candidate.clone(), Cow::Owned(data)));
        }
    }

    Ok(result)
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
}
