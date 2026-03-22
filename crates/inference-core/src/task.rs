//! Task trait for implementing inference workers.
//!
//! This module defines the core abstraction for all inference tasks,
//! regardless of backend (ONNX, Candle, llama.cpp).

use crate::model::{get_available_models, ModelMetadata};
use async_trait::async_trait;
use std::pin::Pin;
use tokio_stream::Stream;

/// Result of a task execution.
#[derive(Debug, Clone)]
pub struct TaskResult {
    /// Whether the task succeeded
    pub success: bool,
    /// JSON result (if success)
    pub result: Option<String>,
    /// Error message (if failed)
    pub error: Option<String>,
}

impl TaskResult {
    /// Create a successful result
    pub fn ok(result: String) -> Self {
        Self {
            success: true,
            result: Some(result),
            error: None,
        }
    }

    /// Create a failed result
    pub fn err(error: String) -> Self {
        Self {
            success: false,
            result: None,
            error: Some(error),
        }
    }
}

/// A chunk in a streaming response
#[derive(Debug, Clone)]
pub struct TaskChunk {
    /// Chunk data (JSON)
    pub data: Option<String>,
    /// Is this the final chunk?
    pub done: bool,
    /// Error if failed
    pub error: Option<String>,
}

impl TaskChunk {
    /// Create a data chunk
    pub fn data(data: String) -> Self {
        Self {
            data: Some(data),
            done: false,
            error: None,
        }
    }

    /// Create the final chunk with data
    pub fn final_data(data: String) -> Self {
        Self {
            data: Some(data),
            done: true,
            error: None,
        }
    }

    /// Create a done marker (no data)
    pub fn done() -> Self {
        Self {
            data: None,
            done: true,
            error: None,
        }
    }

    /// Create an error chunk
    pub fn error(error: String) -> Self {
        Self {
            data: None,
            done: true,
            error: Some(error),
        }
    }
}

/// Stream of task chunks
pub type TaskStream = Pin<Box<dyn Stream<Item = TaskChunk> + Send>>;

/// Trait for implementing an inference task.
///
/// Each backend (ONNX, Candle, llama.cpp) implements this trait for their specific task types.
///
/// # Example
///
/// ```rust,ignore
/// use inference_core::task::{Task, TaskResult, TaskChunk, TaskStream};
/// use async_trait::async_trait;
///
/// struct MyTask {
///     // your state here
/// }
///
/// #[async_trait]
/// impl Task for MyTask {
///     fn name(&self) -> &str {
///         "my.task.v1"
///     }
///
///     async fn execute(&self, payload: &str, request_id: &str) -> TaskResult {
///         // Parse payload JSON, run inference, return result
///         TaskResult::ok(r#"{"result": "output"}"#.to_string())
///     }
///
///     fn supports_streaming(&self) -> bool {
///         false
///     }
/// }
/// ```
#[async_trait]
pub trait Task: Send + Sync {
    /// Get the task name (e.g., "ocr.paddle.v1", "llm.qwen.v1")
    fn name(&self) -> &str;

    /// Execute the task with the given JSON payload.
    ///
    /// # Arguments
    ///
    /// * `payload` - JSON-encoded task arguments
    /// * `request_id` - Request ID for tracing
    ///
    /// # Returns
    ///
    /// TaskResult with success/error status and JSON result
    async fn execute(&self, payload: &str, request_id: &str) -> TaskResult;

    /// Execute the task with streaming response.
    ///
    /// Default implementation calls `execute` and returns a single chunk.
    /// Override for true streaming (e.g., LLM token streaming).
    ///
    /// # Arguments
    ///
    /// * `payload` - JSON-encoded task arguments
    /// * `request_id` - Request ID for tracing
    ///
    /// # Returns
    ///
    /// Stream of TaskChunk
    async fn execute_stream(&self, payload: &str, request_id: &str) -> TaskStream {
        let result = self.execute(payload, request_id).await;
        let chunk = if result.success {
            TaskChunk::final_data(result.result.unwrap_or_default())
        } else {
            TaskChunk::error(result.error.unwrap_or_else(|| "Unknown error".to_string()))
        };
        Box::pin(tokio_stream::once(chunk))
    }

    /// Whether this task supports true streaming.
    ///
    /// If false, `execute_stream` will call `execute` and return a single chunk.
    fn supports_streaming(&self) -> bool {
        false
    }

    /// Check if the task is ready (model loaded, etc.)
    fn is_ready(&self) -> bool {
        true
    }

    /// Execute a batch of requests.
    ///
    /// Default implementation calls `execute` sequentially for each item.
    /// Override this for backends that support true batch inference (e.g., ONNX
    /// embedding/classification models where multiple inputs can be stacked
    /// into a single forward pass).
    ///
    /// # Arguments
    ///
    /// * `payloads` - Slice of (payload_json, request_id) pairs
    ///
    /// # Returns
    ///
    /// Vec of `TaskResult`, one per input, in the same order.
    async fn execute_batch(&self, payloads: &[(&str, &str)]) -> Vec<TaskResult> {
        let mut results = Vec::with_capacity(payloads.len());
        for (payload, request_id) in payloads {
            results.push(self.execute(payload, request_id).await);
        }
        results
    }

    /// Whether this task supports efficient batch execution.
    ///
    /// If true, the batching middleware will collect multiple requests
    /// and call `execute_batch`. If false, batching middleware will still
    /// work but will just call `execute` sequentially (no benefit).
    fn supports_batching(&self) -> bool {
        false
    }

    /// Get available model names for this task.
    ///
    /// Returns a list of model metadata that this task can handle.
    fn get_available_models(&self) -> Vec<ModelMetadata> {
        get_available_models()
    }
}
