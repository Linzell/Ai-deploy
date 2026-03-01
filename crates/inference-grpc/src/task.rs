//! Task trait for implementing workers.
//!
//! This mirrors the Python task pattern from maiia-llm-ocr.

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

/// Trait for implementing a task (like Python's execute_task).
///
/// Each worker implements this trait for their specific task type.
///
/// # Example
///
/// ```rust,ignore
/// use inference_grpc::task::{Task, TaskResult, TaskChunk, TaskStream};
/// use async_trait::async_trait;
///
/// struct OcrTask {
///     // your state here
/// }
///
/// #[async_trait]
/// impl Task for OcrTask {
///     fn name(&self) -> &str {
///         "ocr.paddle.v1"
///     }
///
///     async fn execute(&self, payload: &str, request_id: &str) -> TaskResult {
///         // Parse payload JSON
///         // Do OCR
///         // Return result
///         TaskResult::ok(r#"{"text": "extracted text"}"#.to_string())
///     }
///
///     // Optional: implement streaming for LLM tasks
///     fn supports_streaming(&self) -> bool {
///         false
///     }
/// }
/// ```
#[async_trait]
pub trait Task: Send + Sync {
    /// Get the task name (e.g., "ocr.paddle.v1", "llm.chat-completions.v1")
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
}
