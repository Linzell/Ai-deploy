//! WorkerService gRPC implementation.
//!
//! Routes incoming gRPC requests to the Task implementation.
//!
//! ## Error Handling
//!
//! Uses gRPC status codes for transport-level errors:
//! - `NOT_FOUND` - Unknown task name (client error)
//! - `INVALID_ARGUMENT` - Invalid request format (client error)
//! - `RESOURCE_EXHAUSTED` - Payload too large (client error)
//! - `DEADLINE_EXCEEDED` - Request timed out (server-side timeout)
//! - `INTERNAL` - Server-side errors during task execution
//!
//! Task execution errors are returned in the TaskResponse with `success=false`,
//! allowing clients to distinguish between transport errors and task failures.

use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument, warn};
use uuid::Uuid;

use crate::batcher::BatchHandle;
use crate::generated::worker_pb::{
    worker_service_server::WorkerService, TaskChunk as ProtoTaskChunk, TaskRequest, TaskResponse,
};
use crate::metrics;
use crate::task::Task;

/// WorkerService implementation that routes requests to a Task.
pub struct WorkerServiceImpl {
    task: Arc<dyn Task>,
    /// Per-request timeout (0 = no timeout).
    request_timeout: Duration,
    /// Maximum payload size in bytes.
    max_payload_size: usize,
    /// Optional batch handle — present when batching is enabled and the task supports it.
    batch_handle: Option<BatchHandle>,
}

impl WorkerServiceImpl {
    /// Create a new WorkerService with the given task and limits.
    pub fn new(
        task: Arc<dyn Task>,
        request_timeout_ms: u64,
        max_payload_size_bytes: usize,
    ) -> Self {
        Self {
            task,
            request_timeout: Duration::from_millis(request_timeout_ms),
            max_payload_size: max_payload_size_bytes,
            batch_handle: None,
        }
    }

    /// Attach a batch handle for dynamic batching.
    ///
    /// When set, `execute_task` will route requests through the batcher
    /// instead of calling `task.execute()` directly.
    #[must_use]
    pub fn with_batch_handle(mut self, handle: BatchHandle) -> Self {
        self.batch_handle = Some(handle);
        self
    }

    /// Validate the incoming request payload and parameters.
    #[allow(clippy::result_large_err)] // Status is inherently large in tonic
    fn validate_request(&self, req: &TaskRequest) -> Result<(), Status> {
        // Check payload size
        let payload_len = req.payload.len();
        if payload_len > self.max_payload_size {
            return Err(Status::resource_exhausted(format!(
                "Payload size {} bytes exceeds maximum {} bytes",
                payload_len, self.max_payload_size
            )));
        }

        // Check task_name is not empty
        if req.task_name.is_empty() {
            return Err(Status::invalid_argument("task_name must not be empty"));
        }

        // Check task_name length (prevent abuse)
        if req.task_name.len() > 256 {
            return Err(Status::invalid_argument(
                "task_name must be at most 256 characters",
            ));
        }

        // Check metadata map size (prevent abuse)
        if req.metadata.len() > 64 {
            return Err(Status::invalid_argument(
                "metadata map must have at most 64 entries",
            ));
        }

        Ok(())
    }
}

#[tonic::async_trait]
impl WorkerService for WorkerServiceImpl {
    #[instrument(skip(self, request), fields(task_name, request_id))]
    async fn execute_task(
        &self,
        request: Request<TaskRequest>,
    ) -> Result<Response<TaskResponse>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        // --- Input validation ---
        self.validate_request(&req)?;

        let task_name = &req.task_name;
        let request_id = if req.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.request_id.clone()
        };

        tracing::Span::current().record("task_name", task_name);
        tracing::Span::current().record("request_id", &request_id);

        info!("[{}] ExecuteTask received: {}", request_id, task_name);

        // Warn on task name mismatch but still process the request.
        // Clients may not know the exact task name; blocking them is unhelpful.
        if task_name != self.task.name() {
            warn!(
                "[{}] Task name mismatch: got '{}', this worker handles '{}'. Processing anyway.",
                request_id,
                task_name,
                self.task.name()
            );
        }

        // --- Check if should unload due to idle timeout ---
        if self.task.should_unload() {
            info!(
                "[{}] Unloading model due to idle timeout for task '{}'",
                request_id,
                self.task.name()
            );
            let unload_result = self.task.unload().await;
            if !unload_result.success {
                warn!(
                    "[{}] Failed to unload model: {}",
                    request_id,
                    unload_result.error.as_deref().unwrap_or("unknown")
                );
            }
        }

        // --- Execute task with optional timeout ---
        // Route through batcher if available, otherwise call task.execute() directly.
        let task_future = if let Some(ref batch_handle) = self.batch_handle {
            let bh = batch_handle.clone();
            let payload = req.payload.clone();
            let rid = request_id.clone();
            Box::pin(async move {
                bh.submit(payload, rid).await.unwrap_or_else(|| {
                    inference_core::task::TaskResult::err("Batch scheduler unavailable".to_string())
                })
            })
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = inference_core::task::TaskResult> + Send>,
                >
        } else {
            let task = Arc::clone(&self.task);
            let payload = req.payload.clone();
            let rid = request_id.clone();
            Box::pin(async move { task.execute(&payload, &rid).await })
                as std::pin::Pin<
                    Box<dyn std::future::Future<Output = inference_core::task::TaskResult> + Send>,
                >
        };

        let result = if self.request_timeout.is_zero() {
            task_future.await
        } else {
            match tokio::time::timeout(self.request_timeout, task_future).await {
                Ok(result) => result,
                Err(_elapsed) => {
                    let timeout_ms = self.request_timeout.as_millis();
                    warn!("[{}] Request timed out after {}ms", request_id, timeout_ms);
                    return Err(Status::deadline_exceeded(format!(
                        "Request timed out after {timeout_ms}ms"
                    )));
                }
            }
        };

        #[allow(clippy::cast_possible_truncation)]
        let duration_ms = start.elapsed().as_millis() as i64;

        // --- OTel metrics (no-op when OTEL_ENDPOINT is not set) ---
        let m = metrics::metrics();
        let attrs = &[opentelemetry::KeyValue::new("task", task_name.clone())];
        m.request_count.add(1, attrs);
        #[allow(clippy::cast_precision_loss)]
        m.request_duration_ms.record(duration_ms as f64, attrs);

        if result.success {
            info!("[{}] Task completed in {}ms", request_id, duration_ms);
        } else {
            m.error_count.add(1, attrs);
            error!(
                "[{}] Task failed: {}",
                request_id,
                result.error.as_deref().unwrap_or("unknown")
            );
        }

        let mut response = Response::new(TaskResponse {
            success: result.success,
            result: result.result.unwrap_or_default(),
            error: result.error.unwrap_or_default(),
            duration_ms,
        });
        // Always advertise the worker's task name so clients can discover it.
        if let Ok(val) = self.task.name().parse() {
            response.metadata_mut().insert("x-task-name", val);
        }
        Ok(response)
    }

    type StreamTaskStream =
        std::pin::Pin<Box<dyn tokio_stream::Stream<Item = Result<ProtoTaskChunk, Status>> + Send>>;

    #[instrument(skip(self, request), fields(task_name, request_id))]
    async fn stream_task(
        &self,
        request: Request<TaskRequest>,
    ) -> Result<Response<Self::StreamTaskStream>, Status> {
        let start = Instant::now();
        let req = request.into_inner();

        // --- Input validation ---
        self.validate_request(&req)?;

        let task_name = &req.task_name;
        let request_id = if req.request_id.is_empty() {
            Uuid::new_v4().to_string()
        } else {
            req.request_id.clone()
        };

        tracing::Span::current().record("task_name", task_name);
        tracing::Span::current().record("request_id", &request_id);

        info!("[{}] StreamTask received: {}", request_id, task_name);

        // Warn on task name mismatch but still process the request.
        if task_name != self.task.name() {
            warn!(
                "[{}] Task name mismatch: got '{}', this worker handles '{}'. Processing anyway.",
                request_id,
                task_name,
                self.task.name()
            );
        }

        // Check if should unload due to idle timeout
        if self.task.should_unload() {
            info!(
                "[{}] Unloading model due to idle timeout for task '{}'",
                request_id,
                self.task.name()
            );
            let unload_result = self.task.unload().await;
            if !unload_result.success {
                warn!(
                    "[{}] Failed to unload model: {}",
                    request_id,
                    unload_result.error.as_deref().unwrap_or("unknown")
                );
            }
        }

        // Execute streaming task
        let task = Arc::clone(&self.task);
        let payload = req.payload.clone();
        let rid = request_id.clone();
        let rid_for_log = request_id.clone();
        let stream_task_name = task_name.clone();
        let request_timeout = self.request_timeout;

        let stream = async_stream::stream! {
            let task_stream_fut = task.execute_stream(payload, rid);
            let mut task_stream = if request_timeout.is_zero() {
                task_stream_fut.await
            } else {
                match tokio::time::timeout(request_timeout, task_stream_fut).await {
                    Ok(stream) => stream,
                    Err(_elapsed) => {
                        let timeout_ms = request_timeout.as_millis();
                        warn!("[{}] StreamTask timed out after {}ms", rid_for_log, timeout_ms);
                        yield Err(Status::deadline_exceeded(format!(
                            "StreamTask timed out after {timeout_ms}ms"
                        )));
                        return;
                    }
                }
            };

            while let Some(chunk) = task_stream.next().await {
                yield Ok(ProtoTaskChunk {
                    data: chunk.data.unwrap_or_default(),
                    done: chunk.done,
                    error: chunk.error.unwrap_or_default(),
                });
            }

            let duration_ms = start.elapsed().as_millis();
             info!("[{}] StreamTask completed in {}ms", rid_for_log, duration_ms);

            // OTel metrics (no-op when OTEL_ENDPOINT is not set)
            let m = metrics::metrics();
            let attrs = &[opentelemetry::KeyValue::new("task", stream_task_name)];
            m.request_count.add(1, attrs);
            #[allow(clippy::cast_precision_loss)]
            m.request_duration_ms.record(duration_ms as f64, attrs);
        };

        Ok(Response::new(Box::pin(stream)))
    }
}
