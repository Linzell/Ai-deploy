//! WorkerService gRPC implementation.
//!
//! Routes incoming gRPC requests to the Task implementation.
//!
//! ## Error Handling
//!
//! Uses gRPC status codes for transport-level errors:
//! - `NOT_FOUND` - Unknown task name (client error)
//! - `INVALID_ARGUMENT` - Invalid request format (client error)
//! - `INTERNAL` - Server-side errors during task execution
//!
//! Task execution errors are returned in the TaskResponse with `success=false`,
//! allowing clients to distinguish between transport errors and task failures.

use std::sync::Arc;
use std::time::Instant;

use tokio_stream::StreamExt;
use tonic::{Request, Response, Status};
use tracing::{error, info, instrument};

use crate::generated::worker_pb::{
    worker_service_server::WorkerService, TaskChunk as ProtoTaskChunk, TaskRequest, TaskResponse,
};
use crate::task::Task;

/// WorkerService implementation that routes requests to a Task.
pub struct WorkerServiceImpl {
    task: Arc<dyn Task>,
}

impl WorkerServiceImpl {
    /// Create a new WorkerService with the given task.
    pub fn new(task: Arc<dyn Task>) -> Self {
        Self { task }
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

        let task_name = &req.task_name;
        let request_id = if req.request_id.is_empty() {
            "unknown"
        } else {
            &req.request_id
        };

        tracing::Span::current().record("task_name", task_name);
        tracing::Span::current().record("request_id", request_id);

        info!("[{}] ExecuteTask received: {}", request_id, task_name);

        // Validate task name - return NOT_FOUND status for unknown tasks
        if task_name != self.task.name() {
            let error_msg = format!(
                "Unknown task: '{}'. This worker handles: '{}'",
                task_name,
                self.task.name()
            );
            error!("[{}] {}", request_id, error_msg);
            return Err(Status::not_found(error_msg));
        }

        // Execute task
        let result = self.task.execute(&req.payload, request_id).await;
        #[allow(clippy::cast_possible_truncation)]
        let duration_ms = start.elapsed().as_millis() as i64;

        if result.success {
            info!("[{}] Task completed in {}ms", request_id, duration_ms);
        } else {
            error!(
                "[{}] Task failed: {}",
                request_id,
                result.error.as_deref().unwrap_or("unknown")
            );
        }

        Ok(Response::new(TaskResponse {
            success: result.success,
            result: result.result.unwrap_or_default(),
            error: result.error.unwrap_or_default(),
            duration_ms,
        }))
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

        let task_name = &req.task_name;
        let request_id = if req.request_id.is_empty() {
            "unknown".to_string()
        } else {
            req.request_id.clone()
        };

        tracing::Span::current().record("task_name", task_name);
        tracing::Span::current().record("request_id", &request_id);

        info!("[{}] StreamTask received: {}", request_id, task_name);

        // Validate task name - return NOT_FOUND status for unknown tasks
        if task_name != self.task.name() {
            let error_msg = format!(
                "Unknown task: '{}'. This worker handles: '{}'",
                task_name,
                self.task.name()
            );
            error!("[{}] {}", request_id, error_msg);
            return Err(Status::not_found(error_msg));
        }

        // Execute streaming task
        let task = Arc::clone(&self.task);
        let payload = req.payload.clone();
        let rid = request_id.clone();

        let stream = async_stream::stream! {
            let mut task_stream = task.execute_stream(&payload, &rid).await;

            while let Some(chunk) = task_stream.next().await {
                yield Ok(ProtoTaskChunk {
                    data: chunk.data.unwrap_or_default(),
                    done: chunk.done,
                    error: chunk.error.unwrap_or_default(),
                });
            }

            let duration_ms = start.elapsed().as_millis();
            info!("[{}] StreamTask completed in {}ms", rid, duration_ms);
        };

        Ok(Response::new(Box::pin(stream)))
    }
}
