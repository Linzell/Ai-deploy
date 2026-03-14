//! Dynamic batching middleware for inference requests.
//!
//! The [`BatchScheduler`] collects individual gRPC requests into batches,
//! dispatches them to [`Task::execute_batch()`], and demultiplexes the
//! results back to the original callers.
//!
//! ## Design
//!
//! ```text
//!  gRPC handler 1 ──► ┌───────────────┐
//!  gRPC handler 2 ──► │ BatchScheduler │ ──► Task::execute_batch()
//!  gRPC handler N ──► └───────────────┘          │
//!        ▲                                       │
//!        └── oneshot result ◄─────────────────────┘
//! ```
//!
//! - Each caller sends a [`BatchRequest`] through an mpsc channel.
//! - A background Tokio task collects requests until either:
//!   - `max_batch_size` requests are queued, or
//!   - `batch_timeout` elapses since the first request in the current batch.
//! - The batch is then dispatched to `Task::execute_batch()`.
//! - Results are sent back to callers via oneshot channels.

use std::sync::Arc;
use std::time::Duration;

use tokio::sync::{mpsc, oneshot};
use tracing::{debug, error, info, warn};

use crate::task::Task;
use inference_core::task::TaskResult;

/// A single request submitted to the batcher.
struct BatchRequest {
    /// JSON payload for the task.
    payload: String,
    /// Request ID for tracing.
    request_id: String,
    /// Channel to send the result back to the gRPC handler.
    response_tx: oneshot::Sender<TaskResult>,
}

/// Handle used by gRPC handlers to submit requests to the batcher.
#[derive(Clone)]
pub struct BatchHandle {
    tx: mpsc::Sender<BatchRequest>,
}

impl BatchHandle {
    /// Submit a request to the batcher and wait for the result.
    ///
    /// Returns `None` if the batcher has been shut down (channel closed).
    pub async fn submit(&self, payload: String, request_id: String) -> Option<TaskResult> {
        let (response_tx, response_rx) = oneshot::channel();
        let request = BatchRequest {
            payload,
            request_id,
            response_tx,
        };

        // Send to the batcher; if the channel is closed, the batcher is gone.
        if self.tx.send(request).await.is_err() {
            return None;
        }

        // Wait for the result.
        response_rx.await.ok()
    }
}

/// Dynamic batch scheduler.
///
/// Collects individual requests and dispatches them as batches to the task.
/// Created via [`BatchScheduler::spawn()`], which returns a [`BatchHandle`]
/// for submitting requests.
pub struct BatchScheduler;

impl BatchScheduler {
    /// Spawn the batch scheduler as a background Tokio task.
    ///
    /// # Arguments
    ///
    /// * `task` - The inference task to execute batches on.
    /// * `max_batch_size` - Maximum number of requests per batch.
    /// * `batch_timeout` - Maximum time to wait for a full batch before
    ///   dispatching a partial one.
    /// * `channel_capacity` - Bounded mpsc channel capacity. When full,
    ///   submitters will apply backpressure (await). A good default is
    ///   `max_batch_size * 4`.
    ///
    /// # Returns
    ///
    /// A [`BatchHandle`] that callers use to submit requests.
    pub fn spawn(
        task: Arc<dyn Task>,
        max_batch_size: usize,
        batch_timeout: Duration,
        channel_capacity: usize,
    ) -> BatchHandle {
        let (tx, rx) = mpsc::channel::<BatchRequest>(channel_capacity);

        info!(
            max_batch_size,
            batch_timeout_ms = batch_timeout.as_millis() as u64,
            channel_capacity,
            "BatchScheduler started"
        );

        tokio::spawn(Self::run_loop(task, rx, max_batch_size, batch_timeout));

        BatchHandle { tx }
    }

    /// Main loop: collect requests into batches and dispatch them.
    async fn run_loop(
        task: Arc<dyn Task>,
        mut rx: mpsc::Receiver<BatchRequest>,
        max_batch_size: usize,
        batch_timeout: Duration,
    ) {
        loop {
            // Wait for the first request (blocks until one arrives or channel closes).
            let Some(first) = rx.recv().await else {
                info!("BatchScheduler channel closed, shutting down");
                return;
            };

            // Start collecting a batch.
            let mut batch: Vec<BatchRequest> = Vec::with_capacity(max_batch_size);
            batch.push(first);

            // Collect more requests until batch is full or timeout fires.
            let deadline = tokio::time::sleep(batch_timeout);
            tokio::pin!(deadline);

            while batch.len() < max_batch_size {
                tokio::select! {
                    biased;

                    // Prefer draining the channel over timing out.
                    maybe_req = rx.recv() => {
                        match maybe_req {
                            Some(req) => batch.push(req),
                            None => {
                                // Channel closed — dispatch whatever we have, then exit.
                                break;
                            }
                        }
                    }
                    () = &mut deadline => {
                        // Timeout — dispatch partial batch.
                        break;
                    }
                }
            }

            let batch_size = batch.len();
            debug!(batch_size, "Dispatching batch");

            // Build the payload slice for execute_batch.
            let payloads: Vec<(&str, &str)> = batch
                .iter()
                .map(|r| (r.payload.as_str(), r.request_id.as_str()))
                .collect();

            // Execute the batch.
            let results = task.execute_batch(&payloads).await;

            // Demultiplex results back to individual callers.
            if results.len() == batch_size {
                for (request, result) in batch.into_iter().zip(results) {
                    // If the receiver has been dropped (timeout on the gRPC side),
                    // sending will fail — that's fine, just discard.
                    if request.response_tx.send(result).is_err() {
                        warn!("Batch result dropped — caller already gone");
                    }
                }
            } else {
                // This should never happen if execute_batch is implemented correctly,
                // but guard against it.
                error!(
                    expected = batch_size,
                    got = results.len(),
                    "execute_batch returned wrong number of results"
                );

                // Send errors to any callers that didn't get a result.
                for (i, request) in batch.into_iter().enumerate() {
                    let result = results.get(i).cloned().unwrap_or_else(|| {
                        TaskResult::err("Batch result count mismatch".to_string())
                    });
                    let _ = request.response_tx.send(result);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// A mock task that counts how many times execute and execute_batch are called.
    struct MockBatchableTask {
        execute_count: AtomicUsize,
        execute_batch_count: AtomicUsize,
    }

    impl MockBatchableTask {
        fn new() -> Self {
            Self {
                execute_count: AtomicUsize::new(0),
                execute_batch_count: AtomicUsize::new(0),
            }
        }
    }

    #[async_trait]
    impl Task for MockBatchableTask {
        #[allow(clippy::unnecessary_literal_bound)]
        fn name(&self) -> &str {
            "mock.batch.v1"
        }

        async fn execute(&self, payload: &str, _request_id: &str) -> TaskResult {
            self.execute_count.fetch_add(1, Ordering::SeqCst);
            TaskResult::ok(format!(r#"{{"echo": {payload}}}"#))
        }

        async fn execute_batch(&self, payloads: &[(&str, &str)]) -> Vec<TaskResult> {
            self.execute_batch_count.fetch_add(1, Ordering::SeqCst);
            payloads
                .iter()
                .map(|(payload, _rid)| TaskResult::ok(format!(r#"{{"batched_echo": {payload}}}"#)))
                .collect()
        }

        fn supports_batching(&self) -> bool {
            true
        }
    }

    #[tokio::test]
    async fn test_single_request_dispatched() {
        let task = Arc::new(MockBatchableTask::new());
        let handle = BatchScheduler::spawn(
            Arc::clone(&task) as Arc<dyn Task>,
            4,
            Duration::from_millis(50),
            16,
        );

        let result = handle
            .submit(r#""hello""#.to_string(), "req-1".to_string())
            .await
            .expect("should get result");

        assert!(result.success);
        assert!(result.result.unwrap().contains("batched_echo"));
        assert_eq!(task.execute_batch_count.load(Ordering::SeqCst), 1);
        // execute_batch was called, not execute
        assert_eq!(task.execute_count.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn test_batch_collects_multiple_requests() {
        let task = Arc::new(MockBatchableTask::new());
        let handle = BatchScheduler::spawn(
            Arc::clone(&task) as Arc<dyn Task>,
            4,
            Duration::from_millis(200),
            16,
        );

        // Submit 4 requests concurrently — should fill a batch.
        let mut handles = Vec::new();
        for i in 0..4 {
            let h = handle.clone();
            handles.push(tokio::spawn(async move {
                h.submit(format!(r#""msg-{i}""#), format!("req-{i}"))
                    .await
                    .expect("should get result")
            }));
        }

        let results: Vec<TaskResult> = futures::future::join_all(handles)
            .await
            .into_iter()
            .map(|r| r.unwrap())
            .collect();

        assert_eq!(results.len(), 4);
        for r in &results {
            assert!(r.success);
        }

        // Should have been dispatched as 1 batch (4 items = max_batch_size).
        assert_eq!(task.execute_batch_count.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_partial_batch_dispatched_on_timeout() {
        let task = Arc::new(MockBatchableTask::new());
        let handle = BatchScheduler::spawn(
            Arc::clone(&task) as Arc<dyn Task>,
            8, // max_batch_size = 8
            Duration::from_millis(50),
            16,
        );

        // Submit only 2 requests — should dispatch after 50ms timeout.
        let r1 = handle
            .submit(r#""a""#.to_string(), "req-a".to_string())
            .await
            .expect("result");
        let r2 = handle
            .submit(r#""b""#.to_string(), "req-b".to_string())
            .await
            .expect("result");

        assert!(r1.success);
        assert!(r2.success);
        // At least one execute_batch call happened.
        assert!(task.execute_batch_count.load(Ordering::SeqCst) >= 1);
    }

    #[tokio::test]
    async fn test_handle_dropped_shuts_down() {
        let task = Arc::new(MockBatchableTask::new());
        let handle = BatchScheduler::spawn(
            Arc::clone(&task) as Arc<dyn Task>,
            4,
            Duration::from_millis(50),
            16,
        );

        // Drop the handle — batcher loop should detect closed channel and exit.
        drop(handle);

        // Give the background task time to notice.
        tokio::time::sleep(Duration::from_millis(100)).await;
        // No panic, no hang — test passes if it reaches here.
    }
}
