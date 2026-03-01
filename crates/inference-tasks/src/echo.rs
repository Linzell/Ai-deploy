//! Echo task for testing - no model required.

use async_trait::async_trait;
use inference_core::task::{Task, TaskResult};
use tracing::info;

/// Echo task - returns the input payload for testing.
pub struct EchoTask {
    name: String,
}

impl EchoTask {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

#[async_trait]
impl Task for EchoTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> TaskResult {
        info!(request_id = request_id, "Echo task executing");
        let result = serde_json::json!({
            "echo": payload,
            "task": self.name,
            "request_id": request_id,
        });
        TaskResult::ok(result.to_string())
    }

    fn is_ready(&self) -> bool {
        true
    }
}
