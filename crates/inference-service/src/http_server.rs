use anyhow::Context;
use inference_core::Task;
use inference_http::HttpServer;
use std::sync::Arc;

/// HTTP server wrapper around Task with graceful shutdown support.
pub struct HttpServerWrapper {
    task: Arc<dyn Task>,
    config: inference_core::Config,
}

impl HttpServerWrapper {
    pub fn new(task: Arc<dyn Task>, config: inference_core::Config) -> Self {
        Self { task, config }
    }

    /// Build the router (needed for embedding in a larger server).
    #[allow(dead_code)]
    pub fn build_router(&self) -> HttpServer {
        HttpServer::new(Arc::clone(&self.task), self.config.clone())
    }

    /// Start the HTTP server with graceful shutdown.
    pub async fn serve(self) -> anyhow::Result<()> {
        let server = HttpServer::new(Arc::clone(&self.task), self.config.clone());
        server.serve().await.context("Failed to start HTTP server")
    }
}
