//! gRPC server combining WorkerService and Health.

use std::sync::Arc;

use tonic::transport::Server;
use tonic_reflection::server::Builder as ReflectionBuilder;
use tracing::info;

use crate::generated::health_pb::health_server::HealthServer;
use crate::generated::worker_pb::worker_service_server::WorkerServiceServer;
use crate::health::HealthServiceImpl;
use crate::task::Task;
use crate::worker::WorkerServiceImpl;
use inference_core::{Config, Error, Result};

/// File descriptor set for gRPC reflection
pub const FILE_DESCRIPTOR_SET: &[u8] = tonic::include_file_descriptor_set!("inference_descriptor");

/// Combined gRPC server with WorkerService and Health.
pub struct WorkerServer {
    task: Arc<dyn Task>,
    config: Config,
}

impl WorkerServer {
    /// Create a new worker server.
    ///
    /// Accepts any type implementing Task (boxed or not).
    pub fn new(task: impl Task + 'static, config: Config) -> Self {
        Self {
            task: Arc::new(task),
            config,
        }
    }

    /// Create from a boxed task.
    pub fn from_boxed(task: Box<dyn Task>, config: Config) -> Self {
        Self {
            task: Arc::from(task),
            config,
        }
    }

    /// Start the gRPC server.
    ///
    /// This will block until the server is shut down.
    pub async fn serve(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.grpc_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {e}")))?;

        info!(
            "Starting gRPC server on {} for task '{}'",
            addr,
            self.task.name()
        );

        // Create services
        let worker_service = WorkerServiceImpl::new(Arc::clone(&self.task));
        let health_service = HealthServiceImpl::new(
            Arc::clone(&self.task),
            &format!("maiia.worker.{}", self.task.name()),
        );

        // Create reflection service
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .map_err(|e| Error::Config(format!("Failed to build reflection service: {e}")))?;

        // Build and serve
        Server::builder()
            .add_service(reflection_service)
            .add_service(WorkerServiceServer::new(worker_service))
            .add_service(HealthServer::new(health_service))
            .serve(addr)
            .await
            .map_err(|e| Error::Loader(format!("Server error: {e}")))?;

        Ok(())
    }

    /// Start the server with graceful shutdown on Ctrl+C.
    pub async fn serve_with_shutdown(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.grpc_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {e}")))?;

        info!(
            "Starting gRPC server on {} for task '{}'",
            addr,
            self.task.name()
        );

        // Create services
        let worker_service = WorkerServiceImpl::new(Arc::clone(&self.task));
        let health_service = HealthServiceImpl::new(
            Arc::clone(&self.task),
            &format!("maiia.worker.{}", self.task.name()),
        );

        // Create reflection service
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .map_err(|e| Error::Config(format!("Failed to build reflection service: {e}")))?;

        // Build server
        let server = Server::builder()
            .add_service(reflection_service)
            .add_service(WorkerServiceServer::new(worker_service))
            .add_service(HealthServer::new(health_service))
            .serve_with_shutdown(addr, async {
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to install Ctrl+C handler");
                info!("Received shutdown signal");
            });

        server
            .await
            .map_err(|e| Error::Loader(format!("Server error: {e}")))?;

        info!("Server shutdown complete");
        Ok(())
    }
}
