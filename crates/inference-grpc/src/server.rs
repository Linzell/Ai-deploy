//! gRPC server combining WorkerService and Health.

use std::sync::Arc;
use std::time::Duration;

use tonic::transport::Server;
use tonic_reflection::server::Builder as ReflectionBuilder;
use tower::limit::ConcurrencyLimitLayer;
use tracing::{info, warn};

#[cfg(feature = "tls")]
use tonic::transport::{Identity, ServerTlsConfig};

use crate::batcher::BatchScheduler;
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

    /// Build the `WorkerServiceImpl`, optionally attaching a `BatchScheduler`.
    fn build_worker_service(&self) -> WorkerServiceImpl {
        let mut worker_service = WorkerServiceImpl::new(
            Arc::clone(&self.task),
            self.config.request_timeout_ms,
            self.config.max_payload_size_bytes,
        );

        // Wire batching if enabled AND the task supports it.
        if self.config.enable_batching && self.task.supports_batching() {
            let channel_capacity = self.config.max_batch_size.saturating_mul(4).max(16);
            let handle = BatchScheduler::spawn(
                Arc::clone(&self.task),
                self.config.max_batch_size,
                Duration::from_millis(self.config.batch_timeout_ms),
                channel_capacity,
            );
            info!(
                "Dynamic batching enabled: max_batch_size={}, batch_timeout_ms={}, channel_capacity={}",
                self.config.max_batch_size, self.config.batch_timeout_ms, channel_capacity
            );
            worker_service = worker_service.with_batch_handle(handle);
        } else if self.config.enable_batching && !self.task.supports_batching() {
            info!(
                "Batching is enabled in config but task '{}' does not support it — using direct execution",
                self.task.name()
            );
        }

        worker_service
    }

    /// Apply TLS configuration to a server builder, if the `tls` feature is enabled
    /// and cert/key paths are configured.
    ///
    /// Returns the (possibly TLS-configured) builder. When the `tls` feature is not
    /// compiled in, this is a no-op that returns the builder unchanged.
    #[cfg(feature = "tls")]
    fn apply_tls(&self, builder: Server) -> Result<Server> {
        if let (Some(cert_path), Some(key_path)) =
            (&self.config.tls_cert_path, &self.config.tls_key_path)
        {
            let cert = std::fs::read(cert_path).map_err(|e| {
                Error::Config(format!(
                    "Failed to read TLS certificate at {cert_path}: {e}"
                ))
            })?;
            let key = std::fs::read(key_path).map_err(|e| {
                Error::Config(format!("Failed to read TLS private key at {key_path}: {e}"))
            })?;

            let identity = Identity::from_pem(cert, key);
            let tls_config = ServerTlsConfig::new().identity(identity);

            info!("TLS enabled: cert={cert_path}, key={key_path}");

            builder
                .tls_config(tls_config)
                .map_err(|e| Error::Config(format!("Failed to configure TLS: {e}")))
        } else {
            info!("TLS disabled (no cert/key configured)");
            Ok(builder)
        }
    }

    #[cfg(not(feature = "tls"))]
    #[allow(clippy::unnecessary_wraps)]
    fn apply_tls(&self, builder: Server) -> Result<Server> {
        if self.config.tls_cert_path.is_some() || self.config.tls_key_path.is_some() {
            warn!(
                "TLS cert/key paths are configured but the 'tls' feature is not enabled — \
                 rebuild with `--features tls` to enable TLS"
            );
        }
        Ok(builder)
    }

    /// Start the gRPC server.
    ///
    /// This will block until the server is shut down.
    pub async fn serve(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.grpc_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {e}")))?;

        info!(
            "Starting gRPC server on {} for task '{}' (tls={})",
            addr,
            self.task.name(),
            self.config.tls_enabled(),
        );

        // Create services
        let worker_service = self.build_worker_service();
        let health_service = HealthServiceImpl::new(
            Arc::clone(&self.task),
            &format!("maiia.worker.{}", self.task.name()),
        );

        // Create reflection service
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .map_err(|e| Error::Config(format!("Failed to build reflection service: {e}")))?;

        // Apply TLS before .layer() (tls_config doesn't change the generic type)
        let mut builder = self.apply_tls(Server::builder())?;

        // Build and serve (with optional concurrency limit)
        if self.config.max_concurrent_requests > 0 {
            info!(
                "Concurrency limit: {} max in-flight requests",
                self.config.max_concurrent_requests
            );
            builder
                .layer(ConcurrencyLimitLayer::new(
                    self.config.max_concurrent_requests,
                ))
                .add_service(reflection_service)
                .add_service(WorkerServiceServer::new(worker_service))
                .add_service(HealthServer::new(health_service))
                .serve(addr)
                .await
                .map_err(|e| Error::Loader(format!("Server error: {e}")))?;
        } else {
            builder
                .add_service(reflection_service)
                .add_service(WorkerServiceServer::new(worker_service))
                .add_service(HealthServer::new(health_service))
                .serve(addr)
                .await
                .map_err(|e| Error::Loader(format!("Server error: {e}")))?;
        }

        Ok(())
    }

    /// Start the server with graceful shutdown on Ctrl+C and SIGTERM.
    ///
    /// On receiving a shutdown signal:
    /// 1. Stops accepting new connections
    /// 2. Waits for in-flight requests to complete (handled by tonic)
    /// 3. Returns
    pub async fn serve_with_shutdown(self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.grpc_port)
            .parse()
            .map_err(|e| Error::Config(format!("Invalid address: {e}")))?;

        info!(
            "Starting gRPC server on {} for task '{}' (timeout={}ms, max_payload={}B, concurrency={}, tls={})",
            addr,
            self.task.name(),
            self.config.request_timeout_ms,
            self.config.max_payload_size_bytes,
            if self.config.max_concurrent_requests > 0 {
                self.config.max_concurrent_requests.to_string()
            } else {
                "unlimited".to_string()
            },
            self.config.tls_enabled(),
        );

        // Create services
        let worker_service = self.build_worker_service();
        let health_service = HealthServiceImpl::new(
            Arc::clone(&self.task),
            &format!("maiia.worker.{}", self.task.name()),
        );

        // Create reflection service
        let reflection_service = ReflectionBuilder::configure()
            .register_encoded_file_descriptor_set(FILE_DESCRIPTOR_SET)
            .build_v1()
            .map_err(|e| Error::Config(format!("Failed to build reflection service: {e}")))?;

        // Apply TLS before .layer()
        let mut builder = self.apply_tls(Server::builder())?;

        // Build server (with optional concurrency limit)
        if self.config.max_concurrent_requests > 0 {
            info!(
                "Concurrency limit: {} max in-flight requests",
                self.config.max_concurrent_requests
            );
            builder
                .layer(ConcurrencyLimitLayer::new(
                    self.config.max_concurrent_requests,
                ))
                .add_service(reflection_service)
                .add_service(WorkerServiceServer::new(worker_service))
                .add_service(HealthServer::new(health_service))
                .serve_with_shutdown(addr, shutdown_signal())
                .await
                .map_err(|e| Error::Loader(format!("Server error: {e}")))?;
        } else {
            builder
                .add_service(reflection_service)
                .add_service(WorkerServiceServer::new(worker_service))
                .add_service(HealthServer::new(health_service))
                .serve_with_shutdown(addr, shutdown_signal())
                .await
                .map_err(|e| Error::Loader(format!("Server error: {e}")))?;
        }

        info!("Server shutdown complete");
        Ok(())
    }
}

/// Wait for either SIGTERM (Unix) or Ctrl+C.
async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        () = ctrl_c => {
            info!("Received Ctrl+C (SIGINT)");
        }
        () = terminate => {
            warn!("Received SIGTERM");
        }
    }
}
