//! # Inference gRPC Server
//!
//! This crate provides the gRPC server implementation for the Maiia worker service.
//!
//! ## Services
//!
//! - `WorkerService` - Generic task execution (like Celery)
//! - `Health` - Standard gRPC health checking
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_grpc::server::WorkerServer;
//! use inference_grpc::task::Task;
//!
//! // Implement your task
//! struct MyTask;
//! impl Task for MyTask {
//!     fn name(&self) -> &str { "my.task.v1" }
//!     async fn execute(&self, payload: &str) -> Result<String, String> {
//!         // Your logic here
//!         Ok(r#"{"result": "success"}"#.to_string())
//!     }
//! }
//!
//! // Start server
//! let server = WorkerServer::new(MyTask, config).await?;
//! server.serve().await?;
//! ```

pub mod batcher;
pub mod generated;
pub mod health;
pub mod metrics;
pub mod server;
pub mod task;
pub mod worker;

pub use batcher::BatchHandle;
pub use server::WorkerServer;
pub use task::Task;
