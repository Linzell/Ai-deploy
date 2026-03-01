//! Task trait for implementing workers.
//!
//! Re-exports the core Task trait from inference-core for use in gRPC services.

pub use inference_core::task::{Task, TaskChunk, TaskResult, TaskStream};
