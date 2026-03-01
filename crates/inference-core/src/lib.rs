//! # Inference Core
//!
//! Core traits, configuration, and error types for the inference service.
//! This crate defines the common interfaces that all loaders and models must implement.

pub mod config;
pub mod error;
pub mod generation;
pub mod loader;
pub mod model;
pub mod task;

pub use config::{
    BackendType, Config, DataSourceType, DeviceType, HuggingFaceConfig, InferenceConfig,
    ModelConfig, S3Config, ServiceConfig, TaskConfig, TaskType, TomlConfig, ENV_PREFIX,
};
pub use error::{Error, Result, TaskResult};
pub use generation::{GenerationConfig, ModelArchitecture};
pub use loader::DataLoader;
pub use model::Model;
pub use task::{Task, TaskChunk, TaskResult as TaskExecResult, TaskStream};
