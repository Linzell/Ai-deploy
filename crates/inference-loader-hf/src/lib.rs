//! # HuggingFace Hub Loader
//!
//! This crate provides a `DataLoader` implementation for HuggingFace Hub.
//! It automatically downloads and caches model files from HuggingFace.

mod loader;

pub use loader::HfLoader;
