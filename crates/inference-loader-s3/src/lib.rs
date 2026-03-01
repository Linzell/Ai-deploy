//! # AWS S3 Loader
//!
//! This crate provides a `DataLoader` implementation for AWS S3.
//! It downloads and caches files from S3 buckets.

mod loader;

pub use loader::S3Loader;
