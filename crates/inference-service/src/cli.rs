//! CLI argument parsing with clap.
//!
//! Supports multiple modes:
//! - No args: list task families for browsing
//! - `--task <task>`: query HF API and list popular models
//! - `--model <model_id>`: auto-derive config from HF metadata and run
//! - `--config <path>`: legacy TOML preset mode (backward compat)
//! - Override flags: `--device`, `--max-tokens`, `--backend`, etc.

use clap::Parser;

/// Maiia AI Inference Service — deploy any HuggingFace model with one command.
///
/// Examples:
///   inference-service                                    # List available task families
///   inference-service --task text-generation             # Browse popular models for a task
///   inference-service --model Qwen/Qwen2.5-0.5B-Instruct  # Run a model (auto-config)
///   inference-service --config configs/nlp/bge-m3.toml  # Use TOML preset
#[derive(Parser, Debug)]
#[command(name = "inference-service", version, about, long_about = None)]
pub struct Cli {
    /// HuggingFace model ID to deploy (auto-detects task, backend, files).
    ///
    /// Example: --model Qwen/Qwen2.5-0.5B-Instruct
    #[arg(long, short = 'm')]
    pub model: Option<String>,

    /// Browse popular models for a task type (queries HuggingFace API).
    ///
    /// Example: --task text-generation
    #[arg(long, short = 't')]
    pub task: Option<String>,

    /// Path to TOML config file (legacy preset mode).
    ///
    /// Example: --config configs/nlp/bge-m3.toml
    #[arg(long, short = 'c')]
    pub config: Option<String>,

    /// Device for inference: auto, cpu, gpu, cuda, metal.
    /// Defaults to auto (GPU when available, CPU fallback).
    #[arg(long, short = 'd')]
    pub device: Option<String>,

    /// Backend: auto, onnx, candle, llama.
    #[arg(long, short = 'b')]
    pub backend: Option<String>,

    /// Maximum tokens for generation.
    #[arg(long)]
    pub max_tokens: Option<usize>,

    /// Temperature for generation (0.0-2.0).
    #[arg(long)]
    pub temperature: Option<f32>,

    /// Top-p sampling (0.0-1.0).
    #[arg(long)]
    pub top_p: Option<f32>,

    /// Number of CPU threads.
    #[arg(long)]
    pub num_threads: Option<usize>,

    /// gRPC server port.
    #[arg(long)]
    pub port: Option<u16>,

    /// HTTP server port.
    #[arg(long)]
    pub http_port: Option<u16>,

    /// Number of GPU layers to offload (for llama.cpp).
    #[arg(long)]
    pub n_gpu_layers: Option<u32>,

    /// Eagerly load the model at startup instead of waiting for the first request.
    /// Default: true (model loads immediately). Use --no-eager to defer loading.
    #[arg(long, default_missing_value = "true", default_value = "true", action = clap::ArgAction::Set)]
    pub eager: bool,
}

/// Resolved CLI mode based on which arguments were provided.
#[derive(Debug)]
pub enum CliMode {
    /// No args: interactive browsing (family -> task -> model).
    Browse,
    /// `--task <task>`: interactive model selection for a known task.
    SearchTask(String),
    /// `--model <id>`: auto-derive config and start server.
    RunModel(String),
    /// `--config <path>`: legacy TOML preset mode.
    RunConfig(String),
}

impl Cli {
    /// Determine what mode the CLI is in based on provided arguments.
    pub fn mode(&self) -> CliMode {
        if let Some(ref config) = self.config {
            CliMode::RunConfig(config.clone())
        } else if let Some(ref model) = self.model {
            CliMode::RunModel(model.clone())
        } else if let Some(ref task) = self.task {
            CliMode::SearchTask(task.clone())
        } else {
            CliMode::Browse
        }
    }
}
