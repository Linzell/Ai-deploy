//! # Inference Service
//!
//! Main entrypoint for the Maiia AI worker service.
//!
//! ## Quick Start
//!
//! ```bash
//! # Browse available task families
//! cargo run --release
//!
//! # Browse popular models for a task
//! cargo run --release -- --task text-generation
//!
//! # Run a model directly (auto-detects everything)
//! cargo run --release -- --model Qwen/Qwen2.5-0.5B-Instruct
//!
//! # Use a TOML preset (legacy mode)
//! cargo run --release -- --config configs/nlp/bge-m3.toml
//!
//! # Override flags work with --model
//! cargo run --release -- --model openai/whisper-base --device gpu --max-tokens 512
//! ```
//!
//! ## Configuration
//!
//! Configuration via environment variables with `MAIIA_AI_` prefix:
//!
//! ```bash
//! MAIIA_AI_TASK_TYPE=echo
//! MAIIA_AI_MODEL_PATH=/models/my-model
//! MAIIA_AI_GRPC_PORT=50051
//! ```

mod cli;
mod hf_api;

use clap::Parser;
use cli::{Cli, CliMode};
use hf_api::{format_downloads, TASK_FAMILIES};
use inference_core::Config;
use inference_grpc::WorkerServer;
use inference_tasks::TaskRegistry;
use std::env;
use tracing::{error, info, Level};
use tracing_subscriber::fmt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();

    match cli.mode() {
        CliMode::Browse => {
            print_task_families();
            return Ok(());
        }
        CliMode::SearchTask(task) => {
            // Only init minimal logging for API queries
            init_logging();
            return search_and_print_models(&task, cli.limit).await;
        }
        CliMode::RunModel(model_id) => {
            init_logging();
            run_with_model(&model_id, &cli).await
        }
        CliMode::RunConfig(config_path) => {
            init_logging();
            run_with_config(&config_path, &cli).await
        }
    }
}

/// Initialize tracing/logging — JSON format if MAIIA_AI_LOG_FORMAT=json.
fn init_logging() {
    let log_format = env::var("MAIIA_AI_LOG_FORMAT").unwrap_or_default();
    if log_format.eq_ignore_ascii_case("json") {
        fmt::Subscriber::builder()
            .json()
            .with_max_level(Level::INFO)
            .with_target(true)
            .with_thread_ids(true)
            .with_span_list(true)
            .init();
    } else {
        fmt::Subscriber::builder()
            .with_max_level(Level::INFO)
            .with_target(true)
            .with_thread_ids(true)
            .init();
    }
}

/// No-args mode: print task families and usage instructions.
fn print_task_families() {
    println!("Maiia AI Inference Service\n");
    println!("Browse models by task family, then run any model with one command.\n");

    for family in TASK_FAMILIES {
        println!("  {} — {}", family.name, family.description);
        for (task, desc) in family.tasks {
            println!("    {task:<40} {desc}");
        }
        println!();
    }

    println!("Usage:");
    println!("  inference-service --task text-generation             # Browse models for a task");
    println!("  inference-service --model Qwen/Qwen2.5-0.5B-Instruct  # Run a model");
    println!("  inference-service --config configs/nlp/bge-m3.toml  # Use a TOML preset");
    println!("\nRun with --help for all options.");
}

/// --task mode: query HF API and print popular models.
async fn search_and_print_models(task: &str, limit: usize) -> anyhow::Result<()> {
    info!(task = task, limit = limit, "Searching HuggingFace models");

    let models = hf_api::search_models(task, limit).await?;

    if models.is_empty() {
        println!("No models found for task '{task}'.");
        println!("Use --task with one of the supported pipeline tags (e.g., text-generation, feature-extraction).");
        return Ok(());
    }

    println!("Popular models for task '{task}':\n");
    println!("  {:<50} {:>12} {:>8}", "MODEL", "DOWNLOADS", "LIKES");
    println!("  {}", "-".repeat(74));

    for m in &models {
        println!(
            "  {:<50} {:>12} {:>8}",
            m.id,
            format_downloads(m.downloads),
            m.likes,
        );
    }

    println!("\nRun a model:");
    if let Some(first) = models.first() {
        println!("  inference-service --model {}", first.id);
    }

    Ok(())
}

/// --model mode: fetch HF metadata, auto-derive config, and start server.
async fn run_with_model(model_id: &str, cli: &Cli) -> anyhow::Result<()> {
    info!(model = model_id, "Fetching model info from HuggingFace");

    let model_info = hf_api::get_model_info(model_id).await?;

    // Derive task type from HF pipeline_tag
    let task_type = model_info
        .pipeline_tag
        .as_deref()
        .unwrap_or("feature-extraction");

    // Derive backend + files from repo contents
    let (backend, onnx_file, gguf_file) = model_info.infer_backend();

    // Allow CLI --backend to override
    let backend = cli.backend.as_deref().unwrap_or(backend);

    info!(
        model = model_id,
        task = task_type,
        backend = backend,
        onnx_file = ?onnx_file,
        gguf_file = ?gguf_file,
        has_safetensors = model_info.has_safetensors(),
        has_onnx = model_info.has_onnx(),
        has_gguf = model_info.has_gguf(),
        "Auto-derived configuration"
    );

    let config = Config::from_model_args(
        model_id,
        task_type,
        backend,
        onnx_file,
        gguf_file,
        cli.device.as_deref(),
        cli.max_tokens,
        cli.temperature,
        cli.top_p,
        cli.num_threads,
        cli.port,
        cli.n_gpu_layers,
    );

    start_server(config).await
}

/// --config mode: load TOML config (legacy path), apply CLI overrides, and start server.
async fn run_with_config(config_path: &str, cli: &Cli) -> anyhow::Result<()> {
    info!(config = config_path, "Loading TOML config");

    // Set env var for Config::load() to pick up
    env::set_var("MAIIA_AI_CONFIG_PATH", config_path);

    let mut config = Config::load().map_err(|e| {
        error!("Configuration error: {}", e);
        anyhow::anyhow!("{e}")
    })?;

    // Apply CLI overrides on top of TOML config
    if let Some(ref d) = cli.device {
        config.device = inference_core::DeviceType::from(d.as_str());
    }
    if let Some(ref b) = cli.backend {
        config.backend = inference_core::BackendType::from(b.as_str());
    }
    if let Some(v) = cli.max_tokens {
        config.max_tokens = v;
    }
    if let Some(v) = cli.temperature {
        config.temperature = v;
    }
    if let Some(v) = cli.top_p {
        config.top_p = v;
    }
    if let Some(v) = cli.num_threads {
        config.num_threads = v;
    }
    if let Some(v) = cli.port {
        config.grpc_port = v;
    }
    if let Some(v) = cli.n_gpu_layers {
        config.n_gpu_layers = v;
    }

    start_server(config).await
}

/// Validate config, create task, and start gRPC server.
async fn start_server(config: Config) -> anyhow::Result<()> {
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        return Err(e.into());
    }

    info!(
        task_type = %config.task_type,
        task_name = %config.effective_task_name(),
        model_path = ?config.model_path,
        model_source = ?config.model_source,
        backend = ?config.backend,
        device = ?config.device,
        grpc_port = config.grpc_port,
        "Configuration loaded"
    );

    // Create task from registry
    let task = match TaskRegistry::create(&config).await {
        Ok(t) => t,
        Err(e) => {
            error!("Failed to create task: {}", e);
            return Err(anyhow::anyhow!("{e}"));
        }
    };

    info!(task_name = task.name(), "Task initialized");

    // Start gRPC server
    let server = WorkerServer::from_boxed(task, config);
    server.serve_with_shutdown().await?;

    Ok(())
}
