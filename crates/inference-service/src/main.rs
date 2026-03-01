//! # Inference Service
//!
//! Main entrypoint for the Maiia AI worker service.
//!
//! ## Configuration
//!
//! Configuration via environment variables with `MAIIA_AI_` prefix:
//!
//! ```bash
//! # Task selection
//! MAIIA_AI_TASK_TYPE=echo       # echo, asr, ocr, chat, embed
//! MAIIA_AI_TASK_NAME=maiia.echo.v1
//!
//! # Model (required for non-echo tasks)
//! MAIIA_AI_MODEL_PATH=/models/my-model.onnx
//! MAIIA_AI_MODEL_SOURCE=local   # local, s3, huggingface
//!
//! # Service
//! MAIIA_AI_GRPC_PORT=50051
//! MAIIA_AI_SERVICE_NAME=maiia-ai-worker
//! ```
//!
//! Or via TOML config file:
//!
//! ```bash
//! # Using environment variable
//! MAIIA_AI_CONFIG_PATH=/config/task.toml cargo run
//!
//! # Or using --config flag
//! cargo run -- --config /config/task.toml
//! ```

use inference_core::Config;
use inference_grpc::WorkerServer;
use inference_tasks::TaskRegistry;
use std::env;
use tracing::{error, info, Level};
use tracing_subscriber::FmtSubscriber;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .with_target(true)
        .with_thread_ids(true)
        .init();

    info!("Starting Maiia AI Worker Service");

    // Parse --config flag and set env var for Config::load()
    let args: Vec<String> = env::args().collect();
    for i in 0..args.len() {
        if args[i] == "--config" || args[i] == "-c" {
            if let Some(path) = args.get(i + 1) {
                env::set_var("MAIIA_AI_CONFIG_PATH", path);
            }
        }
    }

    // Load configuration (TOML file + ENV overrides)
    let config = match Config::load() {
        Ok(c) => c,
        Err(e) => {
            error!("Configuration error: {}", e);
            return Err(e.into());
        }
    };

    // Validate configuration
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        return Err(e.into());
    }

    info!(
        task_type = %config.task_type,
        task_name = %config.effective_task_name(),
        model_source = ?config.model_source,
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
