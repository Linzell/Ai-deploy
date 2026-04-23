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
mod interactive;
mod system_metrics;
mod telemetry;

#[cfg(feature = "http")]
mod http_server;

use clap::Parser;
use cli::{Cli, CliMode};
use inference_core::Config;
use inference_tasks::TaskRegistry;
use interactive::{Essentials, ServerMode};
use std::env;
use tracing::{error, info, warn};
use tracing_subscriber::fmt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load .env file if present (before anything reads env vars)
    let _ = dotenvy::dotenv();

    let cli = Cli::parse();

    match cli.mode() {
        CliMode::Browse => {
            // Interactive: essentials -> family -> task -> model -> start
            let selection = interactive::select_model().await?;
            let Some(selection) = selection else {
                return Ok(()); // User pressed Escape
            };
            init_logging();
            run_with_model(&selection.model_id, &selection.essentials, &cli).await
        }
        CliMode::SearchTask(task) => {
            // Interactive: essentials -> model selection for a known task -> start
            let selection = interactive::select_model_for_task_str(&task).await?;
            let Some(selection) = selection else {
                return Ok(()); // User pressed Escape
            };
            init_logging();
            run_with_model(&selection.model_id, &selection.essentials, &cli).await
        }
        CliMode::RunModel(model_id) => {
            init_logging();
            let essentials = Essentials {
                server_mode: ServerMode::Both,
                device: cli.device.clone().unwrap_or_else(|| "auto".to_string()),
                backend: cli.backend.clone().unwrap_or_else(|| "auto".to_string()),
            };
            run_with_model(&model_id, &essentials, &cli).await
        }
        CliMode::RunConfig(config_path) => {
            init_logging();
            run_with_config(&config_path, &cli).await
        }
    }
}

/// Initialize tracing/logging — JSON format if MAIIA_AI_LOG_FORMAT=json.
///
/// Uses `EnvFilter` for fine-grained level control:
/// - `RUST_LOG` env var takes priority if set
/// - Otherwise a sensible default: INFO for the service, WARN for noisy
///   backend crates (ONNX, Candle, tokenizers, hf_hub, etc.)
fn init_logging() {
    use tracing_subscriber::EnvFilter;

    let default_filter = [
        // Service-level: keep INFO for visibility
        "inference_service=info",
        // Core config: keep INFO for startup diagnostics
        "inference_core=info",
        // Task registry: only WARN — model loading milestones are INFO-spammy;
        // the important "Loading model..." and "Task created" are already in inference_service
        "inference_tasks=warn",
        // Backend crates: too noisy at INFO, suppress to WARN
        // Note: inference_llama routes native llama.cpp logs through tracing,
        // so at WARN you'll see model load errors but not the 200-line startup spam.
        // Use RUST_LOG=inference_llama=info to debug model loading issues.
        "inference_onnx=warn",
        "inference_candle=warn",
        "inference_llama=warn",
        "inference_preprocess=warn",
        "inference_postprocess=warn",
        "inference_http=warn",
        "inference_grpc=warn",
        "inference_loader_hf=warn",
        "inference_loader_s3=warn",
        // OpenTelemetry SDK: extremely noisy at INFO
        "opentelemetry=warn",
        "opentelemetry_sdk=warn",
        // Third-party noise
        "tokenizers=warn",
        "hf_hub=warn",
        "onnxruntime=warn",
        "ort=warn",
        "reqwest=warn",
        "hyper=warn",
        "h2=warn",
        "tonic=warn",
        // Catch-all: warn for anything not explicitly set above
        "warn",
    ]
    .join(",");

    let filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new(&default_filter));

    let log_format = env::var("MAIIA_AI_LOG_FORMAT").unwrap_or_default();
    if log_format.eq_ignore_ascii_case("json") {
        fmt::Subscriber::builder()
            .with_env_filter(filter)
            .json()
            .with_target(true)
            .with_thread_ids(true)
            .with_span_list(true)
            .init();
    } else {
        fmt::Subscriber::builder()
            .with_env_filter(filter)
            .with_target(true)
            .with_thread_ids(true)
            .init();
    }
}

/// Safetensors weight size threshold for GGUF auto-routing (2 GB).
/// Models larger than this get auto-routed to GGUF/llama.cpp, which is
/// dramatically faster for autoregressive decoding on any device:
/// - CPU: SIMD-optimized GGUF vs slow F32/broken F16 in Candle
/// - Metal: llama.cpp Metal kernels vs Candle's basic Metal support
/// - CUDA: llama.cpp CUDA kernels vs Candle's CUDA
const LARGE_MODEL_THRESHOLD: u64 = 2 * 1024 * 1024 * 1024;

/// --model mode: fetch HF metadata, auto-derive config, and start server.
async fn run_with_model(model_id: &str, essentials: &Essentials, cli: &Cli) -> anyhow::Result<()> {
    info!(model = model_id, "Fetching model info from HuggingFace");

    let mut model_info = hf_api::get_model_info(model_id).await?;
    let mut effective_model_id = model_id.to_string();

    // If the model has no supported files, search for a compatible variant
    if !model_info.has_supported_files() {
        info!(
            model = model_id,
            "No supported files (.safetensors, .onnx, .gguf, .bin) found, searching for compatible variant..."
        );

        if let Some(variant) = hf_api::find_compatible_variant(model_id).await? {
            info!(
                original = model_id,
                variant = %variant.id,
                "Found compatible variant, using it instead"
            );
            effective_model_id = variant.id.clone();
            model_info = variant;
        } else {
            anyhow::bail!(
                "Model '{model_id}' has no supported files (.safetensors, .onnx, .gguf, .bin) \
                 and no compatible variant was found on HuggingFace. \
                 Try: inference-service --task {task} to browse supported models.",
                task = model_info
                    .pipeline_tag
                    .as_deref()
                    .unwrap_or("text-generation"),
            );
        }
    }

    // Derive task type from HF pipeline_tag
    let task_type = model_info
        .pipeline_tag
        .as_deref()
        .unwrap_or("feature-extraction")
        .to_string();

    // Derive backend + files from repo contents.
    // Convert to owned Strings early — we may replace model_info below.
    let (inferred_backend_ref, inferred_onnx_ref, inferred_gguf_ref) = model_info.infer_backend();
    let mut inferred_backend = inferred_backend_ref.to_string();
    let mut inferred_onnx: Option<String> = inferred_onnx_ref.map(String::from);
    let inferred_gguf: Option<String> = inferred_gguf_ref.map(String::from);

    // If infer_backend returned "onnx" but no ONNX file exists, the model has
    // safetensors for a task Candle doesn't support natively. Try to find an
    // ONNX variant first; if none exists, fall back to Candle for encoder-only
    // tasks or give a clear error for tasks that truly need ONNX.
    if inferred_backend == "onnx" && inferred_onnx.is_none() && !model_info.has_onnx() {
        let task_obj = inference_core::TaskType::new(&task_type);

        info!(
            model = %effective_model_id,
            task = task_type,
            "Model has safetensors but task '{}' has no dedicated Candle backend — searching for ONNX variant...",
            task_type,
        );

        match hf_api::find_compatible_variant(&effective_model_id).await {
            Ok(Some(variant)) if variant.has_onnx() => {
                info!(
                    original = %effective_model_id,
                    variant = %variant.id,
                    "Found ONNX variant, switching model"
                );
                effective_model_id = variant.id.clone();
                let (new_backend, new_onnx, _) = variant.infer_backend();
                inferred_backend = new_backend.to_string();
                inferred_onnx = new_onnx.map(String::from);
                model_info = variant;
            }
            _ => {
                if task_obj.is_encoder_only() {
                    // Encoder-only text tasks can fall back to Candle's BertModel backend
                    info!(
                        model = %effective_model_id,
                        task = task_type,
                        "No ONNX variant found — falling back to Candle encoder backend",
                    );
                    inferred_backend = "candle".to_string();
                } else if task_obj.is_object_detection() {
                    // Object detection (DETR) can fall back to Candle's DETR backend
                    info!(
                        model = %effective_model_id,
                        task = task_type,
                        "No ONNX variant found — falling back to Candle object detection backend",
                    );
                    inferred_backend = "candle".to_string();
                } else if task_obj.is_image_to_text() {
                    // Image-to-text (BLIP) can fall back to Candle's BLIP backend
                    info!(
                        model = %effective_model_id,
                        task = task_type,
                        "No ONNX variant found — falling back to Candle image-to-text backend",
                    );
                    inferred_backend = "candle".to_string();
                } else {
                    anyhow::bail!(
                        "Model '{effective_model_id}' has safetensors but task '{task_type}' \
                         requires an ONNX export and no ONNX variant was found.\n\n\
                         Try one of these:\n  \
                         1. Pick a model that has an ONNX export (look for an 'onnx/' branch)\n  \
                         2. Export the model yourself: optimum-cli export onnx --model {effective_model_id} ./onnx-export/\n  \
                         3. Use an ONNX-exported variant (e.g. from HuggingFace Optimum)",
                    );
                }
            }
        }
    }

    // Allow CLI --backend to override, then interactive essentials, then auto
    let backend_str = match cli.backend.as_deref() {
        Some(b) => b.to_string(),
        None if essentials.backend != "auto" => essentials.backend.clone(),
        _ => inferred_backend.clone(),
    };

    // --- GGUF auto-routing for large text-generation models ---
    // For large models (>2GB safetensors), llama.cpp with GGUF quantized weights
    // is dramatically faster for autoregressive decoding on ALL devices:
    // - CPU: SIMD-optimized GGUF (~4GB Q4_K_M) vs broken F16 / impractical F32
    // - Metal: llama.cpp Metal kernels >> Candle Metal for token-by-token generation
    // - CUDA: llama.cpp CUDA kernels >> Candle CUDA for the same reason
    //
    // Metal is auto-compiled on macOS via target-specific deps, and CUDA requires
    // explicit --features all-cuda.

    // Allow CLI --device to override, then interactive essentials, then auto-detect
    let effective_device = match cli.device.as_deref() {
        Some(d) => inference_core::DeviceType::from(d),
        None if essentials.device != "auto" => {
            inference_core::DeviceType::from(essentials.device.as_str())
        }
        None => inference_core::auto_detect_device(),
    };

    // --- GGUF auto-routing for large text-generation models ---
    // For large models (>2GB safetensors), llama.cpp with GGUF quantized weights
    // is dramatically faster for autoregressive decoding on ALL devices:
    // - CPU: SIMD-optimized GGUF (~4GB Q4_K_M) vs broken F16 / impractical F32
    // - Metal: llama.cpp Metal kernels >> Candle Metal for token-by-token generation
    // - CUDA: llama.cpp CUDA kernels >> Candle CUDA for the same reason
    //
    // Metal is auto-compiled on macOS via target-specific deps, and CUDA requires
    // explicit --features all-cuda.

    let gpu_actually_available = inference_tasks::is_gpu_compiled();
    let _is_gpu = match &effective_device {
        inference_core::DeviceType::Cpu => false,
        inference_core::DeviceType::Metal
        | inference_core::DeviceType::Cuda
        | inference_core::DeviceType::Gpu => {
            if gpu_actually_available {
                true
            } else {
                info!(
                    device = ?effective_device,
                    "GPU detected but not compiled in (on Linux, build with --features all-cuda). \
                     Effective device: CPU"
                );
                false
            }
        }
        inference_core::DeviceType::Auto => gpu_actually_available,
    };
    let is_large = model_info.is_large_model(LARGE_MODEL_THRESHOLD);
    let is_text_gen = task_type == "text-generation";
    let already_gguf = backend_str == "llama";

    // Final resolved values
    let (final_backend, final_model_id, final_onnx_file, final_gguf_file) = if is_large
        && is_text_gen
        && !already_gguf
    {
        info!(
            model = %effective_model_id,
            safetensors_mb = model_info.safetensors_size() / (1024 * 1024),
            device = ?effective_device,
            "Large text-gen model — searching for GGUF variant (llama.cpp is faster for autoregressive decoding)"
        );
        match hf_api::find_gguf_variant(&effective_model_id).await {
            Ok(Some(gguf_info)) => {
                let gguf_model_id = gguf_info.id.clone();
                let gguf_file = gguf_info.find_gguf_file().map(String::from);
                info!(
                    original = %effective_model_id,
                    gguf_variant = %gguf_model_id,
                    gguf_file = ?gguf_file,
                    "Auto-routing to GGUF variant via llama.cpp"
                );
                ("llama".to_string(), gguf_model_id, None, gguf_file)
            }
            Ok(None) => {
                warn!(
                    model = %effective_model_id,
                    "No GGUF variant found — falling back to Candle (may be slow for large models)"
                );
                (
                    backend_str,
                    effective_model_id,
                    inferred_onnx,
                    inferred_gguf,
                )
            }
            Err(e) => {
                warn!(
                    model = %effective_model_id,
                    error = %e,
                    "GGUF variant search failed — falling back to Candle"
                );
                (
                    backend_str,
                    effective_model_id,
                    inferred_onnx,
                    inferred_gguf,
                )
            }
        }
    } else {
        (
            backend_str,
            effective_model_id,
            inferred_onnx,
            inferred_gguf,
        )
    };

    info!(
        model = %final_model_id,
        task = task_type,
        backend = final_backend,
        onnx_file = ?final_onnx_file,
        gguf_file = ?final_gguf_file,
        has_safetensors = model_info.has_safetensors(),
        has_onnx = model_info.has_onnx(),
        has_gguf = model_info.has_gguf(),
        "Auto-derived configuration"
    );

    // Use the already-resolved effective device string for config
    let effective_device_str = match &effective_device {
        inference_core::DeviceType::Auto => "auto",
        inference_core::DeviceType::Cpu => "cpu",
        inference_core::DeviceType::Gpu => "gpu",
        inference_core::DeviceType::Cuda => "cuda",
        inference_core::DeviceType::Metal => "metal",
    };

    let config = Config::from_model_args(
        &final_model_id,
        &task_type,
        &final_backend,
        final_onnx_file.as_deref(),
        final_gguf_file.as_deref(),
        Some(effective_device_str),
        cli.max_tokens,
        cli.temperature,
        cli.top_p,
        cli.num_threads,
        cli.port,
        cli.http_port,
        cli.n_gpu_layers,
    );

    start_server(config, essentials.server_mode, cli.eager).await
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
    if let Some(v) = cli.http_port {
        config.http_port = v;
    }
    if let Some(v) = cli.n_gpu_layers {
        config.n_gpu_layers = v;
    }

    start_server(config, ServerMode::Both, cli.eager).await
}

/// When a GGUF model fails to load via llama.cpp, try to find the original
/// safetensors model on HuggingFace and build a new Config for the Candle backend.
async fn try_fallback_to_safetensors(config: &Config) -> anyhow::Result<Config> {
    let model_id = config.model_path.as_deref().ok_or_else(|| {
        anyhow::anyhow!("No model path in config")
    })?;

    let variant = hf_api::find_safetensors_variant(model_id).await?;
    let variant = variant.ok_or_else(|| {
        anyhow::anyhow!(
            "No safetensors variant found for '{model_id}'. \
             The GGUF format may be incompatible with this llama.cpp version. \
             Try a different model or quantization."
        )
    })?;

    // Build a new config pointing to the safetensors model with Candle backend
    let device_str = match &config.device {
        inference_core::DeviceType::Auto => "auto",
        inference_core::DeviceType::Cpu => "cpu",
        inference_core::DeviceType::Gpu => "gpu",
        inference_core::DeviceType::Cuda => "cuda",
        inference_core::DeviceType::Metal => "metal",
    };

    let new_config = Config::from_model_args(
        &variant.id,
        &config.task_type.to_string(),
        "candle", // Switch to Candle backend
        None,     // No ONNX file
        None,     // No GGUF file
        Some(device_str),
        Some(config.max_tokens),
        Some(config.temperature),
        Some(config.top_p),
        Some(config.num_threads),
        Some(config.grpc_port),
        Some(config.http_port),
        Some(config.n_gpu_layers),
    );

    Ok(new_config)
}

/// Validate config, create task, and start server.
///
/// The `server_mode` parameter controls which protocol to serve on,
/// selected interactively or falling back to `Both`.
/// The `eager` flag controls whether the model is loaded at startup (true)
/// or deferred until the first request (false).
async fn start_server(config: Config, server_mode: ServerMode, eager: bool) -> anyhow::Result<()> {
    if let Err(e) = config.validate() {
        error!("Configuration validation failed: {}", e);
        return Err(e.into());
    }

    // --- OpenTelemetry (env-gated) ---
    // Initialize OTel if OTEL_ENDPOINT is set. The guard must stay alive
    // for the lifetime of the process; it is shut down after the server stops.
    let otel_guard = telemetry::try_init_otel(&config);

    // Warn if the config reports a GPU device but GPU isn't compiled in.
    // Metal is auto-compiled on macOS, but CUDA requires --features all-cuda on Linux.
    let gpu_compiled = inference_tasks::is_gpu_compiled();
    let effective_device = if !gpu_compiled && config.device.is_gpu() {
        warn!(
            config_device = ?config.device,
            "Config reports GPU device but no GPU backend is compiled in \
             (on Linux, build with --features all-cuda). \
             Effective runtime device: CPU"
        );
        "CPU (GPU not compiled)"
    } else {
        match &config.device {
            inference_core::DeviceType::Cpu => "CPU",
            inference_core::DeviceType::Metal => "Metal",
            inference_core::DeviceType::Cuda => "CUDA",
            inference_core::DeviceType::Gpu => "GPU",
            inference_core::DeviceType::Auto => "Auto",
        }
    };

    info!(
        task_type = %config.task_type,
        task_name = %config.effective_task_name(),
        model_path = ?config.model_path,
        model_source = ?config.model_source,
        backend = ?config.backend,
        device = effective_device,
        grpc_port = config.grpc_port,
        "Configuration loaded"
    );

    // Create task — eager or lazy loading based on --eager flag
    let task = if eager {
        // Eager: create and load the model immediately before starting the server.
        // This ensures the first request is fast, but startup takes longer.
        info!("Loading model eagerly at startup (--eager enabled)");
        match TaskRegistry::create(&config).await {
            Ok(t) => t,
            Err(e) => {
                // If this is a GGUF/llama model that failed to load, try to fall back
                // to the same model's safetensors weights via Candle.
                // This is NOT a different model — it's the same weights in a different
                // format. GGUF is a quantized repackaging of the original safetensors.
                let err_msg = format!("{e}");
                if config.backend == inference_core::BackendType::Llama
                    && config.gguf_file.is_some()
                {
                    warn!(
                        error = %err_msg,
                        "GGUF model failed to load via llama.cpp — \
                         attempting to load the same model via Candle (safetensors format)"
                    );
                    match try_fallback_to_safetensors(&config).await {
                        Ok(fallback_config) => {
                            info!(
                                original_model = ?config.model_path,
                                fallback_model = ?fallback_config.model_path,
                                "Found same model with safetensors weights — \
                                 retrying with Candle backend (same weights, different format)"
                            );
                            match TaskRegistry::create(&fallback_config).await {
                                Ok(t) => t,
                                Err(fe) => {
                                    error!("Safetensors fallback also failed: {}", fe);
                                    return Err(anyhow::anyhow!(
                                        "GGUF load failed: {err_msg}\n\
                                         Safetensors fallback also failed: {fe}\n\n\
                                         The GGUF format may be incompatible with this llama.cpp version. \
                                         This is a known issue with Qwen3.5/Qwen3.6 models — \
                                         their rope.dimension_sections format is not yet supported."
                                    ));
                                }
                            }
                        }
                        Err(fe) => {
                            error!("No safetensors variant found: {fe}");
                            return Err(anyhow::anyhow!(
                                "GGUF load failed: {err_msg}\n\n\
                                 No safetensors version of this model was found on HuggingFace. \
                                 The GGUF format may be incompatible with this llama.cpp version. \
                                 This is a known issue with Qwen3.5/Qwen3.6 models."
                            ));
                        }
                    }
                } else {
                    error!("Failed to create and load task: {}", e);
                    return Err(anyhow::anyhow!("{e}"));
                }
            }
        }
    } else {
        // Lazy: defer model loading until the first request arrives.
        // Fast startup but first request is slow.
        match TaskRegistry::create_lazy(&config).await {
            Ok(t) => t,
            Err(e) => {
                error!("Failed to create task: {}", e);
                return Err(anyhow::anyhow!("{e}"));
            }
        }
    };

    let task_name = task.name();
    if eager {
        info!(task_name, "Task created and model loaded (eager mode)");
    } else {
        info!(task_name, "Task created (lazy loading — model loads on first request)");
    }

    // Start server based on runtime server_mode selection
    match server_mode {
        ServerMode::Grpc => {
            #[cfg(feature = "grpc")]
            {
                start_grpc_server(task, config, otel_guard).await?;
            }
            #[cfg(not(feature = "grpc"))]
            {
                error!("gRPC server mode selected but 'grpc' feature is not compiled in.");
                return Err(anyhow::anyhow!(
                    "gRPC not available — rebuild with --features grpc"
                ));
            }
        }
        ServerMode::Http => {
            #[cfg(feature = "http")]
            {
                start_http_server(task, config, otel_guard).await?;
            }
            #[cfg(not(feature = "http"))]
            {
                error!("HTTP server mode selected but 'http' feature is not compiled in.");
                return Err(anyhow::anyhow!(
                    "HTTP not available — rebuild with --features http"
                ));
            }
        }
        ServerMode::Both => {
            let http_available = cfg!(feature = "http");
            let grpc_available = cfg!(feature = "grpc");

            if !http_available && !grpc_available {
                error!("No server protocol enabled. Enable 'http' or 'grpc' feature.");
                return Err(anyhow::anyhow!("No server protocol enabled"));
            }

            // If both features are compiled, prefer gRPC (backward compat)
            // If only one is available, use that one.
            if grpc_available {
                #[cfg(feature = "grpc")]
                start_grpc_server(task, config, otel_guard).await?;
            } else {
                #[cfg(feature = "http")]
                start_http_server(task, config, otel_guard).await?;
            }
        }
    }

    Ok(())
}

use http_server::HttpServerWrapper;
#[cfg(feature = "grpc")]
use inference_grpc::WorkerServer;
use std::sync::Arc;

#[cfg(feature = "grpc")]
async fn start_grpc_server(
    task: Box<dyn inference_core::Task>,
    config: Config,
    otel_guard: Option<telemetry::TelemetryGuard>,
) -> anyhow::Result<()> {
    info!("Starting gRPC server");

    let server = WorkerServer::from_boxed(task, config);
    server.serve_with_shutdown().await?;

    // Flush OTel telemetry on shutdown
    if let Some(guard) = otel_guard {
        guard.shutdown();
    }

    Ok(())
}

#[cfg(feature = "http")]
async fn start_http_server(
    task: Box<dyn inference_core::Task>,
    config: Config,
    otel_guard: Option<telemetry::TelemetryGuard>,
) -> anyhow::Result<()> {
    info!("Starting HTTP server (OpenAI-compatible endpoints)");

    let task_arc: Arc<dyn inference_core::Task> = task.into();
    let wrapper = HttpServerWrapper::new(task_arc, config);
    wrapper.serve().await?;

    // Flush OTel telemetry on shutdown
    if let Some(guard) = otel_guard {
        guard.shutdown();
    }

    Ok(())
}
