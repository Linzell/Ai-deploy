# Maiia AI Inference Service

Generic inference service in Rust. Single binary that runs **any** HuggingFace model — just launch it and go.

## Features

- **Interactive setup**: Run with no args to configure server mode, device, backend, and model through a guided UI
- **One-command deploy**: `--model Xenova/bge-m3` auto-detects task, backend, and files
- **HuggingFace browser**: `--task text-generation` lists popular models from HF Hub with live search & filter
- **Multi-backend**: ONNX Runtime, Candle (Rust-native LLMs), llama.cpp (GGUF)
- **Dual protocol**: gRPC and HTTP (OpenAI-compatible) — run both simultaneously or individually
- **GPU acceleration**: Auto-detects Metal (macOS) / CUDA (Linux) with auto-configured offloading and flash attention
- **Eager loading**: Models load at startup by default for fast first-request response
- **Production-ready**: gRPC health checks, graceful shutdown, batching, concurrency limits, CORS, body size limits
- **Flexible loading**: HuggingFace Hub, S3/MinIO, or local files
- **Secure**: Authenticated shutdown endpoint, path traversal protection, sanitized error messages, CORS allowlist

## Quick Start

```bash
# Interactive setup (recommended for first use)
cargo run --release

# Run a model directly (auto-detects everything)
cargo run --release -- --model Xenova/bge-m3

# Run an LLM (needs candle feature)
cargo run --release --features candle-metal -- --model Qwen/Qwen2.5-0.5B-Instruct

# Run from a TOML preset
cargo run --release -- --config configs/foundation/bge-m3.toml
```

### Interactive Mode

When launched with no arguments, the service walks you through:

```
Step 1/2: Configure deployment
  ? Select server mode
  > gRPC only  (port 50051)
    HTTP only  (port 8080)
    Both       (gRPC + HTTP)

  ? Select device
  > Auto  (GPU when available, CPU fallback)
    CPU   (no GPU)
    GPU   (Metal on macOS, CUDA on Linux/Win)

  ? Select backend
  > Auto    (detect from model files)
    ONNX    (best for embeddings, classification)
    Candle  (best for modern LLMs: Qwen, Llama)
    Llama   (best for GGUF models)

Step 2/2: Select model
  ? Select a task family
  > NLP          Natural Language Processing
    Audio        Speech & audio processing
    Vision       Image & video processing
    Multimodal   Cross-modal tasks

  ? Select a task
  > text-generation      Text generation (LLMs, chatbots)
    feature-extraction   Embeddings & similarity
    ...

  ? Search models
  > Qwen/Qwen3-0.6B                     17.4M dl
    openai-community/gpt2               14.3M dl
    ...
  Filter: qwen3_  (type to search, 400ms debounce)
```

## CLI Reference

```
inference-service [OPTIONS]

Options:
  -m, --model <MODEL>         HuggingFace model ID (auto-detects task/backend/files)
  -t, --task <TASK>           Browse popular models for a task type
  -c, --config <PATH>         Path to TOML config preset
  -d, --device <DEVICE>       Device: auto, cpu, gpu, cuda, metal (default: auto)
  -b, --backend <BACKEND>     Backend: auto, onnx, candle, llama
      --max-tokens <N>        Max tokens for generation
      --temperature <FLOAT>   Temperature (0.0-2.0)
      --top-p <FLOAT>         Top-p sampling (0.0-1.0)
      --num-threads <N>       CPU threads
      --port <PORT>           gRPC port (default: 50051)
      --http-port <PORT>      HTTP port (default: 8080)
      --n-gpu-layers <N>      GPU layers to offload (llama.cpp)
      --limit <N>             Results to show when browsing (default: 10)
      --eager                 Load model at startup (default: true)
      --no-eager              Defer model loading until first request
```

### CLI Modes

| Mode | Command | What it does |
|------|---------|-------------|
| Interactive | `inference-service` | Guided setup: server mode, device, backend, model |
| Search | `inference-service --task <task>` | Query HF API, show top models by downloads |
| Run (model) | `inference-service --model <id>` | Auto-derive config from HF metadata, start server |
| Run (preset) | `inference-service --config <path>` | Load TOML config, start server |

## Supported Tasks

**NLP**: text-generation, text-classification, token-classification, feature-extraction,
question-answering, summarization, translation, fill-mask, zero-shot-classification,
sentence-similarity

**Audio**: automatic-speech-recognition, text-to-speech, audio-classification

**Vision**: image-classification, object-detection, image-segmentation, depth-estimation,
image-to-text, image-feature-extraction, zero-shot-image-classification

**Multimodal**: visual-question-answering, document-question-answering, image-text-to-text,
audio-text-to-text

## Backends

| Backend | Feature Flag | Use Case | Models |
|---------|--------------|----------|--------|
| ONNX Runtime | (default) | Embeddings, classification, seq2seq, vision | All ONNX models |
| Candle | `candle` / `candle-metal` / `candle-cuda` | Rust-native LLMs | Qwen, Llama, Mistral |
| llama.cpp | `llama` | Quantized GGUF models | Any GGUF model |

```bash
# Build with specific backend
cargo build --release --features candle-metal
cargo build --release --features "candle-metal,llama"

# Default features include: candle, llama, http, grpc
# Minimal build (ONNX + HTTP only):
cargo build --release --no-default-features --features http
```

### GPU Auto-Configuration

When a GPU is detected (or selected), the llama.cpp backend automatically:

- Sets `n_gpu_layers=99` (offload all layers to GPU)
- Enables flash attention (major throughput win on Metal/CUDA)
- Sets KV cache dtype to Q8_0 (reduces memory bandwidth, ~lossless)

This applies whether the GPU was auto-detected or explicitly selected via `--device gpu`.

## API

### HTTP (OpenAI-compatible)

Available when built with the `http` feature (included by default).

```
POST /v1/chat/completions    # Chat completion
POST /v1/completions         # Text completion
POST /v1/embeddings          # Text embeddings
GET  /v1/models              # List loaded models
GET  /health                 # Health check
POST /shutdown               # Graceful shutdown (requires MAIIA_AI_ADMIN_TOKEN)
```

Example:

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen3-0.6B","messages":[{"role":"user","content":"Hello!"}]}'
```

### gRPC

```protobuf
service WorkerService {
  rpc ExecuteTask(TaskRequest) returns (TaskResponse);
}

message TaskRequest {
  string task_name = 1;    // e.g., "maiia.text-generation.v1"
  string payload = 2;      // JSON with model inputs
  string request_id = 3;
}
```

### Testing

```bash
# gRPC health check
grpcurl -plaintext 127.0.0.1:50051 grpc.health.v1.Health/Check

# gRPC echo test
grpcurl -plaintext \
  -d '{"task_name":"maiia.echo.v1", "payload":"{\"message\":\"hello\"}", "request_id":"1"}' \
  127.0.0.1:50051 maiia.worker.v1.WorkerService/ExecuteTask

# HTTP health check
curl http://localhost:8080/health
```

## Configuration

### Auto-derivation (--model)

When using `--model`, the service queries the HuggingFace API and auto-derives:

| Field | Source |
|-------|--------|
| Task type | HF model card `pipeline_tag` |
| Backend | Repo files: `.gguf` -> llama, `.safetensors` + text-gen -> candle, `.onnx` -> onnx |
| ONNX file | Auto-discovered from repo (priority: `onnx/model.onnx` > `model.onnx` > first `.onnx`) |
| Service name | `maiia-{task_type}-worker` |
| Task name | `maiia.{task_type}.v1` |
| n_gpu_layers | 99 if GPU detected and backend is llama (0 otherwise) |

Override any auto-derived value with CLI flags (`--device`, `--backend`, `--max-tokens`, etc.).

### Device Auto-Detection

Device defaults to `auto` — the service probes for GPU support at startup:
- **macOS**: uses Metal if available, falls back to CPU
- **Linux**: uses CUDA if available, falls back to CPU
- **llama.cpp backend**: auto-sets `n_gpu_layers=99`, flash attention, and Q8_0 KV cache when GPU is detected

To force a specific device, use `--device cpu`, `MAIIA_AI_DEVICE=cuda`, or `device = "gpu"` in TOML.
Omitting the device setting enables auto-detection.

### Eager vs Lazy Loading

By default (`--eager`), the model loads at startup before the server accepts connections.
This ensures the first inference request is fast. Use `--no-eager` to defer loading
until the first request arrives (faster startup, slower first request).

### Environment Variables

All settings use `MAIIA_AI_` prefix:

```bash
# Server
MAIIA_AI_GRPC_PORT=50051
MAIIA_AI_HTTP_PORT=8080
MAIIA_AI_DEVICE=auto              # auto (default), cpu, gpu, cuda, metal
MAIIA_AI_LOG_FORMAT=json           # json or text
MAIIA_AI_REQUEST_TIMEOUT_MS=300000
MAIIA_AI_MAX_CONCURRENT_REQUESTS=64
MAIIA_AI_SHUTDOWN_DRAIN_SECONDS=30
MAIIA_AI_MAX_PAYLOAD_SIZE=10485760 # HTTP body size limit in bytes (10 MB)

# Security
MAIIA_AI_ADMIN_TOKEN=secret123     # Required for /shutdown endpoint
MAIIA_AI_CORS_ORIGINS=http://localhost:3000,https://app.example.com

# Logging
RUST_LOG=info                      # Or per-crate: inference_service=debug,inference_onnx=info
```

### TOML Config Presets

Pre-configured presets are available in `configs/` for common models.
Use `--config` to load one:

```bash
inference-service --config configs/foundation/bge-m3.toml
inference-service --config configs/nlp/text-generation.toml
```

Config layering: Rust defaults -> `configs/defaults.toml` -> model config -> env vars -> device auto-detection.

## Security

| Feature | Description |
|---------|-------------|
| CORS | Restricted to explicit origins via `MAIIA_AI_CORS_ORIGINS` (GET+POST only) |
| Body size limit | Configurable max payload size (default: 10 MB) |
| Shutdown auth | `/shutdown` requires `Authorization: Bearer <MAIIA_AI_ADMIN_TOKEN>` |
| Path traversal | Local file reads are sandboxed to the working directory |
| Error sanitization | Clients see generic errors; details logged server-side only |
| Credential-free | No hardcoded secrets — all credentials via env vars |

## Logging

The service uses structured logging with per-crate level control:

| Crate | Default Level | Why |
|-------|---------------|-----|
| `inference_service` | INFO | Startup, config, server status |
| `inference_core` | INFO | Config diagnostics |
| All backend crates | WARN | Suppress routine inference noise |
| `opentelemetry*` | WARN | Suppress SDK internals |
| Third-party (tokenizers, ort, etc.) | WARN | Suppress dependency noise |

Override with `RUST_LOG`:

```bash
RUST_LOG=info cargo run                          # Everything at INFO (verbose)
RUST_LOG=inference_service=info cargo run        # Service only (default-like)
RUST_LOG=inference_onnx=debug,inference_llama=debug cargo run  # Debug specific crates
```

Native C/C++ logs from llama.cpp and ONNX Runtime are suppressed by default.

## Docker

```bash
# Build (default features: candle, llama, http, grpc)
docker build -t maiia-inference .

# Build with specific features
docker build --build-arg FEATURES="candle,llama" -t maiia-inference .

# Run a model directly
docker run --rm -p 50051:50051 -p 8080:8080 maiia-inference --model Xenova/bge-m3

# Run from TOML preset
docker run --rm -p 50051:50051 \
  -v $(pwd)/configs:/app/configs:ro \
  maiia-inference --config /app/configs/foundation/bge-m3.toml

# Docker Compose
docker compose up                              # Default (echo task)
MODEL=Xenova/bge-m3 docker compose up          # Run a model
CONFIG=configs/echo.toml docker compose up     # TOML preset
FEATURES=candle docker compose up --build      # With backend features
```

## Project Structure

```
AI-deploy/
├── Cargo.toml                # Workspace root
├── Dockerfile                # Multi-stage build (pinned base images)
├── docker-compose.yml        # Compose with MinIO
├── configs/                  # TOML presets (optional, for --config mode)
│   ├── defaults.toml         # Shared defaults
│   ├── echo.toml             # Test task (no model)
│   ├── nlp/                  # NLP task presets
│   ├── audio/                # Audio task presets
│   ├── vision/               # Vision task presets
│   ├── multimodal/           # Multimodal task presets
│   └── foundation/           # Production model presets
├── scripts/
│   └── test-all-configs.sh   # Integration test suite
└── crates/
    ├── inference-service/    # Main binary (CLI, interactive UI, HF API client)
    ├── inference-core/       # Config, error types, generation config
    ├── inference-onnx/       # ONNX Runtime backend
    ├── inference-candle/     # Candle backend (Rust-native LLMs)
    ├── inference-llama/      # llama.cpp backend (GGUF models)
    ├── inference-tasks/      # Task registry (routes to backends)
    ├── inference-http/       # HTTP server (OpenAI-compatible API)
    ├── inference-grpc/       # gRPC server, health, batching
    ├── inference-preprocess/ # Tokenizers, image/audio preprocessing
    ├── inference-postprocess/# Output decoding
    ├── inference-loader-hf/  # HuggingFace Hub loader
    └── inference-loader-s3/  # S3/MinIO loader
```

## Development

```bash
cargo test --workspace         # Run all tests
cargo clippy --workspace -- -D warnings  # Lint
cargo fmt --all                # Format code
cargo doc --no-deps --open     # Generate & open docs
```

## License

MIT
