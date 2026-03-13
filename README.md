# AI Inference Service

Generic inference service in Rust. Single binary that runs **any** HuggingFace model — just pass `--model` and go.

## Features

- **One-command deploy**: `--model Xenova/bge-m3` auto-detects task, backend, and files
- **HuggingFace browser**: `--task text-generation` lists popular models from HF Hub
- **Multi-backend**: ONNX Runtime (default), Candle (Rust-native LLMs), llama.cpp (GGUF)
- **Production-ready**: gRPC API, health checks, graceful shutdown, batching, concurrency limits
- **Flexible loading**: HuggingFace Hub, S3/MinIO, or local files

## Quick Start

```bash
# Browse available task families
cargo run --release

# Browse popular models for a task
cargo run --release -- --task text-generation

# Run a model (auto-detects everything)
cargo run --release -- --model Xenova/bge-m3

# Run an LLM (needs candle feature)
cargo run --release --features candle-metal -- --model Qwen/Qwen2.5-0.5B-Instruct

# Run from a TOML preset
cargo run --release -- --config configs/foundation/bge-m3.toml
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
      --port <PORT>            gRPC port (default: 50051)
      --n-gpu-layers <N>      GPU layers to offload (llama.cpp)
      --limit <N>             Results to show when browsing (default: 10)
```

### CLI Modes

| Mode | Command | What it does |
|------|---------|-------------|
| Browse | `inference-service` | List task families (NLP, Audio, Vision, Multimodal) |
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

Override any auto-derived value with CLI flags (`--device`, `--backend`, `--max-tokens`, etc.).

### Device Auto-Detection

Device defaults to `auto` — the service probes for GPU support at startup:
- **macOS**: uses Metal if available, falls back to CPU
- **Linux**: uses CUDA if available, falls back to CPU
- **llama.cpp backend**: auto-sets `n_gpu_layers=99` when GPU is detected

To force a specific device, use `--device cpu`, `MAIIA_AI_DEVICE=cuda`, or `device = "gpu"` in TOML.
Omitting the device setting enables auto-detection.

### Environment Variables

All settings use `MAIIA_AI_` prefix:

```bash
MAIIA_AI_GRPC_PORT=50051
MAIIA_AI_DEVICE=auto              # auto (default), cpu, gpu, cuda, metal
MAIIA_AI_LOG_FORMAT=json           # json or text
MAIIA_AI_REQUEST_TIMEOUT_MS=300000
MAIIA_AI_MAX_CONCURRENT_REQUESTS=64
MAIIA_AI_SHUTDOWN_DRAIN_SECONDS=30
```

### TOML Config Presets

Pre-configured presets are available in `configs/` for common models.
Use `--config` to load one:

```bash
inference-service --config configs/foundation/bge-m3.toml
inference-service --config configs/nlp/text-generation.toml
```

Config layering: Rust defaults -> `configs/defaults.toml` -> model config -> env vars -> device auto-detection.

## API

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
# Health check
grpcurl -plaintext 127.0.0.1:50051 grpc.health.v1.Health/Check

# Echo test
grpcurl -plaintext \
  -d '{"task_name":"maiia.echo.v1", "payload":"{\"message\":\"hello\"}", "request_id":"1"}' \
  127.0.0.1:50051 maiia.worker.v1.WorkerService/ExecuteTask

# Embeddings
grpcurl -plaintext \
  -d '{"task_name":"maiia.feature-extraction.v1", "payload":"{\"text\":\"Hello world\"}", "request_id":"1"}' \
  127.0.0.1:50051 maiia.worker.v1.WorkerService/ExecuteTask
```

## Docker

```bash
# Build (ONNX only - default)
docker build -t maiia-inference .

# Build with Candle + llama backends
docker build --build-arg FEATURES="candle,llama" -t maiia-inference .

# Run a model directly
docker run --rm -p 50051:50051 maiia-inference --model Xenova/bge-m3

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
├── Dockerfile                # Multi-stage build
├── docker-compose.yml
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
    ├── inference-service/    # Main binary (CLI, HF API client)
    ├── inference-core/       # Config, error types, generation config
    ├── inference-onnx/       # ONNX Runtime backend
    ├── inference-candle/     # Candle backend (Rust-native LLMs)
    ├── inference-llama/      # llama.cpp backend (GGUF models)
    ├── inference-tasks/      # Task registry (routes to backends)
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
