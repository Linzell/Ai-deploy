# Maiia AI Inference Service

Generic inference service in Rust. Single binary that runs **any** HuggingFace model type without code changes - configure via TOML files.

## Features

- **Universal**: Supports all HuggingFace task types (NLP, Audio, Vision, Multimodal)
- **Multi-backend**: ONNX Runtime (default), Candle (Rust-native), llama.cpp (GGUF models)
- **Zero code changes**: Add new models by creating a TOML config file
- **Fast**: Rust + optimized backends for high-performance inference
- **Production-ready**: gRPC API, health checks, batching support
- **Flexible loading**: HuggingFace Hub, S3/MinIO, or local files

## Quick Start

```bash
# Run echo task (no model needed, for testing)
make run-echo

# Run with a specific task
make run-embed              # Text embeddings
make run-text-classification  # Sentiment analysis
make run-asr                # Speech recognition
make run-image-classification # Image classification

# Or with Docker
docker compose up
```

## Supported Task Types

### NLP
| Task | Config | Example Models |
|------|--------|----------------|
| Feature Extraction | `nlp/feature-extraction.toml` | BGE-M3, all-MiniLM-L6-v2 |
| Text Classification | `nlp/text-classification.toml` | DistilBERT-SST2, RoBERTa |
| Token Classification | `nlp/token-classification.toml` | BERT-NER |
| Question Answering | `nlp/question-answering.toml` | DistilBERT-SQuAD |
| Text Generation | `nlp/text-generation.toml` | GPT-2, Phi-3 |
| Summarization | `nlp/summarization.toml` | BART, T5 |
| Translation | `nlp/translation.toml` | OPUS-MT, NLLB |
| Fill-Mask | `nlp/fill-mask.toml` | BERT, RoBERTa |
| Zero-Shot Classification | `nlp/zero-shot-classification.toml` | DeBERTa-NLI |
| Sentence Similarity | `nlp/sentence-similarity.toml` | MiniLM |

### Audio
| Task | Config | Example Models |
|------|--------|----------------|
| Speech Recognition | `audio/automatic-speech-recognition.toml` | Whisper |
| Text-to-Speech | `audio/text-to-speech.toml` | SpeechT5, MMS-TTS |
| Audio Classification | `audio/audio-classification.toml` | Wav2Vec2, AST |

### Vision
| Task | Config | Example Models |
|------|--------|----------------|
| Image Classification | `vision/image-classification.toml` | ViT, ResNet |
| Object Detection | `vision/object-detection.toml` | DETR, YOLOS |
| Image Segmentation | `vision/image-segmentation.toml` | SegFormer |
| OCR / Image-to-Text | `vision/image-to-text.toml` | TrOCR |
| Depth Estimation | `vision/depth-estimation.toml` | DPT, Depth Anything |
| Image Features | `vision/image-feature-extraction.toml` | CLIP, DINOv2 |
| Zero-Shot Image | `vision/zero-shot-image-classification.toml` | CLIP, SigLIP |

### Multimodal
| Task | Config | Example Models |
|------|--------|----------------|
| Visual QA | `multimodal/visual-question-answering.toml` | ViLT, BLIP |
| Document QA | `multimodal/document-question-answering.toml` | Donut, LayoutLM |
| Vision-Language | `multimodal/image-text-to-text.toml` | Florence-2 |

### Foundation Models (Production-Ready)

Pre-configured for popular production models with best practices:

| Model | Config | Use Case | Make Target |
|-------|--------|----------|-------------|
| Multilingual E5 Large | `foundation/multilingual-e5-large.toml` | Cross-lingual semantic search (100+ languages) | `make run-e5-large` |
| Nomic Embed Text v1 | `foundation/nomic-embed-text-v1.toml` | Matryoshka embeddings (truncate to any dim) | `make run-nomic` |
| BGE-M3 | `foundation/bge-m3.toml` | Hybrid search (dense + sparse + ColBERT) | `make run-bge-m3` |
| BGE Reranker Large | `foundation/bge-reranker-large.toml` | Cross-encoder reranking for RAG | `make run-reranker` |
| Whisper Large v3 Turbo | `foundation/whisper-large-v3-turbo.toml` | Production ASR (99 languages) | `make run-whisper-large` |
| Florence-2 Large | `foundation/florence-2-large.toml` | Vision-language (OCR, captioning, detection) | `make run-florence` |

## Configuration

### Environment Variables

All settings use `MAIIA_AI_` prefix:

```bash
MAIIA_AI_CONFIG_PATH=./configs/nlp/feature-extraction.toml
MAIIA_AI_TASK_TYPE=feature-extraction
MAIIA_AI_MODEL_PATH=Xenova/all-MiniLM-L6-v2
MAIIA_AI_MODEL_SOURCE=huggingface  # or: s3, local
MAIIA_AI_DEVICE=cpu                # or: cuda, metal
MAIIA_AI_GRPC_PORT=50051
```

### TOML Config File

```toml
[task]
type = "feature-extraction"        # Any string - fully generic
name = "maiia.embed.v1"            # gRPC task name

[model]
source = "huggingface"             # huggingface, s3, local
path = "Xenova/all-MiniLM-L6-v2"   # HF repo, S3 prefix, or local path
onnx_file = "onnx/model.onnx"      # ONNX file within the model

[inference]
device = "cpu"                     # cpu, cuda, metal
num_threads = 4

[service]
grpc_port = 50051
enable_batching = true
max_batch_size = 64
```

## API

### gRPC

```protobuf
service WorkerService {
  rpc ExecuteTask(TaskRequest) returns (TaskResponse);
}

message TaskRequest {
  string task_name = 1;    // Must match config task.name
  string payload = 2;      // JSON with model inputs
  string request_id = 3;
}
```

### Input Format

All models expect JSON with an `inputs` object containing named tensors:

```json
{
  "inputs": {
    "input_ids": [[101, 2054, 2003, 102]],
    "attention_mask": [[1, 1, 1, 1]]
  }
}
```

### Output Format

```json
{
  "outputs": {
    "last_hidden_state": [[[0.1, 0.2, ...], ...]]
  }
}
```

## Testing

```bash
# Run tests
make test

# Test gRPC (requires server running)
make grpcurl-health
make grpcurl-echo

# Test with grpcurl manually
grpcurl -plaintext \
  -d '{"task_name":"maiia.echo.v1", "payload":"{\"msg\":\"hello\"}", "request_id":"1"}' \
  127.0.0.1:50051 maiia.worker.v1.WorkerService/ExecuteTask
```

## Docker

```bash
# Build (ONNX only - default)
docker build -t maiia-inference .

# Build with Candle backend (Rust-native LLMs)
docker build --build-arg FEATURES=candle -t maiia-inference .

# Build with llama.cpp backend (GGUF models)
docker build --build-arg FEATURES=llama -t maiia-inference .

# Build with all backends
docker build --build-arg FEATURES="candle,llama" -t maiia-inference .

# Run with specific task
TASK_CONFIG=nlp/feature-extraction.toml docker compose up

# Run with Candle backend enabled
FEATURES=candle docker compose up --build

# Run with GPU (NVIDIA)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

## Project Structure

```
AI-deploy/
├── Cargo.toml              # Workspace root
├── Makefile                # Build/run commands
├── Dockerfile
├── docker-compose.yml
├── configs/                # Task configurations
│   ├── echo.toml           # Test task (no model)
│   ├── nlp/                # NLP tasks
│   ├── audio/              # Audio tasks
│   ├── vision/             # Vision tasks
│   ├── multimodal/         # Multimodal tasks
│   └── foundation/         # Production-ready foundation models
├── models/                 # Local model storage
└── crates/
    ├── inference-core/       # Shared config, error types, generation config
    ├── inference-onnx/       # ONNX Runtime backend (embeddings, seq2seq, vision)
    ├── inference-candle/     # Candle backend (Rust-native LLMs)
    ├── inference-llama/      # llama.cpp backend (GGUF models)
    ├── inference-tasks/      # Facade crate (re-exports backends, task registry)
    ├── inference-grpc/       # gRPC server, proto definitions
    ├── inference-service/    # Main binary
    ├── inference-preprocess/ # Tokenizers, image/audio preprocessing
    ├── inference-postprocess/# Output decoding, post-processing
    ├── inference-loader-hf/  # HuggingFace Hub loader
    └── inference-loader-s3/  # S3/MinIO loader
```

## Backends

| Backend | Feature Flag | Use Case | Models |
|---------|--------------|----------|--------|
| ONNX Runtime | (default) | Embeddings, classification, seq2seq, vision | All ONNX models |
| Candle | `candle` | Rust-native LLMs, no Python deps | Qwen3, Llama 3.x, Mistral |
| llama.cpp | `llama` | Quantized GGUF models | Any GGUF model |

```bash
# Build with all backends
cargo build --release --features "candle,llama"

# Build with specific backend
cargo build --release --features "candle"
```

## Adding a New Model

1. Create a TOML config in `configs/`:

```toml
[task]
type = "my-custom-task"
name = "mycompany.custom.v1"

[model]
source = "local"
path = "/models/my-model"
onnx_file = "model.onnx"
```

2. Run:

```bash
MAIIA_AI_CONFIG_PATH=./configs/my-custom-task.toml cargo run --release
```

No code changes needed!

## License

MIT
