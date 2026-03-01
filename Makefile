# =============================================================================
# Maiia AI Inference Service (Rust) - Makefile
# =============================================================================
#
# Generic inference service supporting ONNX and Candle backends.
# No code changes needed - configure via TOML files.
#
# Usage:
#   make help          - Show this help message
#   make run-echo      - Run echo task (testing, no model)
#   make run-<task>    - Run specific task (e.g., make run-embed)
#   make test          - Run tests
#   make docker-up     - Run with docker-compose
#
# =============================================================================

.PHONY: help build release run test lint format check clean \
        docker-build docker-run docker-up docker-down \
        run-echo run-embed run-asr run-ocr run-chat \
        run-text-classification run-token-classification \
        run-image-classification run-object-detection \
        run-e5-large run-nomic run-bge-m3 run-reranker run-whisper-large run-florence \
        run-voxtral \
        grpcurl-health grpcurl-echo grpcurl-list

# Detect platform for Candle backend
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    CANDLE_FEATURES := --features candle-metal
    LLAMA_FEATURES := --features "candle-metal,llama"
else ifeq ($(UNAME_S),Linux)
    CANDLE_FEATURES := --features candle-cuda
    LLAMA_FEATURES := --features "candle-cuda,llama"
else
    CANDLE_FEATURES := --features candle
    LLAMA_FEATURES := --features "candle,llama"
endif

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════════╗"
	@echo "║     Maiia AI Inference Service - ONNX / Candle / llama.cpp       ║"
	@echo "╚══════════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "  Development:"
	@echo "    make build            Build debug binary"
	@echo "    make release          Build release binary"
	@echo "    make run-echo         Run echo task (no model, for testing)"
	@echo ""
	@echo "  Run by Task Category:"
	@echo "    NLP:"
	@echo "      make run-embed                 Feature extraction / embeddings"
	@echo "      make run-text-classification   Text classification"
	@echo "      make run-token-classification  NER / token classification"
	@echo "      make run-qa                    Question answering"
	@echo "      make run-text-generation       LLM / text generation"
	@echo "      make run-text-generation-gpt   LLM / text generation (GPT OSS ONNX)"
	@echo ""
	@echo "    Audio:"
	@echo "      make run-asr                   Speech recognition (Whisper ONNX)"
	@echo "      make run-voxtral               Voxtral ASR (Candle, Metal/CUDA)"
	@echo "      make run-tts                   Text-to-speech"
	@echo "      make run-audio-classification  Audio classification"
	@echo ""
	@echo "    Vision:"
	@echo "      make run-image-classification  Image classification"
	@echo "      make run-object-detection      Object detection"
	@echo "      make run-ocr                   OCR / image-to-text"
	@echo "      make run-depth                 Depth estimation"
	@echo ""
	@echo "    Multimodal:"
	@echo "      make run-vqa                   Visual question answering"
	@echo "      make run-document-qa           Document question answering"
	@echo ""
	@echo "    Foundation Models (Production-Ready):"
	@echo "      make run-e5-large              Multilingual E5 Large embeddings"
	@echo "      make run-nomic                 Nomic Embed Text v1 (Matryoshka)"
	@echo "      make run-bge-m3                BGE-M3 multilingual embeddings"
	@echo "      make run-reranker              BGE Reranker Large (cross-encoder)"
	@echo "      make run-whisper-large         Whisper Large v3 Turbo ASR"
	@echo "      make run-florence              Florence-2 Large vision-language"
	@echo ""
	@echo "  Testing & Quality:"
	@echo "    make test             Run all tests (start server first for integration)"
	@echo "    make lint             Run clippy linter"
	@echo "    make format           Format code with rustfmt"
	@echo "    make check            Run all checks (format, lint, test)"
	@echo ""
	@echo "  Docker:"
	@echo "    make docker-build     Build Docker image"
	@echo "    make docker-up        Start with docker-compose (+ MinIO)"
	@echo "    make docker-down      Stop docker-compose"
	@echo ""
	@echo "  gRPC Testing:"
	@echo "    make grpcurl-list     List available gRPC services"
	@echo "    make grpcurl-health   Health check"
	@echo "    make grpcurl-echo     Test echo task"
	@echo ""
	@echo "  Custom config:"
	@echo "    MAIIA_AI_CONFIG_PATH=./configs/nlp/text-classification.toml make run"
	@echo ""

# =============================================================================
# Development
# =============================================================================

build:
	cargo build

release:
	cargo build --release

# Default run uses echo task (no model required)
run:
	MAIIA_AI_CONFIG_PATH=./configs/echo.toml \
	RUST_LOG=info cargo run -p inference-service --release

# =============================================================================
# Run by Task Type (NLP)
# =============================================================================

run-echo:
	MAIIA_AI_CONFIG_PATH=./configs/echo.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-embed:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/feature-extraction.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-text-classification:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/text-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-token-classification:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/token-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-qa:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/question-answering.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-fill-mask:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/fill-mask.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-text-generation:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/text-generation.toml \
	RUST_LOG=info cargo run -p inference-service --release $(CANDLE_FEATURES)

run-text-generation-gpt:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/gpt-oss.toml \
	RUST_LOG=info cargo run -p inference-service --release $(LLAMA_FEATURES)

run-summarization:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/summarization.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-translation:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/translation.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-zero-shot:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/zero-shot-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-sentence-similarity:
	MAIIA_AI_CONFIG_PATH=./configs/nlp/sentence-similarity.toml \
	RUST_LOG=info cargo run -p inference-service --release

# =============================================================================
# Run by Task Type (Audio)
# =============================================================================

run-asr:
	MAIIA_AI_CONFIG_PATH=./configs/audio/automatic-speech-recognition.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-tts:
	MAIIA_AI_CONFIG_PATH=./configs/audio/text-to-speech.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-audio-classification:
	MAIIA_AI_CONFIG_PATH=./configs/audio/audio-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-voxtral:
	MAIIA_AI_CONFIG_PATH=./configs/audio/voxtral.toml \
	RUST_LOG=info cargo run -p inference-service --release $(CANDLE_FEATURES)

# =============================================================================
# Run by Task Type (Vision)
# =============================================================================

run-image-classification:
	MAIIA_AI_CONFIG_PATH=./configs/vision/image-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-object-detection:
	MAIIA_AI_CONFIG_PATH=./configs/vision/object-detection.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-image-segmentation:
	MAIIA_AI_CONFIG_PATH=./configs/vision/image-segmentation.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-ocr:
	MAIIA_AI_CONFIG_PATH=./configs/vision/image-to-text.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-depth:
	MAIIA_AI_CONFIG_PATH=./configs/vision/depth-estimation.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-image-feature:
	MAIIA_AI_CONFIG_PATH=./configs/vision/image-feature-extraction.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-zero-shot-image:
	MAIIA_AI_CONFIG_PATH=./configs/vision/zero-shot-image-classification.toml \
	RUST_LOG=info cargo run -p inference-service --release

# =============================================================================
# Run by Task Type (Multimodal)
# =============================================================================

run-vqa:
	MAIIA_AI_CONFIG_PATH=./configs/multimodal/visual-question-answering.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-document-qa:
	MAIIA_AI_CONFIG_PATH=./configs/multimodal/document-question-answering.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-vlm:
	MAIIA_AI_CONFIG_PATH=./configs/multimodal/image-text-to-text.toml \
	RUST_LOG=info cargo run -p inference-service --release

# =============================================================================
# Run by Task Type (Foundation Models)
# =============================================================================

run-e5-large:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/multilingual-e5-large.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-nomic:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/nomic-embed-text-v1.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-bge-m3:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/bge-m3.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-reranker:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/bge-reranker-large.toml \
	RUST_LOG=info cargo run -p inference-service --release

run-whisper-large:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/whisper-large-v3-turbo.toml \
	RUST_LOG=info cargo run -p inference-service --release $(CANDLE_FEATURES)

run-florence:
	MAIIA_AI_CONFIG_PATH=./configs/foundation/florence-2-large.toml \
	RUST_LOG=info cargo run -p inference-service --release

# =============================================================================
# Testing & Quality
# =============================================================================

test:
	cargo test --workspace --release -- --nocapture

test-verbose:
	cargo test --workspace --release -- --nocapture

lint:
	cargo clippy --workspace -- -D warnings

lint-fix:
	cargo clippy --workspace --fix --allow-dirty

format:
	cargo fmt --all

format-check:
	cargo fmt --all -- --check

check: format-check lint test
	@echo "✅ All checks passed!"

# =============================================================================
# Docker
# =============================================================================

IMAGE_NAME ?= maiia-ai-inference
IMAGE_TAG ?= latest

docker-build:
	docker build -t $(IMAGE_NAME):$(IMAGE_TAG) .

docker-run:
	docker run --rm -it \
		-e MAIIA_AI_CONFIG_PATH=/app/configs/echo.toml \
		-e RUST_LOG=info \
		-v $(PWD)/configs:/app/configs:ro \
		-p 50051:50051 \
		-p 8080:8080 \
		$(IMAGE_NAME):$(IMAGE_TAG)

docker-up:
	docker compose up --build

docker-up-d:
	docker compose up --build -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f inference

# Run specific task in Docker
docker-run-embed:
	docker run --rm -it \
		-e MAIIA_AI_CONFIG_PATH=/app/configs/nlp/feature-extraction.toml \
		-e RUST_LOG=info \
		-v $(PWD)/configs:/app/configs:ro \
		-v $(PWD)/models:/app/models:ro \
		-p 50051:50051 \
		$(IMAGE_NAME):$(IMAGE_TAG)

# =============================================================================
# gRPC Testing (requires server running)
# =============================================================================

GRPC_HOST ?= 127.0.0.1:50051

grpcurl-list:
	@grpcurl -plaintext $(GRPC_HOST) list

grpcurl-health:
	@grpcurl -plaintext $(GRPC_HOST) grpc.health.v1.Health/Check

grpcurl-echo:
	@grpcurl -plaintext \
		-d '{"task_name":"maiia.echo.v1", "payload":"{\"message\":\"hello world\"}", "request_id":"test-1"}' \
		$(GRPC_HOST) maiia.worker.v1.WorkerService/ExecuteTask

grpcurl-embed:
	@grpcurl -plaintext \
		-d '{"task_name":"maiia.feature-extraction.v1", "payload":"{\"inputs\":{\"input_ids\":[[101,2054,2003,102]],\"attention_mask\":[[1,1,1,1]]}}", "request_id":"test-embed"}' \
		$(GRPC_HOST) maiia.worker.v1.WorkerService/ExecuteTask

grpcurl-gpt-gguf:
	@grpcurl -plaintext \
		-d '{"task_name":"maiia.gpt-oss-gguf.v1", "payload":"{\"text\":\"Hello, how are you?\"}", "request_id":"test-gpt-gguf"}' \
		$(GRPC_HOST) maiia.worker.v1.WorkerService/ExecuteTask

# =============================================================================
# Cleanup
# =============================================================================

clean:
	cargo clean

rebuild: clean build

# =============================================================================
# Documentation
# =============================================================================

docs:
	cargo doc --no-deps --open

# =============================================================================
# Development Utilities
# =============================================================================

watch:
	cargo watch -x build

watch-test:
	cargo watch -x test

update:
	cargo update

# Download a test model (all-MiniLM-L6-v2)
download-test-model:
	@mkdir -p models
	@echo "Downloading all-MiniLM-L6-v2 ONNX model..."
	@curl -L -o models/model.onnx \
		"https://huggingface.co/Xenova/all-MiniLM-L6-v2/resolve/main/onnx/model.onnx"
	@echo "✅ Model downloaded to models/model.onnx"
