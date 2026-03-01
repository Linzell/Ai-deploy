# ============================================================================
# Multi-stage Dockerfile for Inference Service
# ============================================================================
# Backends: ONNX (default), Candle (Rust-native LLMs), llama.cpp (GGUF models)
#
# Build with default (ONNX only):
#   docker build -t inference-service .
#
# Build with Candle backend:
#   docker build --build-arg FEATURES=candle -t inference-service .
#
# Build with llama.cpp backend:
#   docker build --build-arg FEATURES=llama -t inference-service .
#
# Build with all backends:
#   docker build --build-arg FEATURES="candle,llama" -t inference-service .
#
# Run:
#   docker run --env-file .env inference-service

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM rust:latest AS builder

# Build argument for optional features (candle, llama, or both)
ARG FEATURES=""

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y \
    protobuf-compiler \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build release binary with optional features
RUN if [ -z "$FEATURES" ]; then \
        cargo build --release --package inference-service; \
    else \
        cargo build --release --package inference-service --features "$FEATURES"; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: Runtime
# -----------------------------------------------------------------------------
FROM debian:trixie-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 inference
USER inference

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/inference-service /app/inference-service

# Default environment variables
ENV MAIIA_AI_GRPC_PORT=50051
ENV MAIIA_AI_HEALTH_PORT=8080
ENV MAIIA_AI_CACHE_DIR=/tmp/inference-cache
ENV RUST_LOG=info

# Expose ports
EXPOSE 50051 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run the service
CMD ["./inference-service"]
