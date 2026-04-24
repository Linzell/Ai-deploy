# ============================================================================
# Multi-stage Dockerfile for Maiia AI Inference Service
# ============================================================================
# Default features include: candle, llama, http, grpc
#
# Build with defaults (all backends + both protocols):
#   docker build -t inference-service .
#
# Build with specific features:
#   docker build --build-arg FEATURES=candle -t inference-service .
#   docker build --build-arg FEATURES=llama -t inference-service .
#   docker build --build-arg FEATURES="candle,llama" -t inference-service .
#
# Run:
#   docker run --env-file .env inference-service
#
# Run with CLI args:
#   docker run -p 50051:50051 -p 8080:8080 inference-service --model Qwen/Qwen2.5-0.5B-Instruct
#   docker run inference-service --task text-generation
#   docker run inference-service --config /config/task.toml

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM rust:1.85.1-bookworm AS builder

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
FROM debian:bookworm-slim AS runtime

# Install runtime dependencies + grpc_health_probe for container health checks
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

# Install grpc_health_probe (static binary) with checksum verification
ADD https://github.com/grpc-ecosystem/grpc-health-probe/releases/download/v0.4.37/grpc_health_probe-linux-amd64 /bin/grpc_health_probe
RUN echo "8302d54fc41d4ffbfea6871b6a04584265d5eabbe738aee2966f9a574b6f17d1  /bin/grpc_health_probe" | sha256sum -c - \
    && chmod +x /bin/grpc_health_probe

# Create non-root user
RUN useradd -m -u 1000 inference
USER inference

WORKDIR /app

# Copy binary from builder
COPY --from=builder /app/target/release/inference-service /app/inference-service

# Default environment variables
ENV MAIIA_AI_GRPC_PORT=50051
ENV MAIIA_AI_HTTP_PORT=8080
ENV MAIIA_AI_CACHE_DIR=/tmp/inference-cache
# RUST_LOG is intentionally not set — the binary has sensible per-crate defaults.
# Override with: docker run -e RUST_LOG=info ...

# Expose gRPC and HTTP ports
EXPOSE 50051 8080

# Health check via gRPC health protocol (matches our HealthServiceImpl)
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD ["/bin/grpc_health_probe", "-addr=:50051"]

# Run the service (ENTRYPOINT so `docker run <image> --model foo` works)
ENTRYPOINT ["./inference-service"]
