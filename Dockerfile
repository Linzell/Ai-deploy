# ============================================================================
# Multi-stage Dockerfile for Inference Service
# ============================================================================
# Build: docker build -t inference-service .
# Run:   docker run --env-file .env inference-service

# -----------------------------------------------------------------------------
# Stage 1: Builder
# -----------------------------------------------------------------------------
FROM rust:latest AS builder

WORKDIR /app

# Install protobuf compiler for gRPC
RUN apt-get update && apt-get install -y protobuf-compiler && rm -rf /var/lib/apt/lists/*

# Copy workspace files
COPY Cargo.toml Cargo.lock ./
COPY crates ./crates

# Build release binary
RUN cargo build --release --package inference-service

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
