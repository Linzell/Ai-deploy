//! HTTP server with OpenAI-compatible endpoints.
//!
//! This crate provides HTTP endpoints that follow the OpenAI API standard:
//! - POST /v1/chat/completions
//! - POST /v1/completions
//! - POST /v1/embeddings
//! - GET /v1/models
//! - GET /v1/models/{name}

#![allow(dead_code)]

use std::sync::Arc;

mod handlers;
mod models;
mod preprocess;

use axum::{
    extract::Request,
    routing::{get, post},
    Router,
};
use tower::limit::ConcurrencyLimitLayer;
use tower_http::{
    cors::{Any, CorsLayer},
    trace::TraceLayer,
};
use tracing::info;

use inference_core::{Config, Result, Task};

use handlers::*;

/// HTTP server wrapper around Task.
#[derive(Clone)]
pub struct HttpServer {
    task: Arc<dyn Task>,
    config: Config,
}

impl HttpServer {
    /// Create a new HTTP server.
    pub fn new(task: Arc<dyn Task>, config: Config) -> Self {
        Self { task, config }
    }

    /// Build the router with all endpoints and state.
    pub fn build_router(&self) -> Router {
        let state = AppState {
            task: Arc::clone(&self.task),
        };
        self.build_router_with_state(state)
    }

    /// Build the router with state.
    fn build_router_with_state(&self, state: AppState) -> Router {
        // CORS middleware
        let cors = CorsLayer::new()
            .allow_methods(Any)
            .allow_headers(Any)
            .allow_origin(Any);

        // Request tracing
        let trace = TraceLayer::new_for_http().make_span_with(|request: &Request<_>| {
            let method = request.method().clone();
            let uri = request.uri().clone();
            tracing::span!(
                tracing::Level::INFO,
                "http_request",
                method = %method,
                uri = %uri,
            )
        });

        // Concurrency limit
        let concurrency_layer = if self.config.max_concurrent_requests > 0 {
            info!(
                "HTTP concurrency limit: {} max in-flight requests",
                self.config.max_concurrent_requests
            );
            ConcurrencyLimitLayer::new(self.config.max_concurrent_requests as usize)
        } else {
            ConcurrencyLimitLayer::new(usize::MAX)
        };

        Router::new()
            .route("/v1/models", get(list_models))
            .route("/v1/models/{model}", get(get_model))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            .with_state(state)
            .layer(cors)
            .layer(trace)
            .layer(concurrency_layer)
    }

    /// Start the HTTP server.
    pub async fn serve(&self) -> Result<()> {
        let addr = format!("0.0.0.0:{}", self.config.http_port);

        info!(
            "Starting HTTP server on {} for task '{}'",
            addr,
            self.task.name()
        );

        let listener = tokio::net::TcpListener::bind(&addr).await?;

        let state = AppState {
            task: Arc::clone(&self.task),
        };

        let router = self.build_router_with_state(state);

        axum::serve(listener, router).await?;

        Ok(())
    }
}
