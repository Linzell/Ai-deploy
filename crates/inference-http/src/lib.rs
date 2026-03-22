//! HTTP server with OpenAI-compatible endpoints.
//!
//! This crate provides HTTP endpoints that follow the OpenAI API standard:
//! - POST /v1/chat/completions
//! - POST /v1/completions
//! - POST /v1/embeddings
//! - GET /v1/models
//! - GET /v1/models/{name}

#![allow(dead_code)]

use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::net::TcpListener;

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

use handlers::{
    chat_completions, completions, embeddings, get_model, list_models, model_status, reload_model,
    shutdown, unload_model, AppState,
};

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
        self.build_router_with_state(AppState {
            task: self.task.clone(),
            idle_timeout_seconds: Arc::new(Mutex::new(self.config.idle_timeout_seconds)),
            last_access_time: Arc::new(Mutex::new(Some(Instant::now()))),
            should_unload: Arc::new(Mutex::new(false)),
            shutdown_requested: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        })
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
            ConcurrencyLimitLayer::new(self.config.max_concurrent_requests)
        } else {
            ConcurrencyLimitLayer::new(usize::MAX)
        };

        Router::new()
            .route("/v1/models", get(list_models))
            .route("/v1/models/{model}", get(get_model))
            .route("/v1/chat/completions", post(chat_completions))
            .route("/v1/completions", post(completions))
            .route("/v1/embeddings", post(embeddings))
            .route("/reload/{model}", post(reload_model))
            .route("/unload/{model}", post(unload_model))
            .route("/status", get(model_status))
            .route("/shutdown", post(shutdown))
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

        let mut state = AppState {
            task: self.task.clone(),
            idle_timeout_seconds: Arc::new(Mutex::new(self.config.idle_timeout_seconds)),
            last_access_time: Arc::new(Mutex::new(Some(Instant::now()))),
            should_unload: Arc::new(Mutex::new(false)),
            shutdown_requested: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        };
        state.start_idle_timeout_loop();

        let listener = TcpListener::bind(addr.clone()).await?;
        info!("HTTP server listening on {}", addr);

        let app = self.build_router_with_state(state.clone());
        let shutdown_flag = state.shutdown_requested.clone();

        tokio::select! {
            result = axum::serve(listener, app) => {
                result?;
            }
            () = async {
                while !shutdown_flag.load(std::sync::atomic::Ordering::SeqCst) {
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                }
                info!("Shutdown signal received");
            } => {}
        };

        Ok(())
    }
}
