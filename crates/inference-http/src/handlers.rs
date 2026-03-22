//! OpenAI-compatible request handlers.

use axum::{
    extract::{Json, State},
    response::Json as ResponseJson,
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};
use std::time::Instant;

use tracing::{info, warn};

use crate::preprocess::{
    postprocess_chat_completion, postprocess_completion, postprocess_embedding,
    preprocess_chat_completion, preprocess_completion, preprocess_embedding,
};
use inference_core::task::TaskResult;
use inference_core::Task;

/// Shared state for handlers.
#[derive(Clone)]
pub struct AppState {
    pub task: Arc<dyn Task>,
    pub should_unload: Arc<Mutex<bool>>,
    pub last_access_time: Arc<Mutex<Option<Instant>>>,
    pub idle_timeout_seconds: Arc<Mutex<u64>>,
    pub shutdown_requested: Arc<AtomicBool>,
}

impl AppState {
    /// Check if model should be unloaded based on idle timeout
    pub fn should_unload(&self) -> bool {
        let now = Instant::now();
        let last_access = {
            let guard = self.last_access_time.lock().unwrap();
            *guard.as_ref().map_or(&now, |t| t)
        };

        let idle_duration = now.duration_since(last_access);
        let idle_timeout = {
            let guard = self.idle_timeout_seconds.lock().unwrap();
            *guard
        };
        idle_duration.as_secs() >= idle_timeout
    }

    /// Update the last access time
    pub fn update_last_access(&self) {
        *self.last_access_time.lock().unwrap() = Some(Instant::now());
    }

    /// Set whether the model should be unloaded
    pub fn set_should_unload(&self, value: bool) {
        *self.should_unload.lock().unwrap() = value;
    }

    /// Start the idle timeout tracking loop
    pub fn start_idle_timeout_loop(&mut self) {
        let idle_timeout = {
            let guard = self.idle_timeout_seconds.lock().unwrap();
            *guard
        };

        if idle_timeout == 0 {
            info!("Idle timeout disabled (0 seconds)");
            return;
        }

        let idle_timeout_seconds = Arc::clone(&self.idle_timeout_seconds);
        let last_access_time = Arc::clone(&self.last_access_time);
        let should_unload = Arc::clone(&self.should_unload);
        let task = Arc::clone(&self.task);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(1));

            loop {
                interval.tick().await;

                let now = Instant::now();

                let idle_duration = {
                    let last_access = *last_access_time.lock().unwrap();
                    if let Some(la) = last_access {
                        now.duration_since(la)
                    } else {
                        now.duration_since(now)
                    }
                };
                let timeout = {
                    let guard = idle_timeout_seconds.lock().unwrap();
                    *guard
                };

                if idle_duration.as_secs() >= timeout {
                    info!("Idle timeout reached for task '{}'", task.name());
                    *should_unload.lock().unwrap() = true;

                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    *should_unload.lock().unwrap() = false;
                }
            }
        });
    }
}

/// Get available models.
#[axum::debug_handler]
pub async fn list_models(state: State<AppState>) -> ResponseJson<serde_json::Value> {
    let models = state.task.get_available_models();
    ResponseJson(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

/// Get a specific model.
#[axum::debug_handler]
pub async fn get_model(
    _state: State<AppState>,
    model_name: String,
) -> ResponseJson<serde_json::Value> {
    ResponseJson(serde_json::json!({
        "id": model_name,
        "object": "model",
        "created": 0,
        "owned_by": "default",
        "root": true,
        "parent": null
    }))
}

/// Handle chat completion request.
#[axum::debug_handler]
pub async fn chat_completions(
    mut state: State<AppState>,
    body: Json<serde_json::Value>,
) -> ResponseJson<serde_json::Value> {
    let body_value = body.0;
    let request: crate::models::ChatCompletionRequest = match serde_json::from_value(
        body_value.clone(),
    ) {
        Ok(r) => r,
        Err(_) => {
            return ResponseJson(
                serde_json::json!({ "error": { "message": "Invalid request", "type": "invalid_request_error" } }),
            )
        }
    };

    let request_id = uuid::Uuid::new_v4().to_string();
    // Preprocess request into task format
    let (prompt, temperature, max_tokens) = preprocess_chat_completion(&body_value);
    let payload = serde_json::json!({
        "text": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    });
    let payload_str = payload.to_string();

    // Check if should unload due to idle timeout
    if state.should_unload() {
        info!(
            "[{}] Unloading model due to idle timeout for task '{}'",
            request_id,
            state.task.name()
        );
        let task = Arc::get_mut(&mut state.task).unwrap();
        let unload_result = task.unload().await;
        if !unload_result.success {
            warn!(
                "[{}] Failed to unload model: {}",
                request_id,
                unload_result.error.as_deref().unwrap_or("unknown")
            );
        }
    }

    // Lazy load: reload model if not loaded
    if !state.task.is_ready() {
        info!("Model not loaded, reloading on first request");
        let reload_result = state.task.reload().await;
        if !reload_result.success {
            return ResponseJson(serde_json::json!({
                "error": { "message": format!("Failed to load model: {}", reload_result.error.unwrap_or_else(|| "Unknown error".to_string())), "type": "model_load_error" }
            }));
        }
    }

    state.update_last_access();
    let result = state.task.execute(&payload_str, &request_id).await;

    match result {
        TaskResult {
            success: true,
            result: Some(completion),
            error: None,
        } => {
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|e| e.as_secs() * 1000 + u64::from(e.subsec_millis()))
                .unwrap_or(0);

            let usage = serde_json::json!({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            });

            // Postprocess result into OpenAI format
            let actual_content = postprocess_chat_completion(&completion);

            let message = serde_json::json!({
                "role": "assistant",
                "content": actual_content
            });
            ResponseJson(serde_json::json!({
                "id": request_id,
                "object": "chat.completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "index": 0,
                    "message": message,
                    "finish_reason": "stop"
                }],
                "usage": usage
            }))
        }
        TaskResult {
            success: false,
            error: Some(err),
            ..
        } => ResponseJson(serde_json::json!({ "error": { "message": err, "type": "error" } })),
        _ => ResponseJson(
            serde_json::json!({ "error": { "message": "Unknown error", "type": "error" } }),
        ),
    }
}

/// Handle completion request.
#[axum::debug_handler]
pub async fn completions(
    mut state: State<AppState>,
    body: Json<serde_json::Value>,
) -> ResponseJson<serde_json::Value> {
    let body_value = body.0;
    let request: crate::models::CompletionRequest = match serde_json::from_value(body_value.clone())
    {
        Ok(r) => r,
        Err(_) => {
            return ResponseJson(
                serde_json::json!({ "error": { "message": "Invalid request", "type": "invalid_request_error" } }),
            )
        }
    };

    let request_id = uuid::Uuid::new_v4().to_string();
    // Preprocess request into task format
    let (prompt, temperature, max_tokens) = preprocess_completion(&body_value);
    let payload = serde_json::json!({
        "text": prompt,
        "temperature": temperature,
        "max_tokens": max_tokens
    });
    let payload_str = payload.to_string();

    // Check if should unload due to idle timeout
    if state.should_unload() {
        info!(
            "[{}] Unloading model due to idle timeout for task '{}'",
            request_id,
            state.task.name()
        );
        let task = Arc::get_mut(&mut state.task).unwrap();
        let unload_result = task.unload().await;
        if !unload_result.success {
            warn!(
                "[{}] Failed to unload model: {}",
                request_id,
                unload_result.error.as_deref().unwrap_or("unknown")
            );
        }
    }

    // Lazy load: reload model if not loaded
    if !state.task.is_ready() {
        info!("Model not loaded, reloading on first request");
        let reload_result = state.task.reload().await;
        if !reload_result.success {
            return ResponseJson(serde_json::json!({
                "error": { "message": format!("Failed to load model: {}", reload_result.error.unwrap_or_else(|| "Unknown error".to_string())), "type": "model_load_error" }
            }));
        }
    }

    state.update_last_access();
    let result = state.task.execute(&payload_str, &request_id).await;

    match result {
        TaskResult {
            success: _,
            result: Some(text),
            error: None,
        } => {
            let created = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|e| e.as_secs() * 1000 + u64::from(e.subsec_millis()))
                .unwrap_or(0);

            let usage = serde_json::json!({
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            });

            // Postprocess result into OpenAI format
            let actual_text = postprocess_completion(&text);

            ResponseJson(serde_json::json!({
                "id": request_id,
                "object": "text_completion",
                "created": created,
                "model": request.model,
                "choices": [{
                    "text": actual_text,
                    "index": 0,
                    "logprobs": null,
                    "finish_reason": "stop"
                }],
                "usage": usage
            }))
        }
        TaskResult {
            success: false,
            error: Some(err),
            ..
        } => ResponseJson(serde_json::json!({ "error": { "message": err, "type": "error" } })),
        _ => ResponseJson(
            serde_json::json!({ "error": { "message": "Unknown error", "type": "error" } }),
        ),
    }
}

/// Handle embeddings request.
#[axum::debug_handler]
pub async fn embeddings(
    mut state: State<AppState>,
    body: Json<serde_json::Value>,
) -> ResponseJson<serde_json::Value> {
    let body_value = body.0;
    let request: crate::models::EmbeddingRequest = match serde_json::from_value(body_value.clone())
    {
        Ok(r) => r,
        Err(_) => {
            return ResponseJson(
                serde_json::json!({ "error": { "message": "Invalid request", "type": "invalid_request_error" } }),
            )
        }
    };

    let request_id = uuid::Uuid::new_v4().to_string();
    // Preprocess request into task format
    let payload_str = preprocess_embedding(&body_value);

    // Check if should unload due to idle timeout
    if state.should_unload() {
        info!(
            "[{}] Unloading model due to idle timeout for task '{}'",
            request_id,
            state.task.name()
        );
        let task = Arc::get_mut(&mut state.task).unwrap();
        let unload_result = task.unload().await;
        if !unload_result.success {
            warn!(
                "[{}] Failed to unload model: {}",
                request_id,
                unload_result.error.as_deref().unwrap_or("unknown")
            );
        }
    }

    // Lazy load: reload model if not loaded
    if !state.task.is_ready() {
        info!("Model not loaded, reloading on first request");
        let reload_result = state.task.reload().await;
        if !reload_result.success {
            return ResponseJson(serde_json::json!({
                "error": { "message": format!("Failed to load model: {}", reload_result.error.unwrap_or_else(|| "Unknown error".to_string())), "type": "model_load_error" }
            }));
        }
    }

    state.update_last_access();
    let result = state.task.execute(&payload_str, &request_id).await;

    match result {
        TaskResult {
            success: false,
            error: Some(err),
            ..
        } => ResponseJson(serde_json::json!({ "error": { "message": err, "type": "error" } })),
        TaskResult {
            success: _,
            result: Some(result_text),
            error: None,
        } => {
            // Postprocess result into OpenAI format
            let embeddings = postprocess_embedding(&result_text);

            let data = embeddings
                .into_iter()
                .enumerate()
                .map(|(i, emb)| {
                    serde_json::json!({
                        "object": "embedding",
                        "embedding": emb,
                        "index": i
                    })
                })
                .collect::<Vec<_>>();

            ResponseJson(serde_json::json!({
                "object": "list",
                "data": data,
                "model": request.model,
                "usage": {
                    "prompt_tokens": 0,
                    "total_tokens": 0
                }
            }))
        }
        TaskResult { .. } => ResponseJson(
            serde_json::json!({ "error": { "message": "Unknown error", "type": "error" } }),
        ),
    }
}

/// Reload model endpoint.
#[axum::debug_handler]
pub async fn reload_model(state: State<AppState>) -> ResponseJson<serde_json::Value> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let result = state.task.reload().await;

    match result {
        TaskResult {
            success: true,
            result: Some(msg),
            error: None,
        } => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "reload",
            "status": "success",
            "message": msg
        })),
        TaskResult {
            success: false,
            error: Some(err),
            ..
        } => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "reload",
            "status": "error",
            "error": { "message": err, "type": "error" }
        })),
        _ => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "reload",
            "status": "error",
            "error": { "message": "Unknown error", "type": "error" }
        })),
    }
}

/// Unload model endpoint.
#[axum::debug_handler]
pub async fn unload_model(mut state: State<AppState>) -> ResponseJson<serde_json::Value> {
    let request_id = uuid::Uuid::new_v4().to_string();
    let task = Arc::get_mut(&mut state.task).unwrap();
    let result = task.unload().await;

    match result {
        TaskResult {
            success: true,
            result: Some(msg),
            error: None,
        } => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "unload",
            "status": "success",
            "message": msg
        })),
        TaskResult {
            success: false,
            error: Some(err),
            ..
        } => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "unload",
            "status": "error",
            "error": { "message": err, "type": "error" }
        })),
        _ => ResponseJson(serde_json::json!({
            "id": request_id,
            "object": "unload",
            "status": "error",
            "error": { "message": "Unknown error", "type": "error" }
        })),
    }
}

/// Check model status (loaded/unloaded).
#[axum::debug_handler]
pub async fn model_status(state: State<AppState>) -> ResponseJson<serde_json::Value> {
    let is_ready = state.task.is_ready();
    let should_unload = state.should_unload();

    ResponseJson(serde_json::json!({
        "id": state.task.name(),
        "object": "model",
        "status": if is_ready && !should_unload { "loaded" } else { "unloaded" },
        "should_unload": should_unload,
        "is_ready": is_ready
    }))
}

/// Request shutdown of the HTTP server
pub async fn shutdown(State(state): State<AppState>) -> ResponseJson<serde_json::Value> {
    info!("Shutdown requested via HTTP endpoint");
    state.shutdown_requested.store(true, Ordering::SeqCst);
    ResponseJson(serde_json::json!({
        "status": "ok",
        "message": "Shutdown requested"
    }))
}
