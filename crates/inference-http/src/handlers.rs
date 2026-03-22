//! OpenAI-compatible request handlers.

use axum::{
    extract::{Json, State},
    response::Json as ResponseJson,
};

use crate::preprocess::{
    postprocess_chat_completion, postprocess_completion, postprocess_embedding,
    preprocess_chat_completion, preprocess_completion, preprocess_embedding,
};
use inference_core::task::{Task, TaskResult};

/// Shared state for handlers.
#[derive(Clone)]
pub struct AppState {
    pub task: std::sync::Arc<dyn Task>,
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
    state: State<AppState>,
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
    state: State<AppState>,
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
    state: State<AppState>,
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
