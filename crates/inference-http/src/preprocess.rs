//! Preprocessing utilities for HTTP handlers.
//!
//! Converts OpenAI-compatible request formats into the task-specific
//! input format expected by underlying inference tasks.

use serde::{Deserialize, Serialize};
use std::fmt::Write;

/// Hard upper bound for `max_tokens` to prevent unbounded generation / DoS.
const MAX_TOKENS_HARD_LIMIT: u32 = 32_768;

/// Clamp `max_tokens` to a safe range [1, MAX_TOKENS_HARD_LIMIT].
fn clamp_max_tokens(v: u32) -> u32 {
    v.clamp(1, MAX_TOKENS_HARD_LIMIT)
}

/// OpenAI chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// OpenAI chat completion request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logit_bias: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

fn default_max_tokens() -> u32 {
    256
}

/// OpenAI completion request (legacy).
#[derive(Debug, Clone, Deserialize)]
pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub n: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub echo: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stop: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub presence_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frequency_penalty: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// OpenAI embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingRequest {
    pub model: String,
    pub input: serde_json::Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub user: Option<String>,
}

/// Preprocess a chat completion request into task format.
///
/// The task expects a JSON payload with specific fields depending on the task type.
/// For text generation tasks, it expects: `{"text": "..."}`
pub fn preprocess_chat_completion(body: &serde_json::Value) -> (String, Option<f32>, Option<u32>) {
    // Extract messages array
    let Some(messages) = body.get("messages").and_then(|v| v.as_array()) else {
        return (String::new(), None, None);
    };

    // Extract the full conversation context as a single prompt
    let mut prompt = String::new();
    for msg in messages {
        if let Some(role) = msg.get("role").and_then(|v| v.as_str()) {
            let role_marker = match role {
                "system" => "[SYSTEM] ",
                "user" => "[USER] ",
                "assistant" => "[ASSISTANT] ",
                _ => "",
            };
            if let Some(content) = msg.get("content").and_then(|v| v.as_str()) {
                let _ = write!(prompt, "{role_marker}{content}");
                prompt.push('\n');
            }
        }
    }

    // Extract temperature and max_tokens
    let temperature = body
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .and_then(|f| (f as f32).into());
    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .map(|n| clamp_max_tokens(n as u32));

    (prompt, temperature, max_tokens)
}

/// Preprocess a completion request into task format.
pub fn preprocess_completion(body: &serde_json::Value) -> (String, Option<f32>, Option<u32>) {
    let prompt = body
        .get("prompt")
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();
    let temperature = body
        .get("temperature")
        .and_then(serde_json::Value::as_f64)
        .and_then(|f| (f as f32).into());
    let max_tokens = body
        .get("max_tokens")
        .and_then(serde_json::Value::as_u64)
        .map(|n| clamp_max_tokens(n as u32));

    (prompt, temperature, max_tokens)
}

/// Preprocess an embedding request into task format.
///
/// The task expects: `{"text": "..."}` or `{"input": "..."}`
pub fn preprocess_embedding(body: &serde_json::Value) -> String {
    let input = match body.get("input") {
        Some(v) => v.clone(),
        None => return serde_json::json!({ "text": "", "truncate": true }).to_string(),
    };

    match &input {
        serde_json::Value::String(s) => {
            serde_json::json!({ "text": s, "truncate": true }).to_string()
        }
        serde_json::Value::Array(arr) => {
            let texts: Vec<String> = arr
                .iter()
                .filter_map(|v| v.as_str().map(std::string::ToString::to_string))
                .collect();
            if texts.is_empty() {
                serde_json::json!({ "text": "", "truncate": true }).to_string()
            } else if texts.len() == 1 {
                serde_json::json!({ "text": texts[0], "truncate": true }).to_string()
            } else {
                serde_json::json!({ "text": texts.join("\n"), "truncate": true }).to_string()
            }
        }
        _ => serde_json::json!({ "text": "", "truncate": true }).to_string(),
    }
}

/// Postprocess a task result into OpenAI chat completion format.
///
/// Extracts text from JSON responses (e.g., `{"text": "..."}`) and formats
/// as OpenAI chat completion response.
pub fn postprocess_chat_completion(result: &str) -> String {
    // Try to extract text from JSON response
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(result) {
        if let Some(text_val) = json.get("text").and_then(|v| v.as_str()) {
            return text_val.to_string();
        }
    }
    result.to_string()
}

/// Postprocess a task result into OpenAI completion format.
pub fn postprocess_completion(result: &str) -> String {
    // Try to extract text from JSON response
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(result) {
        if let Some(text_val) = json.get("text").and_then(|v| v.as_str()) {
            return text_val.to_string();
        }
    }
    result.to_string()
}

/// Postprocess a task result into OpenAI embedding format.
///
/// Expects the task to return embeddings in one of these formats:
/// - JSON array of arrays: `[[0.1, 0.2, ...], [0.3, 0.4, ...]]`
/// - JSON object with embeddings: `[{"embedding": [...], "index": 0}, ...]`
/// - Plain text (fallback to dummy embedding)
pub fn postprocess_embedding(result: &str) -> Vec<Vec<f64>> {
    // Try to parse as array of arrays
    if let Ok(embeddings) = serde_json::from_str::<Vec<Vec<f64>>>(result) {
        return embeddings;
    }

    // Try to parse as array of objects with embeddings
    if let Ok(arr) = serde_json::from_str::<Vec<serde_json::Value>>(result) {
        let mut embeddings = Vec::new();
        for item in arr {
            if let Some(emb) = item.get("embedding").and_then(|v| v.as_array()) {
                if let Ok(embedding) =
                    serde_json::from_value::<Vec<f64>>(serde_json::Value::Array(emb.clone()))
                {
                    embeddings.push(embedding);
                }
            }
        }
        if !embeddings.is_empty() {
            return embeddings;
        }
    }

    // Fallback: return dummy embedding
    vec![vec![0.0; 768]]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_chat_completion() {
        let request = ChatCompletionRequest {
            model: "test".to_string(),
            messages: vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                    name: None,
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hello".to_string(),
                    name: None,
                },
            ],
            #[allow(unused)]
            temperature: Some(0.7),
            #[allow(unused)]
            max_tokens: 100,
            #[allow(unused)]
            top_p: None,
            #[allow(unused)]
            stream: None,
            #[allow(unused)]
            n: None,
            #[allow(unused)]
            stop: None,
            #[allow(unused)]
            presence_penalty: None,
            #[allow(unused)]
            frequency_penalty: None,
            #[allow(unused)]
            logit_bias: None,
            #[allow(unused)]
            user: None,
        };

        let value = serde_json::to_value(&request).unwrap();
        let (prompt, temp, tokens) = preprocess_chat_completion(&value);
        assert!(prompt.contains("You are helpful."));
        assert!(prompt.contains("Hello"));
        assert_eq!(temp, Some(0.7));
        assert_eq!(tokens, Some(100));
    }

    #[test]
    fn test_preprocess_embedding_string() {
        let request = EmbeddingRequest {
            model: "test".to_string(),
            input: serde_json::json!("Hello world"),
            user: None,
        };

        let value = serde_json::to_value(&request).unwrap();
        let payload = preprocess_embedding(&value);
        assert!(payload.contains("Hello"));
    }

    #[test]
    fn test_preprocess_embedding_array() {
        let request = EmbeddingRequest {
            model: "test".to_string(),
            input: serde_json::json!(["Hello", "World"]),
            user: None,
        };

        let value = serde_json::to_value(&request).unwrap();
        let payload = preprocess_embedding(&value);
        assert!(payload.contains("Hello"));
        assert!(payload.contains("World"));
    }

    #[test]
    fn test_postprocess_chat_completion_json() {
        let result = r#"{"text": "Hello, how can I help you?"}"#;
        let processed = postprocess_chat_completion(result);
        assert_eq!(processed, "Hello, how can I help you?");
    }

    #[test]
    fn test_postprocess_chat_completion_plain() {
        let result = "Hello, how can I help you?";
        let processed = postprocess_chat_completion(result);
        assert_eq!(processed, "Hello, how can I help you?");
    }
}
