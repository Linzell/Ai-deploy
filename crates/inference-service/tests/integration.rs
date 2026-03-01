//! Integration tests for the inference service.
//!
//! Tests auto-discover the server's task name via gRPC and run appropriate tests.
//!
//! ## Running Tests
//!
//! 1. Start server with any config:
//!    ```bash
//!    make run-echo      # or make run-embed, make run-qa, etc.
//!    ```
//!
//! 2. Run tests:
//!    ```bash
//!    make test
//!    ```

mod fixtures;

use std::collections::HashMap;
use std::sync::OnceLock;
use tonic::transport::Channel;

use inference_grpc::generated::health_pb::{health_client::HealthClient, HealthCheckRequest};
use inference_grpc::generated::worker_pb::{
    worker_service_client::WorkerServiceClient, TaskRequest,
};

const SERVER_ADDR: &str = "http://127.0.0.1:50051";

/// Cached task name discovered from server
static DISCOVERED_TASK: OnceLock<Option<String>> = OnceLock::new();

// ============================================================================
// Helper Functions
// ============================================================================

mod helpers {
    use super::*;

    pub async fn connect_with_retry(
        addr: &str,
        max_retries: u32,
    ) -> Result<Channel, tonic::transport::Error> {
        let mut last_err = None;
        for i in 0..max_retries {
            match Channel::from_shared(addr.to_string())
                .unwrap()
                .connect()
                .await
            {
                Ok(channel) => return Ok(channel),
                Err(e) => {
                    last_err = Some(e);
                    if i < max_retries - 1 {
                        tokio::time::sleep(std::time::Duration::from_millis(
                            100 * (u64::from(i) + 1),
                        ))
                        .await;
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }

    pub async fn worker_client(
        addr: &str,
    ) -> Result<WorkerServiceClient<Channel>, tonic::transport::Error> {
        let channel = connect_with_retry(addr, 3).await?;
        Ok(WorkerServiceClient::new(channel))
    }

    pub async fn health_client(
        addr: &str,
    ) -> Result<HealthClient<Channel>, tonic::transport::Error> {
        let channel = connect_with_retry(addr, 3).await?;
        Ok(HealthClient::new(channel))
    }

    pub async fn check_health(addr: &str) -> bool {
        if let Ok(mut client) = health_client(addr).await {
            let request = HealthCheckRequest {
                service: String::new(),
            };
            client.check(request).await.is_ok()
        } else {
            false
        }
    }

    /// Discover task name by probing server
    pub async fn discover_task_name(addr: &str) -> Option<String> {
        let mut client = worker_client(addr).await.ok()?;

        let request = TaskRequest {
            task_name: "__probe__".to_string(),
            payload: "{}".to_string(),
            request_id: "probe".to_string(),
            metadata: HashMap::new(),
        };

        match client.execute_task(request).await {
            Err(status) => {
                let msg = status.message();
                if let Some(start) = msg.find("handles: '") {
                    let rest = &msg[start + 10..];
                    if let Some(end) = rest.find('\'') {
                        return Some(rest[..end].to_string());
                    }
                }
                None
            }
            Ok(_) => Some("__probe__".to_string()),
        }
    }

    pub async fn get_task_name(addr: &str) -> Option<String> {
        if let Some(cached) = DISCOVERED_TASK.get() {
            return cached.clone();
        }
        let task = discover_task_name(addr).await;
        let _ = DISCOVERED_TASK.set(task.clone());
        task
    }

    pub async fn execute_task(
        addr: &str,
        task_name: &str,
        payload: &str,
    ) -> Result<String, String> {
        let mut client = worker_client(addr)
            .await
            .map_err(|e| format!("Connection failed: {e}"))?;

        let request = TaskRequest {
            task_name: task_name.to_string(),
            payload: payload.to_string(),
            request_id: uuid::Uuid::new_v4().to_string(),
            metadata: HashMap::new(),
        };

        let response = match client.execute_task(request).await {
            Ok(r) => r,
            Err(status) => {
                return Err(format!("gRPC {}: {}", status.code(), status.message()));
            }
        };

        let result = response.into_inner();
        if result.success {
            Ok(result.result)
        } else {
            Err(result.error)
        }
    }
}

// ============================================================================
// Task Type Detection
// ============================================================================

#[derive(Debug, Clone, PartialEq)]
enum TaskCategory {
    Echo,
    // NLP Tasks
    FeatureExtraction,
    TextClassification,
    TokenClassification,
    QuestionAnswering,
    FillMask,
    TextGeneration,
    Summarization,
    Translation,
    SentenceSimilarity,
    ZeroShotClassification,
    // Audio Tasks
    AutomaticSpeechRecognition,
    TextToSpeech,
    AudioClassification,
    AudioTextToText,
    // Vision Tasks
    ImageClassification,
    ObjectDetection,
    ImageSegmentation,
    DepthEstimation,
    ImageToText,
    ImageFeatureExtraction,
    ZeroShotImageClassification,
    // Multimodal Tasks
    VisualQuestionAnswering,
    DocumentQuestionAnswering,
    ImageTextToText,
    // Unknown
    Unknown,
}

fn detect_task_category(task_name: &str) -> TaskCategory {
    let name = task_name.to_lowercase();

    // Echo
    if name.contains("echo") {
        return TaskCategory::Echo;
    }

    // NLP Tasks
    if name.contains("feature-extraction") || name.contains("embed") {
        return TaskCategory::FeatureExtraction;
    }
    if name.contains("text-classification") || name.contains("sentiment") {
        return TaskCategory::TextClassification;
    }
    if name.contains("token-classification") || name.contains("ner") {
        return TaskCategory::TokenClassification;
    }
    if name.contains("question-answering") && !name.contains("visual") && !name.contains("document")
    {
        return TaskCategory::QuestionAnswering;
    }
    if name.contains("fill-mask") {
        return TaskCategory::FillMask;
    }
    if name.contains("text-generation") || name.contains("causal-lm") {
        return TaskCategory::TextGeneration;
    }
    if name.contains("summarization") {
        return TaskCategory::Summarization;
    }
    if name.contains("translation") {
        return TaskCategory::Translation;
    }
    if name.contains("sentence-similarity") {
        return TaskCategory::SentenceSimilarity;
    }
    if name.contains("zero-shot-classification") && !name.contains("image") {
        return TaskCategory::ZeroShotClassification;
    }

    // Audio Tasks
    if name.contains("speech-recognition") || name.contains("asr") || name.contains("whisper") {
        return TaskCategory::AutomaticSpeechRecognition;
    }
    if name.contains("text-to-speech") || name.contains("tts") {
        return TaskCategory::TextToSpeech;
    }
    if name.contains("audio-classification") {
        return TaskCategory::AudioClassification;
    }
    if name.contains("audio-text-to-text") {
        return TaskCategory::AudioTextToText;
    }

    // Vision Tasks
    if name.contains("image-classification") && !name.contains("zero-shot") {
        return TaskCategory::ImageClassification;
    }
    if name.contains("object-detection") {
        return TaskCategory::ObjectDetection;
    }
    if name.contains("image-segmentation") {
        return TaskCategory::ImageSegmentation;
    }
    if name.contains("depth-estimation") {
        return TaskCategory::DepthEstimation;
    }
    if name.contains("image-to-text") || name.contains("ocr") {
        return TaskCategory::ImageToText;
    }
    if name.contains("image-feature-extraction") {
        return TaskCategory::ImageFeatureExtraction;
    }
    if name.contains("zero-shot-image-classification") {
        return TaskCategory::ZeroShotImageClassification;
    }

    // Multimodal Tasks
    if name.contains("visual-question-answering") || name.contains("vqa") {
        return TaskCategory::VisualQuestionAnswering;
    }
    if name.contains("document-question-answering") {
        return TaskCategory::DocumentQuestionAnswering;
    }
    if name.contains("image-text-to-text") || name.contains("vlm") {
        return TaskCategory::ImageTextToText;
    }

    TaskCategory::Unknown
}

/// Check if task uses text/tokenized input
fn is_text_input_task(cat: &TaskCategory) -> bool {
    matches!(
        cat,
        TaskCategory::FeatureExtraction
            | TaskCategory::TextClassification
            | TaskCategory::TokenClassification
            | TaskCategory::QuestionAnswering
            | TaskCategory::FillMask
            | TaskCategory::TextGeneration
            | TaskCategory::Summarization
            | TaskCategory::Translation
            | TaskCategory::SentenceSimilarity
            | TaskCategory::ZeroShotClassification
    )
}

/// Check if task uses image/pixel input
fn is_image_input_task(cat: &TaskCategory) -> bool {
    matches!(
        cat,
        TaskCategory::ImageClassification
            | TaskCategory::ObjectDetection
            | TaskCategory::ImageSegmentation
            | TaskCategory::DepthEstimation
            | TaskCategory::ImageToText
            | TaskCategory::ImageFeatureExtraction
            | TaskCategory::ZeroShotImageClassification
            | TaskCategory::VisualQuestionAnswering
            | TaskCategory::DocumentQuestionAnswering
            | TaskCategory::ImageTextToText
    )
}

/// Check if task uses audio input
fn is_audio_input_task(cat: &TaskCategory) -> bool {
    matches!(
        cat,
        TaskCategory::AutomaticSpeechRecognition
            | TaskCategory::AudioClassification
            | TaskCategory::AudioTextToText
    )
}

// ============================================================================
// Test Payloads - Natural Language Format
// ============================================================================
// The service accepts natural language inputs and handles preprocessing internally.
// LangChain sends text/S3 URIs, the Rust service tokenizes/processes them.

/// Simple text input for NLP tasks
fn text_payload_single() -> String {
    serde_json::json!({
        "text": "Hello world, this is a test sentence."
    })
    .to_string()
}

/// Batch of texts for NLP tasks
fn text_payload_batch() -> String {
    serde_json::json!({
        "text": [
            "Hello world, this is the first sentence.",
            "What is the meaning of life?"
        ]
    })
    .to_string()
}

/// Question-answering payload with question + context
fn qa_payload() -> String {
    serde_json::json!({
        "question": "What is the capital of France?",
        "context": "Paris is the capital and most populous city of France."
    })
    .to_string()
}

/// Fill-mask payload with [MASK] token
fn fill_mask_payload() -> String {
    serde_json::json!({
        "text": "The sky is [MASK] today."
    })
    .to_string()
}

/// Image input with embedded base64 data (no external dependencies).
fn image_payload_224() -> String {
    fixtures::image_payload_base64()
}

/// Image input with embedded base64 (alias for smaller test image).
fn image_payload_small() -> String {
    fixtures::image_payload_base64()
}

/// Audio input with embedded base64 WAV data (no external dependencies).
fn audio_payload() -> String {
    fixtures::audio_payload_base64()
}

/// Multimodal payload: image + text (VQA style) with embedded base64 image.
fn multimodal_payload() -> String {
    fixtures::vqa_payload_base64("What is shown in this image?")
}

/// TTS payload: text to synthesize
fn tts_payload() -> String {
    serde_json::json!({
        "text": "Hello, this is a test of text to speech."
    })
    .to_string()
}

/// Document QA payload (DocVQA)
fn document_qa_payload() -> String {
    serde_json::json!({
        "document": "s3://test-bucket/documents/invoice.pdf",
        "question": "What is the total amount?"
    })
    .to_string()
}

/// Zero-shot classification with candidate labels
fn zero_shot_payload() -> String {
    serde_json::json!({
        "text": "I love this product, it works great!",
        "candidate_labels": ["positive", "negative", "neutral"]
    })
    .to_string()
}

/// Zero-shot image classification with embedded base64 image.
fn zero_shot_image_payload() -> String {
    fixtures::zero_shot_image_payload_base64(&["cat", "dog", "bird", "fish"])
}

/// Translation payload
fn translation_payload() -> String {
    serde_json::json!({
        "text": "Hello, how are you today?"
    })
    .to_string()
}

/// Summarization payload (longer text)
fn summarization_payload() -> String {
    serde_json::json!({
        "text": "Artificial intelligence is transforming the way we live and work. Machine learning models can now understand natural language, recognize images, and generate creative content. These advancements are being applied across industries from healthcare to finance."
    })
    .to_string()
}

/// Sentence similarity payload (text pair)
fn sentence_similarity_payload() -> String {
    serde_json::json!({
        "text_a": "The cat sat on the mat.",
        "text_b": "A feline rested on the rug."
    })
    .to_string()
}

/// Echo payload (for testing)
fn echo_payload() -> String {
    serde_json::json!({"message": "hello world", "number": 42}).to_string()
}

// ============================================================================
// Health & Discovery Tests
// ============================================================================

#[tokio::test]
async fn test_01_health_check() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running on {SERVER_ADDR}");
        return;
    }
    println!("Health check: PASSED");
}

#[tokio::test]
async fn test_02_discover_task() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let task_name = helpers::get_task_name(SERVER_ADDR).await;
    match &task_name {
        Some(name) => {
            let category = detect_task_category(name);
            println!("Discovered task: {name} (category: {category:?})");
        }
        None => panic!("Failed to discover task name"),
    }
}

// ============================================================================
// Echo Task Test
// ============================================================================

#[tokio::test]
async fn test_echo_task() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::Echo {
        eprintln!("SKIP: server running {category:?}, not Echo");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &echo_payload())
        .await
        .expect("Echo task failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert!(parsed["echo"].is_string(), "Expected echo field");
    println!("Echo task: PASSED");
}

// ============================================================================
// NLP Task Tests
// ============================================================================

#[tokio::test]
async fn test_feature_extraction() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::FeatureExtraction {
        eprintln!("SKIP: server running {category:?}, not FeatureExtraction");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &text_payload_single())
        .await
        .expect("Feature extraction failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");

    // Should have last_hidden_state or similar
    assert!(!outputs.is_empty(), "Expected output tensors");
    println!(
        "Feature extraction: PASSED (outputs: {:?})",
        outputs.keys().collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_text_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::TextClassification {
        eprintln!("SKIP: server running {category:?}, not TextClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &text_payload_single())
        .await
        .expect("Text classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");

    // Should have logits
    assert!(outputs.contains_key("logits") || !outputs.is_empty());
    println!("Text classification: PASSED");
}

#[tokio::test]
async fn test_token_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::TokenClassification {
        eprintln!("SKIP: server running {category:?}, not TokenClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &text_payload_single())
        .await
        .expect("Token classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Token classification: PASSED");
}

#[tokio::test]
async fn test_question_answering() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::QuestionAnswering {
        eprintln!("SKIP: server running {category:?}, not QuestionAnswering");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &qa_payload())
        .await
        .expect("Question answering failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();

    // Postprocessed output format: {"answer": "...", "score": ..., "start": ..., "end": ...}
    // Or raw output format: {"outputs": {"start_logits": [...], "end_logits": [...]}}
    if parsed.get("answer").is_some() {
        // Postprocessed format
        assert!(parsed.get("score").is_some(), "Expected score in QA output");
        println!(
            "Question answering: PASSED (answer: {:?}, score: {:?})",
            parsed.get("answer"),
            parsed.get("score")
        );
    } else if let Some(outputs) = parsed.get("outputs").and_then(|o| o.as_object()) {
        // Raw output format (legacy)
        assert!(
            outputs.contains_key("start_logits") || !outputs.is_empty(),
            "Expected start_logits or non-empty outputs"
        );
        println!(
            "Question answering: PASSED (outputs: {:?})",
            outputs.keys().collect::<Vec<_>>()
        );
    } else {
        panic!("Unexpected QA output format: {parsed}");
    }
}

#[tokio::test]
async fn test_fill_mask() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::FillMask {
        eprintln!("SKIP: server running {category:?}, not FillMask");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &fill_mask_payload())
        .await
        .expect("Fill-mask failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Fill-mask: PASSED");
}

#[tokio::test]
async fn test_text_generation() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::TextGeneration {
        eprintln!("SKIP: server running {category:?}, not TextGeneration");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &text_payload_single())
        .await
        .expect("Text generation failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Text generation: PASSED");
}

#[tokio::test]
async fn test_summarization() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::Summarization {
        eprintln!("SKIP: server running {category:?}, not Summarization");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &summarization_payload())
        .await
        .expect("Summarization failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Summarization: PASSED");
}

#[tokio::test]
async fn test_translation() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::Translation {
        eprintln!("SKIP: server running {category:?}, not Translation");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &translation_payload())
        .await
        .expect("Translation failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Translation: PASSED");
}

#[tokio::test]
async fn test_sentence_similarity() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::SentenceSimilarity {
        eprintln!("SKIP: server running {category:?}, not SentenceSimilarity");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &sentence_similarity_payload())
        .await
        .expect("Sentence similarity failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Sentence similarity: PASSED");
}

#[tokio::test]
async fn test_zero_shot_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ZeroShotClassification {
        eprintln!("SKIP: server running {category:?}, not ZeroShotClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &zero_shot_payload())
        .await
        .expect("Zero-shot classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Zero-shot classification: PASSED");
}

// ============================================================================
// Audio Task Tests
// ============================================================================

#[tokio::test]
async fn test_automatic_speech_recognition() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::AutomaticSpeechRecognition {
        eprintln!("SKIP: server running {category:?}, not ASR");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &audio_payload())
        .await
        .expect("ASR failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Automatic speech recognition: PASSED");
}

#[tokio::test]
async fn test_text_to_speech() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::TextToSpeech {
        eprintln!("SKIP: server running {category:?}, not TTS");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &tts_payload())
        .await
        .expect("TTS failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Text-to-speech: PASSED");
}

#[tokio::test]
async fn test_audio_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::AudioClassification {
        eprintln!("SKIP: server running {category:?}, not AudioClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &audio_payload())
        .await
        .expect("Audio classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Audio classification: PASSED");
}

#[tokio::test]
async fn test_audio_text_to_text() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::AudioTextToText {
        eprintln!("SKIP: server running {category:?}, not AudioTextToText");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &audio_payload())
        .await
        .expect("Audio-text-to-text failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Audio-text-to-text: PASSED");
}

// ============================================================================
// Vision Task Tests
// ============================================================================

#[tokio::test]
async fn test_image_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ImageClassification {
        eprintln!("SKIP: server running {category:?}, not ImageClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Image classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Image classification: PASSED");
}

#[tokio::test]
async fn test_object_detection() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ObjectDetection {
        eprintln!("SKIP: server running {category:?}, not ObjectDetection");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Object detection failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Object detection: PASSED");
}

#[tokio::test]
async fn test_image_segmentation() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ImageSegmentation {
        eprintln!("SKIP: server running {category:?}, not ImageSegmentation");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Image segmentation failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Image segmentation: PASSED");
}

#[tokio::test]
async fn test_depth_estimation() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::DepthEstimation {
        eprintln!("SKIP: server running {category:?}, not DepthEstimation");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Depth estimation failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Depth estimation: PASSED");
}

#[tokio::test]
async fn test_image_to_text() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ImageToText {
        eprintln!("SKIP: server running {category:?}, not ImageToText/OCR");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Image-to-text failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Image-to-text (OCR): PASSED");
}

#[tokio::test]
async fn test_image_feature_extraction() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ImageFeatureExtraction {
        eprintln!("SKIP: server running {category:?}, not ImageFeatureExtraction");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &image_payload_224())
        .await
        .expect("Image feature extraction failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Image feature extraction: PASSED");
}

#[tokio::test]
async fn test_zero_shot_image_classification() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ZeroShotImageClassification {
        eprintln!("SKIP: server running {category:?}, not ZeroShotImageClassification");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &zero_shot_image_payload())
        .await
        .expect("Zero-shot image classification failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Zero-shot image classification: PASSED");
}

// ============================================================================
// Multimodal Task Tests
// ============================================================================

#[tokio::test]
async fn test_visual_question_answering() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::VisualQuestionAnswering {
        eprintln!("SKIP: server running {category:?}, not VQA");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &multimodal_payload())
        .await
        .expect("VQA failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Visual question answering: PASSED");
}

#[tokio::test]
async fn test_document_question_answering() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::DocumentQuestionAnswering {
        eprintln!("SKIP: server running {category:?}, not DocumentQA");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &document_qa_payload())
        .await
        .expect("Document QA failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Document question answering: PASSED");
}

#[tokio::test]
async fn test_image_text_to_text() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);
    if category != TaskCategory::ImageTextToText {
        eprintln!("SKIP: server running {category:?}, not ImageTextToText/VLM");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &multimodal_payload())
        .await
        .expect("Image-text-to-text failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    assert!(!outputs.is_empty());
    println!("Image-text-to-text (VLM): PASSED");
}

// ============================================================================
// Generic Tests (work with any task)
// ============================================================================

#[tokio::test]
async fn test_batch_processing() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);

    // Only test batch for simple text classification tasks (text -> output)
    // QA, similarity, and other structured input tasks don't support simple batching
    let batchable_tasks = matches!(
        category,
        TaskCategory::TextClassification
            | TaskCategory::TokenClassification
            | TaskCategory::FeatureExtraction
            | TaskCategory::FillMask
            | TaskCategory::TextGeneration
            | TaskCategory::Summarization
            | TaskCategory::Translation
    );

    if !batchable_tasks {
        eprintln!("SKIP: batch test only for simple text tasks, got {category:?}");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, &text_payload_batch())
        .await
        .expect("Batch processing failed");

    let parsed: serde_json::Value = serde_json::from_str(&result).unwrap();
    let outputs = parsed["outputs"].as_object().expect("Expected outputs");
    let first_output = outputs.values().next().expect("Expected output tensor");
    let batch = first_output.as_array().expect("Expected array");

    assert_eq!(batch.len(), 2, "Batch size should be 2");
    println!("Batch processing: PASSED");
}

#[tokio::test]
async fn test_concurrent_requests() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    let category = detect_task_category(&task_name);

    // Choose appropriate payload
    let payload = match category {
        TaskCategory::Echo => echo_payload(),
        cat if is_text_input_task(&cat) => text_payload_single(),
        cat if is_image_input_task(&cat) => image_payload_small(),
        cat if is_audio_input_task(&cat) => audio_payload(),
        _ => {
            eprintln!("SKIP: unknown task category {category:?}");
            return;
        }
    };

    let num_requests = 10;
    let mut handles = Vec::new();

    for _ in 0..num_requests {
        let p = payload.clone();
        let t = task_name.clone();
        handles.push(tokio::spawn(async move {
            helpers::execute_task(SERVER_ADDR, &t, &p).await
        }));
    }

    let mut success = 0;
    for handle in handles {
        if handle.await.unwrap().is_ok() {
            success += 1;
        }
    }

    assert_eq!(
        success, num_requests,
        "All concurrent requests should succeed"
    );
    println!("Concurrent requests ({success}/{num_requests}): PASSED");
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
async fn test_unknown_task_error() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, "nonexistent.task.v1", "{}").await;
    assert!(result.is_err(), "Unknown task should return error");
    println!("Unknown task error: PASSED");
}

#[tokio::test]
async fn test_invalid_payload_error() {
    if !helpers::check_health(SERVER_ADDR).await {
        eprintln!("SKIP: server not running");
        return;
    }

    let Some(task_name) = helpers::get_task_name(SERVER_ADDR).await else {
        eprintln!("SKIP: could not discover task");
        return;
    };

    // Echo accepts anything, skip for echo
    if detect_task_category(&task_name) == TaskCategory::Echo {
        eprintln!("SKIP: echo task accepts any payload");
        return;
    }

    let result = helpers::execute_task(SERVER_ADDR, &task_name, "not valid json {{{").await;
    assert!(result.is_err(), "Invalid JSON should return error");
    println!("Invalid payload error: PASSED");
}
