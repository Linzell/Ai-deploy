//! Test fixtures for integration tests.
//!
//! Provides minimal valid test data for images, audio, and documents
//! that can be used without requiring external resources (S3, files).
//!
//! Note: Some functions are not yet used but provided for future test expansion.

#![allow(dead_code)]

use base64::{engine::general_purpose::STANDARD as BASE64, Engine};

/// Minimal valid 8x8 red PNG image (89 bytes).
/// Generated programmatically - a tiny red square.
pub const TEST_IMAGE_PNG_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAAAgAAAAICAIAAABLbSncAAAADklEQVQI12P4z8DAwMAAAAQAAXnhfPgAAAAASUVORK5CYII=";

/// Minimal valid 16x16 RGB PNG image for models requiring larger input.
/// This is a simple gradient image.
pub const TEST_IMAGE_16X16_BASE64: &str = "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAKElEQVQoz2P4z8DAwMDAxAAGjCQp+M/AwMBIsgKG/wwMDIxDyQUAAAD//wMAGHwFfRhBvC8AAAAASUVORK5CYII=";

/// Minimal valid WAV file (1 second of silence at 16kHz mono).
/// 44-byte header + 32000 bytes of silence = 32044 bytes total.
/// This is a valid WAV that Whisper and other audio models can process.
fn generate_silent_wav() -> Vec<u8> {
    let sample_rate: u32 = 16000;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let num_samples: u32 = sample_rate; // 1 second
    let data_size: u32 = num_samples * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let file_size: u32 = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes()); // chunk size
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * (bits_per_sample / 8);
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    // Silent audio data (all zeros)
    wav.resize(44 + data_size as usize, 0);

    wav
}

/// Generate a short WAV with a simple tone (more interesting than silence).
fn generate_tone_wav(duration_ms: u32, frequency_hz: u32) -> Vec<u8> {
    let sample_rate: u32 = 16000;
    let num_channels: u16 = 1;
    let bits_per_sample: u16 = 16;
    let num_samples: u32 = sample_rate * duration_ms / 1000;
    let data_size: u32 = num_samples * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    let file_size: u32 = 36 + data_size;

    let mut wav = Vec::with_capacity(44 + data_size as usize);

    // RIFF header
    wav.extend_from_slice(b"RIFF");
    wav.extend_from_slice(&file_size.to_le_bytes());
    wav.extend_from_slice(b"WAVE");

    // fmt chunk
    wav.extend_from_slice(b"fmt ");
    wav.extend_from_slice(&16u32.to_le_bytes());
    wav.extend_from_slice(&1u16.to_le_bytes()); // PCM
    wav.extend_from_slice(&num_channels.to_le_bytes());
    wav.extend_from_slice(&sample_rate.to_le_bytes());
    let byte_rate = sample_rate * u32::from(num_channels) * u32::from(bits_per_sample / 8);
    wav.extend_from_slice(&byte_rate.to_le_bytes());
    let block_align = num_channels * (bits_per_sample / 8);
    wav.extend_from_slice(&block_align.to_le_bytes());
    wav.extend_from_slice(&bits_per_sample.to_le_bytes());

    // data chunk
    wav.extend_from_slice(b"data");
    wav.extend_from_slice(&data_size.to_le_bytes());

    // Generate sine wave
    let amplitude: f32 = 16000.0; // ~50% of i16 max
    #[allow(clippy::cast_precision_loss)] // Precision loss acceptable for audio generation
    let angular_freq =
        2.0 * std::f32::consts::PI * f32::from(frequency_hz as u16) / f32::from(sample_rate as u16);

    for i in 0..num_samples {
        #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
        let sample = (amplitude * (angular_freq * i as f32).sin()) as i16;
        wav.extend_from_slice(&sample.to_le_bytes());
    }

    wav
}

/// Get test image as base64 data URI.
pub fn test_image_data_uri() -> String {
    format!("data:image/png;base64,{TEST_IMAGE_PNG_BASE64}")
}

/// Get larger test image as base64 data URI.
pub fn test_image_16x16_data_uri() -> String {
    format!("data:image/png;base64,{TEST_IMAGE_16X16_BASE64}")
}

/// Get test audio (1 second silence) as base64 data URI.
pub fn test_audio_data_uri() -> String {
    let wav = generate_silent_wav();
    let b64 = BASE64.encode(&wav);
    format!("data:audio/wav;base64,{b64}")
}

/// Get test audio (short tone) as base64 data URI.
pub fn test_audio_tone_data_uri(duration_ms: u32) -> String {
    let wav = generate_tone_wav(duration_ms, 440); // 440 Hz = A4 note
    let b64 = BASE64.encode(&wav);
    format!("data:audio/wav;base64,{b64}")
}

/// Get raw test audio bytes (for direct processing).
pub fn test_audio_bytes() -> Vec<u8> {
    generate_silent_wav()
}

/// Get raw test audio bytes with tone.
pub fn test_audio_tone_bytes(duration_ms: u32) -> Vec<u8> {
    generate_tone_wav(duration_ms, 440)
}

/// Decode base64 image to bytes.
pub fn decode_test_image() -> Vec<u8> {
    BASE64
        .decode(TEST_IMAGE_PNG_BASE64)
        .expect("Invalid base64 image")
}

// ============================================================================
// Test Payload Generators
// ============================================================================

/// Generate a text payload for NLP tasks.
pub fn text_payload(text: &str) -> String {
    serde_json::json!({ "text": text }).to_string()
}

/// Generate a batch text payload.
pub fn text_batch_payload(texts: &[&str]) -> String {
    serde_json::json!({ "text": texts }).to_string()
}

/// Generate a QA payload.
pub fn qa_payload(question: &str, context: &str) -> String {
    serde_json::json!({
        "question": question,
        "context": context
    })
    .to_string()
}

/// Generate a text pair payload (for similarity).
pub fn text_pair_payload(text_a: &str, text_b: &str) -> String {
    serde_json::json!({
        "text_a": text_a,
        "text_b": text_b
    })
    .to_string()
}

/// Generate an image payload with embedded base64 data.
pub fn image_payload_base64() -> String {
    serde_json::json!({
        "image": test_image_data_uri()
    })
    .to_string()
}

/// Generate an image payload with fake S3 URI (for testing URI parsing).
pub fn image_payload_s3(bucket: &str, key: &str) -> String {
    serde_json::json!({
        "image": format!("s3://{}/{}", bucket, key)
    })
    .to_string()
}

/// Generate an audio payload with embedded base64 data.
pub fn audio_payload_base64() -> String {
    serde_json::json!({
        "audio": test_audio_data_uri(),
        "language": "en"
    })
    .to_string()
}

/// Generate an audio payload with fake S3 URI.
pub fn audio_payload_s3(bucket: &str, key: &str) -> String {
    serde_json::json!({
        "audio": format!("s3://{}/{}", bucket, key),
        "language": "en"
    })
    .to_string()
}

/// Generate a VQA payload with embedded image.
pub fn vqa_payload_base64(question: &str) -> String {
    serde_json::json!({
        "image": test_image_data_uri(),
        "text": question
    })
    .to_string()
}

/// Generate a document QA payload with fake S3 URI.
pub fn document_qa_payload_s3(bucket: &str, key: &str, question: &str) -> String {
    serde_json::json!({
        "document": format!("s3://{}/{}", bucket, key),
        "question": question
    })
    .to_string()
}

/// Generate a zero-shot classification payload.
pub fn zero_shot_payload(text: &str, labels: &[&str]) -> String {
    serde_json::json!({
        "text": text,
        "candidate_labels": labels
    })
    .to_string()
}

/// Generate a zero-shot image classification payload with embedded image.
pub fn zero_shot_image_payload_base64(labels: &[&str]) -> String {
    serde_json::json!({
        "image": test_image_data_uri(),
        "candidate_labels": labels
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wav_generation() {
        let wav = generate_silent_wav();
        // 44 byte header + 32000 bytes of 16-bit samples (16000 samples * 2 bytes)
        assert_eq!(wav.len(), 44 + 32000);
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
    }

    #[test]
    fn test_tone_wav_generation() {
        let wav = generate_tone_wav(100, 440); // 100ms tone
                                               // 44 byte header + 1600 samples * 2 bytes = 44 + 3200
        assert_eq!(wav.len(), 44 + 3200);
    }

    #[test]
    fn test_image_decode() {
        let bytes = decode_test_image();
        // PNG magic bytes
        assert_eq!(
            &bytes[0..8],
            &[0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A]
        );
    }

    #[test]
    fn test_payload_generation() {
        let payload = text_payload("Hello world");
        let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(parsed["text"], "Hello world");
    }

    #[test]
    fn test_qa_payload_generation() {
        let payload = qa_payload("What is AI?", "AI stands for artificial intelligence.");
        let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap();
        assert_eq!(parsed["question"], "What is AI?");
        assert_eq!(parsed["context"], "AI stands for artificial intelligence.");
    }

    #[test]
    fn test_image_payload_has_data_uri() {
        let payload = image_payload_base64();
        let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap();
        let image = parsed["image"].as_str().unwrap();
        assert!(image.starts_with("data:image/png;base64,"));
    }

    #[test]
    fn test_audio_payload_has_data_uri() {
        let payload = audio_payload_base64();
        let parsed: serde_json::Value = serde_json::from_str(&payload).unwrap();
        let audio = parsed["audio"].as_str().unwrap();
        assert!(audio.starts_with("data:audio/wav;base64,"));
    }
}
