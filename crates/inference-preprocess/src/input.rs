//! Raw input types from LangChain/LangGraph.
//!
//! These represent the user-friendly input formats that the service accepts.
//! The preprocessor converts these to tensor format for ONNX inference.

use serde::{Deserialize, Serialize};

/// Raw input from LangChain/LangGraph.
///
/// Supports multiple input types:
/// - Text: Single string or list of strings
/// - Image: S3 URI, base64, or local path
/// - Audio: S3 URI or local path
/// - Multimodal: Combinations of the above
///
/// NOTE: Order matters for serde(untagged) - more specific variants must come first.
/// VisionLanguage must come before Image, Document before VisionLanguage, etc.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum RawInput {
    /// Document understanding input (DocVQA) - has document + question
    Document(DocumentInput),

    /// Vision-language input (VQA, image captioning with prompt) - has image + text
    VisionLanguage(VisionLanguageInput),

    /// Question-answering input (question + context) - has question + context
    QuestionAnswer(QAInput),

    /// Text pair for similarity/reranking - has text_a + text_b
    TextPair(TextPairInput),

    /// Image input (classification, detection, etc.) - has image only
    Image(ImageInput),

    /// Audio input (ASR, audio classification) - has audio only
    Audio(AudioInput),

    /// Simple text input - has text only (must be last as it's most generic)
    Text(TextInput),
}

/// Simple text input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextInput {
    /// Single text or batch of texts
    #[serde(alias = "texts")]
    pub text: StringOrVec,

    /// Optional: prefix to add (e.g., "query: " for E5)
    #[serde(default)]
    pub prefix: Option<String>,
}

/// Question-answering input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAInput {
    pub question: String,
    pub context: String,
}

/// Text pair for similarity or reranking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextPairInput {
    /// Query/sentence A
    pub text_a: String,
    /// Document/sentence B
    pub text_b: String,
}

/// Image input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageInput {
    /// S3 URI (s3://bucket/key), base64 string, or local path
    #[serde(alias = "image_url", alias = "image_path")]
    pub image: String,

    /// Image source type (auto-detected if not specified)
    #[serde(default)]
    pub source: Option<ImageSource>,
}

/// Audio input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AudioInput {
    /// S3 URI (s3://bucket/key) or local path
    #[serde(alias = "audio_url", alias = "audio_path")]
    pub audio: String,

    /// Optional: language hint for ASR
    #[serde(default)]
    pub language: Option<String>,
}

/// Vision-language input (VQA, captioning).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VisionLanguageInput {
    /// Image (S3 URI, base64, or local path)
    pub image: String,

    /// Text prompt/question
    #[serde(alias = "question", alias = "prompt")]
    pub text: String,
}

/// Document understanding input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentInput {
    /// Document image (S3 URI, base64, or local path)
    #[serde(alias = "image")]
    pub document: String,

    /// Question about the document
    pub question: String,
}

/// Image source type.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ImageSource {
    #[default]
    Auto,
    S3,
    Base64,
    Local,
    Url,
}

/// String or vector of strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum StringOrVec {
    Single(String),
    Multiple(Vec<String>),
}

impl StringOrVec {
    /// Convert to a vector of strings.
    pub fn to_vec(&self) -> Vec<String> {
        match self {
            StringOrVec::Single(s) => vec![s.clone()],
            StringOrVec::Multiple(v) => v.clone(),
        }
    }

    /// Get the number of strings.
    pub fn len(&self) -> usize {
        match self {
            StringOrVec::Single(_) => 1,
            StringOrVec::Multiple(v) => v.len(),
        }
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        match self {
            StringOrVec::Single(s) => s.is_empty(),
            StringOrVec::Multiple(v) => v.is_empty(),
        }
    }
}

impl RawInput {
    /// Create a simple text input.
    pub fn text(s: impl Into<String>) -> Self {
        RawInput::Text(TextInput {
            text: StringOrVec::Single(s.into()),
            prefix: None,
        })
    }

    /// Create a text input with prefix.
    pub fn text_with_prefix(s: impl Into<String>, prefix: impl Into<String>) -> Self {
        RawInput::Text(TextInput {
            text: StringOrVec::Single(s.into()),
            prefix: Some(prefix.into()),
        })
    }

    /// Create a batch text input.
    pub fn texts(texts: Vec<String>) -> Self {
        RawInput::Text(TextInput {
            text: StringOrVec::Multiple(texts),
            prefix: None,
        })
    }

    /// Create a QA input.
    pub fn qa(question: impl Into<String>, context: impl Into<String>) -> Self {
        RawInput::QuestionAnswer(QAInput {
            question: question.into(),
            context: context.into(),
        })
    }

    /// Create a text pair input (for similarity/reranking).
    pub fn text_pair(text_a: impl Into<String>, text_b: impl Into<String>) -> Self {
        RawInput::TextPair(TextPairInput {
            text_a: text_a.into(),
            text_b: text_b.into(),
        })
    }

    /// Create an image input from S3.
    pub fn image_s3(uri: impl Into<String>) -> Self {
        RawInput::Image(ImageInput {
            image: uri.into(),
            source: Some(ImageSource::S3),
        })
    }

    /// Create an image input from base64.
    pub fn image_base64(data: impl Into<String>) -> Self {
        RawInput::Image(ImageInput {
            image: data.into(),
            source: Some(ImageSource::Base64),
        })
    }

    /// Create an audio input from S3.
    pub fn audio_s3(uri: impl Into<String>) -> Self {
        RawInput::Audio(AudioInput {
            audio: uri.into(),
            language: None,
        })
    }

    /// Create a vision-language input.
    pub fn vqa(image: impl Into<String>, question: impl Into<String>) -> Self {
        RawInput::VisionLanguage(VisionLanguageInput {
            image: image.into(),
            text: question.into(),
        })
    }

    /// Create a document QA input.
    pub fn doc_qa(document: impl Into<String>, question: impl Into<String>) -> Self {
        RawInput::Document(DocumentInput {
            document: document.into(),
            question: question.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_text_input() {
        let json = r#"{"text": "Hello world"}"#;
        let input: RawInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input, RawInput::Text(_)));
    }

    #[test]
    fn test_parse_qa_input() {
        let json =
            r#"{"question": "What is Paris?", "context": "Paris is the capital of France."}"#;
        let input: RawInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input, RawInput::QuestionAnswer(_)));
    }

    #[test]
    fn test_parse_image_input() {
        let json = r#"{"image": "s3://bucket/image.jpg"}"#;
        let input: RawInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input, RawInput::Image(_)));
    }

    #[test]
    fn test_parse_audio_input() {
        let json = r#"{"audio": "s3://bucket/audio.wav", "language": "en"}"#;
        let input: RawInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input, RawInput::Audio(_)));
    }

    #[test]
    fn test_parse_vqa_input() {
        let json = r#"{"image": "s3://bucket/img.jpg", "text": "What is in this image?"}"#;
        let input: RawInput = serde_json::from_str(json).unwrap();
        assert!(matches!(input, RawInput::VisionLanguage(_)));
    }
}
