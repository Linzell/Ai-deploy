//! Output types for postprocessing.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Postprocessed output in human-readable format.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PostprocessedOutput {
    /// Text output (generation, translation, ASR)
    Text(TextOutput),

    /// Classification output (single label)
    Classification(ClassificationOutput),

    /// Multi-label classification
    MultiLabel(MultiLabelOutput),

    /// Question answering output
    QuestionAnswer(QAOutput),

    /// Embedding output (feature extraction)
    Embedding(EmbeddingOutput),

    /// Token classification (NER, POS)
    TokenClassification(TokenClassificationOutput),

    /// Object detection
    ObjectDetection(ObjectDetectionOutput),

    /// Generic output (fallback)
    Generic(GenericOutput),
}

impl PostprocessedOutput {
    /// Convert to JSON value.
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }

    /// Convert to JSON string.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string(self).unwrap_or_default()
    }
}

impl fmt::Display for PostprocessedOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Text(t) => write!(f, "{t}"),
            Self::Classification(c) => write!(f, "{c}"),
            Self::MultiLabel(m) => write!(f, "{m}"),
            Self::QuestionAnswer(q) => write!(f, "{q}"),
            Self::Embedding(e) => write!(f, "{e}"),
            Self::TokenClassification(t) => write!(f, "{t}"),
            Self::ObjectDetection(o) => write!(f, "{o}"),
            Self::Generic(g) => write!(f, "{g}"),
        }
    }
}

/// Text output for generation tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextOutput {
    /// Generated text
    pub text: String,
    /// Optional: tokens generated
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tokens: Option<Vec<String>>,
}

impl TextOutput {
    /// Create a new text output.
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: None,
        }
    }

    /// Add tokens to the output.
    #[must_use]
    pub fn with_tokens(mut self, tokens: Vec<String>) -> Self {
        self.tokens = Some(tokens);
        self
    }
}

impl fmt::Display for TextOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.text)
    }
}

/// Classification output for single-label tasks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationOutput {
    /// Predicted label
    pub label: String,
    /// Confidence score (0-1)
    pub score: f32,
    /// Optional: all scores
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scores: Option<HashMap<String, f32>>,
}

impl ClassificationOutput {
    /// Create a new classification output.
    pub fn new(label: impl Into<String>, score: f32) -> Self {
        Self {
            label: label.into(),
            score,
            scores: None,
        }
    }

    /// Add all label scores.
    #[must_use]
    pub fn with_all_scores(mut self, scores: HashMap<String, f32>) -> Self {
        self.scores = Some(scores);
        self
    }
}

impl fmt::Display for ClassificationOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.2}%)", self.label, self.score * 100.0)
    }
}

/// Multi-label classification output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiLabelOutput {
    /// Labels with scores above threshold
    pub labels: Vec<LabelScore>,
}

impl fmt::Display for MultiLabelOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let labels: Vec<String> = self
            .labels
            .iter()
            .map(|l| format!("{} ({:.2}%)", l.label, l.score * 100.0))
            .collect();
        write!(f, "{}", labels.join(", "))
    }
}

/// A single label with its score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LabelScore {
    /// Label name
    pub label: String,
    /// Confidence score
    pub score: f32,
}

impl LabelScore {
    /// Create a new label score.
    pub fn new(label: impl Into<String>, score: f32) -> Self {
        Self {
            label: label.into(),
            score,
        }
    }
}

/// Question answering output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QAOutput {
    /// Extracted answer text
    pub answer: String,
    /// Confidence score
    pub score: f32,
    /// Start position in context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    /// End position in context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
}

impl QAOutput {
    /// Create a new QA output.
    pub fn new(answer: impl Into<String>, score: f32) -> Self {
        Self {
            answer: answer.into(),
            score,
            start: None,
            end: None,
        }
    }

    /// Add span positions.
    #[must_use]
    pub fn with_span(mut self, start: usize, end: usize) -> Self {
        self.start = Some(start);
        self.end = Some(end);
        self
    }
}

impl fmt::Display for QAOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} ({:.2}%)", self.answer, self.score * 100.0)
    }
}

/// Embedding output for feature extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingOutput {
    /// Embedding vector
    pub embedding: Vec<f32>,
    /// Optional: dimensionality
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dimensions: Option<usize>,
}

impl EmbeddingOutput {
    /// Create a new embedding output.
    pub fn new(embedding: Vec<f32>) -> Self {
        let dimensions = Some(embedding.len());
        Self {
            embedding,
            dimensions,
        }
    }
}

impl fmt::Display for EmbeddingOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[embedding: {} dimensions]",
            self.dimensions.unwrap_or(0)
        )
    }
}

/// Token classification output (NER, POS tagging).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenClassificationOutput {
    /// Entities or tagged tokens
    pub entities: Vec<Entity>,
}

impl fmt::Display for TokenClassificationOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entities: Vec<String> = self
            .entities
            .iter()
            .map(|e| format!("{}:{}", e.label, e.text))
            .collect();
        write!(f, "[{}]", entities.join(", "))
    }
}

/// A single entity from token classification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    /// Entity type/label
    pub label: String,
    /// Entity text
    pub text: String,
    /// Confidence score
    pub score: f32,
    /// Start character position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start: Option<usize>,
    /// End character position
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end: Option<usize>,
}

impl Entity {
    /// Create a new entity.
    pub fn new(label: impl Into<String>, text: impl Into<String>, score: f32) -> Self {
        Self {
            label: label.into(),
            text: text.into(),
            score,
            start: None,
            end: None,
        }
    }
}

/// Object detection output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectDetectionOutput {
    /// Detected objects
    pub objects: Vec<DetectedObject>,
}

impl fmt::Display for ObjectDetectionOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} objects detected]", self.objects.len())
    }
}

/// A single detected object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Object label/class
    pub label: String,
    /// Confidence score
    pub score: f32,
    /// Bounding box [x_min, y_min, x_max, y_max]
    pub bbox: [f32; 4],
}

impl DetectedObject {
    /// Create a new detected object.
    pub fn new(label: impl Into<String>, score: f32, bbox: [f32; 4]) -> Self {
        Self {
            label: label.into(),
            score,
            bbox,
        }
    }
}

/// Generic output (raw values or unsupported task types).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenericOutput {
    /// Raw outputs as JSON
    pub outputs: HashMap<String, serde_json::Value>,
}

impl GenericOutput {
    /// Create a new generic output.
    pub fn new(outputs: HashMap<String, serde_json::Value>) -> Self {
        Self { outputs }
    }
}

impl fmt::Display for GenericOutput {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[generic: {} outputs]", self.outputs.len())
    }
}
