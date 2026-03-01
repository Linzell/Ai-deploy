//! PaddleOCR-based Optical Character Recognition.
//!
//! PaddleOCR uses a multi-model pipeline:
//! - Detection model: detects text regions in an image → bounding boxes
//! - Recognition model: reads text from each cropped region → text strings
//!
//! ## Usage
//!
//! ```rust,ignore
//! use inference_onnx::PaddleOcrTask;
//! use inference_core::Config;
//!
//! let task = PaddleOcrTask::from_model_dir("/path/to/paddleocr", "ocr-task", &config)?;
//! let result = task.execute(r#"{"image":"data:image/png;base64,..."}"#, "req-1").await;
//! ```
//!
//! ## Input Format
//!
//! ```json
//! {"image": "s3://bucket/document.jpg"}
//! {"image": "data:image/jpeg;base64,..."}
//! ```
//!
//! ## Output Format
//!
//! ```json
//! {"text": "extracted text", "boxes": [[x1,y1,x2,y2], ...], "confidences": [0.98, ...]}
//! ```
//!
//! ## Model Files
//!
//! Expects the directory to contain:
//! - Detection model: `det_model.onnx` or `detection/*/det.onnx`
//! - Recognition model: `rec_model.onnx` or `languages/*/rec.onnx`
//! - Character dictionary: `dict.txt` or `languages/*/dict.txt`

use crate::error::{TaskError, TaskResult};
use crate::session::load_session_from_file;
use async_trait::async_trait;
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use ndarray::{s, Array4, ArrayD};
use ort::session::Session;
use ort::value::TensorRef;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

#[cfg(feature = "preprocess")]
use inference_preprocess::Preprocessor;

/// Input for PaddleOCR.
#[derive(Debug, Deserialize)]
pub struct PaddleOcrInput {
    /// Image as base64 data URI or S3/HTTP URL
    pub image: String,
}

/// A detected text region with bounding box.
#[derive(Debug, Clone, Serialize)]
pub struct TextBox {
    /// Bounding box [x1, y1, x2, y2]
    pub bbox: [f32; 4],
    /// Recognized text
    pub text: String,
    /// Confidence score
    pub confidence: f32,
}

/// Output from PaddleOCR.
#[derive(Debug, Serialize)]
pub struct PaddleOcrOutput {
    /// Concatenated text from all detected regions
    pub text: String,
    /// Individual text boxes with bounding boxes
    pub boxes: Vec<TextBox>,
    /// Number of detected text regions
    pub num_boxes: usize,
}

/// PaddleOCR task with detection and recognition models.
pub struct PaddleOcrTask {
    name: String,
    det_session: Arc<Mutex<Session>>,
    rec_session: Arc<Mutex<Session>>,
    /// Character dictionary for decoding recognition output
    char_dict: Vec<String>,
    #[cfg(feature = "preprocess")]
    preprocessor: Arc<Preprocessor>,
}

impl PaddleOcrTask {
    /// Create a new PaddleOCR task from a model directory.
    ///
    /// Expects the directory to contain detection and recognition ONNX models.
    #[cfg(feature = "preprocess")]
    #[allow(clippy::unused_async)]
    pub async fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
        preprocessor: Arc<Preprocessor>,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading PaddleOCR models"
        );

        // Find detection model
        let det_path = Self::find_model_file(
            model_dir,
            &[
                "det_model.onnx",
                "det.onnx",
                "detection/v5/det.onnx",
                "detection/v4/det.onnx",
                "detection/v3/det.onnx",
            ],
        )?;
        info!(path = %det_path.display(), "Loading detection model");
        let det_session = load_session_from_file(&det_path, config)?;

        // Find recognition model
        let rec_path = Self::find_model_file(
            model_dir,
            &[
                "rec_model.onnx",
                "rec.onnx",
                "languages/english/rec.onnx",
                "languages/latin/rec.onnx",
                "languages/multilingual/rec.onnx",
            ],
        )?;
        info!(path = %rec_path.display(), "Loading recognition model");
        let rec_session = load_session_from_file(&rec_path, config)?;

        // Load character dictionary
        let dict_path = Self::find_dict_file(model_dir)?;
        let char_dict = Self::load_char_dict(&dict_path)?;
        info!(
            path = %dict_path.display(),
            num_chars = char_dict.len(),
            "Loaded character dictionary"
        );

        // Log input/output names for debugging
        let det_inputs: Vec<String> = det_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let det_outputs: Vec<String> = det_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();
        let rec_inputs: Vec<String> = rec_session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let rec_outputs: Vec<String> = rec_session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        info!(
            det_inputs = ?det_inputs,
            det_outputs = ?det_outputs,
            rec_inputs = ?rec_inputs,
            rec_outputs = ?rec_outputs,
            "PaddleOCR models loaded"
        );

        Ok(Self {
            name,
            det_session: Arc::new(Mutex::new(det_session)),
            rec_session: Arc::new(Mutex::new(rec_session)),
            char_dict,
            preprocessor,
        })
    }

    /// Find a model file from a list of possible paths.
    fn find_model_file(model_dir: &Path, candidates: &[&str]) -> TaskResult<PathBuf> {
        for candidate in candidates {
            let path = model_dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }
        Err(TaskError::ModelNotFound(format!(
            "Could not find any of {:?} in {}",
            candidates,
            model_dir.display()
        )))
    }

    /// Find the character dictionary file.
    fn find_dict_file(model_dir: &Path) -> TaskResult<PathBuf> {
        let candidates = [
            "dict.txt",
            "ppocr_keys_v1.txt",
            "languages/english/dict.txt",
            "languages/latin/dict.txt",
            "languages/multilingual/dict.txt",
        ];

        for candidate in &candidates {
            let path = model_dir.join(candidate);
            if path.exists() {
                return Ok(path);
            }
        }

        // Also search recursively for any dict.txt
        if let Ok(entries) = std::fs::read_dir(model_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    if let Ok(dict_path) = Self::find_dict_file(&path) {
                        return Ok(dict_path);
                    }
                } else if path.file_name().is_some_and(|n| n == "dict.txt") {
                    return Ok(path);
                }
            }
        }

        Err(TaskError::ModelNotFound(format!(
            "Could not find character dictionary in {}",
            model_dir.display()
        )))
    }

    /// Load the character dictionary from a file.
    fn load_char_dict(path: &Path) -> TaskResult<Vec<String>> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| TaskError::Config(format!("Failed to read dict file: {e}")))?;

        // Each line is a character, first add blank token for CTC decoding
        let mut chars: Vec<String> = vec![String::new()]; // Blank token at index 0
        chars.extend(content.lines().map(std::string::ToString::to_string));
        chars.push(" ".to_string()); // Space token at end

        Ok(chars)
    }

    /// Preprocess image for detection model.
    /// PaddleOCR detection expects: [1, 3, H, W] normalized to [0, 1]
    fn preprocess_for_detection(pixel_values: &ArrayD<f32>) -> TaskResult<Array4<f32>> {
        // Input is [1, 3, H, W] from preprocessor, already normalized
        let shape = pixel_values.shape();
        if shape.len() != 4 || shape[1] != 3 {
            return Err(TaskError::InvalidInput(format!(
                "Expected [1, 3, H, W] input, got shape {shape:?}"
            )));
        }

        // Clone and convert to owned Array4
        let arr = pixel_values
            .to_owned()
            .into_dimensionality::<ndarray::Ix4>()
            .map_err(|e| TaskError::InvalidInput(format!("Failed to convert to 4D: {e}")))?;

        Ok(arr)
    }

    /// Run detection model to find text regions.
    async fn detect_text_regions(
        &self,
        pixel_values: Array4<f32>,
        original_h: usize,
        original_w: usize,
    ) -> TaskResult<Vec<[f32; 4]>> {
        use ort::session::input::SessionInputValue;

        let session = Arc::clone(&self.det_session);

        let result = tokio::task::spawn_blocking(move || {
            let mut session = session.blocking_lock();

            // Get input name
            let input_name = session.inputs()[0].name().to_string();

            // Create tensor
            let tensor =
                TensorRef::from_array_view(pixel_values.view().into_dyn()).map_err(|e| {
                    TaskError::Inference(format!("Failed to create detection tensor: {e}"))
                })?;

            let input_values: Vec<(&str, SessionInputValue)> =
                vec![(input_name.as_str(), tensor.into_dyn().into())];

            // Run detection
            let outputs = session
                .run(input_values)
                .map_err(|e| TaskError::Inference(format!("Detection failed: {e}")))?;

            // Get output (probability map)
            let output = outputs
                .values()
                .next()
                .ok_or_else(|| TaskError::Inference("No detection output".into()))?;

            let prob_map = output.try_extract_array::<f32>().map_err(|e| {
                TaskError::Inference(format!("Failed to extract detection output: {e}"))
            })?;

            // Extract bounding boxes from probability map
            let boxes = Self::extract_boxes_from_prob_map(&prob_map, original_h, original_w);

            Ok::<_, TaskError>(boxes)
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))??;

        Ok(result)
    }

    /// Extract bounding boxes from the detection probability map.
    fn extract_boxes_from_prob_map(
        prob_map: &ndarray::ArrayViewD<f32>,
        original_h: usize,
        original_w: usize,
    ) -> Vec<[f32; 4]> {
        let shape = prob_map.shape();
        // Shape is typically [1, 1, H, W]
        let (map_h, map_w) = if shape.len() == 4 {
            (shape[2], shape[3])
        } else if shape.len() == 3 {
            (shape[1], shape[2])
        } else {
            return vec![];
        };

        // Threshold the probability map
        let threshold = 0.3;
        let mut boxes = Vec::new();

        // Simple connected component analysis
        // For each row, find contiguous regions above threshold
        let prob_2d = if shape.len() == 4 {
            prob_map.slice(s![0, 0, .., ..]).to_owned()
        } else {
            prob_map.slice(s![0, .., ..]).to_owned()
        };

        // Find bounding box of all pixels above threshold
        let mut min_x = map_w;
        let mut max_x = 0;
        let mut min_y = map_h;
        let mut max_y = 0;
        let mut found = false;

        for y in 0..map_h {
            for x in 0..map_w {
                if prob_2d[[y, x]] > threshold {
                    found = true;
                    min_x = min_x.min(x);
                    max_x = max_x.max(x);
                    min_y = min_y.min(y);
                    max_y = max_y.max(y);
                }
            }
        }

        if found {
            // Scale back to original image size
            let scale_x = original_w as f32 / map_w as f32;
            let scale_y = original_h as f32 / map_h as f32;

            // Add some padding
            let pad = 5.0;
            let x1 = (min_x as f32 * scale_x - pad).max(0.0);
            let y1 = (min_y as f32 * scale_y - pad).max(0.0);
            let x2 = (max_x as f32 * scale_x + pad).min(original_w as f32);
            let y2 = (max_y as f32 * scale_y + pad).min(original_h as f32);

            boxes.push([x1, y1, x2, y2]);
        } else {
            // If no text detected, use the whole image
            boxes.push([0.0, 0.0, original_w as f32, original_h as f32]);
        }

        boxes
    }

    /// Preprocess a cropped region for recognition.
    /// PaddleOCR recognition expects: [1, 3, 48, W] where W depends on text width
    fn preprocess_for_recognition(
        pixel_values: &ArrayD<f32>,
        _bbox: &[f32; 4],
    ) -> TaskResult<Array4<f32>> {
        // For now, resize the entire image to recognition model input size
        // In a full implementation, we'd crop to the bbox first
        let shape = pixel_values.shape();
        if shape.len() != 4 {
            return Err(TaskError::InvalidInput(format!(
                "Expected 4D input, got shape {shape:?}"
            )));
        }

        // Recognition model typically expects height of 32 or 48
        // Width is variable but we'll use a fixed size for simplicity
        let target_h = 48;
        let target_w = 320;

        // Simple resize by sampling (bilinear would be better but more complex)
        let src_h = shape[2];
        let src_w = shape[3];

        let mut result = Array4::<f32>::zeros((1, 3, target_h, target_w));

        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let src_y = (y * src_h) / target_h;
                    let src_x = (x * src_w) / target_w;
                    result[[0, c, y, x]] = pixel_values[[0, c, src_y, src_x]];
                }
            }
        }

        Ok(result)
    }

    /// Run recognition model on a cropped text region.
    async fn recognize_text(&self, pixel_values: Array4<f32>) -> TaskResult<(String, f32)> {
        use ort::session::input::SessionInputValue;

        let session = Arc::clone(&self.rec_session);
        let char_dict = self.char_dict.clone();

        let result = tokio::task::spawn_blocking(move || {
            let mut session = session.blocking_lock();

            // Get input name
            let input_name = session.inputs()[0].name().to_string();

            // Create tensor
            let tensor =
                TensorRef::from_array_view(pixel_values.view().into_dyn()).map_err(|e| {
                    TaskError::Inference(format!("Failed to create recognition tensor: {e}"))
                })?;

            let input_values: Vec<(&str, SessionInputValue)> =
                vec![(input_name.as_str(), tensor.into_dyn().into())];

            // Run recognition
            let outputs = session
                .run(input_values)
                .map_err(|e| TaskError::Inference(format!("Recognition failed: {e}")))?;

            // Get output logits
            let output = outputs
                .values()
                .next()
                .ok_or_else(|| TaskError::Inference("No recognition output".into()))?;

            let logits = output.try_extract_array::<f32>().map_err(|e| {
                TaskError::Inference(format!("Failed to extract recognition output: {e}"))
            })?;

            // Decode using CTC decoding
            let (text, confidence) = Self::ctc_decode(&logits.view(), &char_dict);

            Ok::<_, TaskError>((text, confidence))
        })
        .await
        .map_err(|e| TaskError::Inference(format!("spawn_blocking failed: {e}")))??;

        Ok(result)
    }

    /// CTC decode the recognition output.
    /// Takes argmax at each timestep, removes blanks and repeated characters.
    fn ctc_decode(logits: &ndarray::ArrayViewD<f32>, char_dict: &[String]) -> (String, f32) {
        let shape = logits.shape();
        // Shape is typically [1, T, num_classes] or [T, num_classes]
        let (seq_len, num_classes) = if shape.len() == 3 {
            (shape[1], shape[2])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return (String::new(), 0.0);
        };

        let mut text = String::new();
        let mut total_confidence = 0.0;
        let mut num_chars = 0;
        let mut prev_idx = 0usize;

        for t in 0..seq_len {
            // Get logits for this timestep
            let timestep_logits: Vec<f32> = if shape.len() == 3 {
                (0..num_classes).map(|c| logits[[0, t, c]]).collect()
            } else {
                (0..num_classes).map(|c| logits[[t, c]]).collect()
            };

            // Softmax
            let max_val = timestep_logits
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let exp_sum: f32 = timestep_logits.iter().map(|&x| (x - max_val).exp()).sum();
            let probs: Vec<f32> = timestep_logits
                .iter()
                .map(|&x| (x - max_val).exp() / exp_sum)
                .collect();

            // Argmax
            let (max_idx, &max_prob) = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or((0, &0.0));

            // CTC decoding: skip blank (index 0) and repeated characters
            if max_idx != 0 && max_idx != prev_idx && max_idx < char_dict.len() {
                text.push_str(&char_dict[max_idx]);
                total_confidence += max_prob;
                num_chars += 1;
            }
            prev_idx = max_idx;
        }

        let avg_confidence = if num_chars > 0 {
            total_confidence / num_chars as f32
        } else {
            0.0
        };

        (text, avg_confidence)
    }
}

#[async_trait]
impl Task for PaddleOcrTask {
    fn name(&self) -> &str {
        &self.name
    }

    #[cfg(feature = "preprocess")]
    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id = request_id, "PaddleOCR task executing");

        // Parse input
        let input: PaddleOcrInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input: {e}")),
        };

        debug!("Processing OCR request");

        // 1. Preprocess image using the standard preprocessor
        let image_input = format!(r#"{{"image": "{}"}}"#, input.image);
        let preprocessed = match self.preprocessor.process_json(&image_input).await {
            Ok(p) => p,
            Err(e) => return GrpcTaskResult::err(format!("Image preprocessing failed: {e}")),
        };

        // Extract pixel_values
        let pixel_values = match preprocessed.inputs.get("pixel_values") {
            Some(v) => match crate::tensor_utils::json_to_array_f32(v) {
                Ok(arr) => arr,
                Err(e) => return GrpcTaskResult::err(format!("Invalid pixel_values: {e}")),
            },
            None => {
                return GrpcTaskResult::err("No pixel_values in preprocessed output".to_string())
            }
        };

        // Get original image dimensions from preprocessing metadata or estimate
        let shape = pixel_values.shape();
        let (original_h, original_w) = if shape.len() == 4 {
            (shape[2], shape[3])
        } else {
            (224, 224) // fallback
        };

        // 2. Preprocess for detection
        let det_input = match Self::preprocess_for_detection(&pixel_values) {
            Ok(arr) => arr,
            Err(e) => return GrpcTaskResult::err(format!("Detection preprocessing failed: {e}")),
        };

        // 3. Run detection to find text regions
        let boxes = match self
            .detect_text_regions(det_input, original_h, original_w)
            .await
        {
            Ok(b) => b,
            Err(e) => {
                warn!("Detection failed, using full image: {e}");
                vec![[0.0, 0.0, original_w as f32, original_h as f32]]
            }
        };

        debug!(num_boxes = boxes.len(), "Detected text regions");

        // 4. Run recognition on each detected region
        let mut text_boxes = Vec::new();
        let mut all_text = String::new();

        for bbox in &boxes {
            // Preprocess cropped region for recognition
            let rec_input = match Self::preprocess_for_recognition(&pixel_values, bbox) {
                Ok(arr) => arr,
                Err(e) => {
                    warn!("Recognition preprocessing failed for box {:?}: {e}", bbox);
                    continue;
                }
            };

            // Run recognition
            match self.recognize_text(rec_input).await {
                Ok((text, confidence)) => {
                    if !text.is_empty() {
                        if !all_text.is_empty() {
                            all_text.push(' ');
                        }
                        all_text.push_str(&text);

                        text_boxes.push(TextBox {
                            bbox: *bbox,
                            text,
                            confidence,
                        });
                    }
                }
                Err(e) => {
                    warn!("Recognition failed for box {:?}: {e}", bbox);
                }
            }
        }

        let output = PaddleOcrOutput {
            text: all_text,
            num_boxes: text_boxes.len(),
            boxes: text_boxes,
        };

        match serde_json::to_string(&output) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("Serialization error: {e}")),
        }
    }

    fn is_ready(&self) -> bool {
        true
    }
}
