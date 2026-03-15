//! Object detection via DETR / Table-Transformer.
//!
//! Implements the `Task` trait for DETR-family object detection models:
//! - **DETR** (`facebook/detr-resnet-50`): COCO 91 classes, ResNet-50 backbone
//! - **Table-Transformer** (`microsoft/table-transformer-detection`): ResNet-18 backbone
//!
//! ## Input format
//!
//! Base64-encoded image (JPEG/PNG):
//! ```json
//! {"inputs": "<base64-encoded image>"}
//! ```
//!
//! ## Output format
//!
//! Returns detected objects with labels, scores, and bounding boxes:
//! ```json
//! [{"label": "cat", "score": 0.98, "box": {"xmin": 10, "ymin": 20, "xmax": 100, "ymax": 200}}]
//! ```

pub mod config;
pub mod detr;
pub mod position_encoding;
pub mod resnet;
pub mod transformer;

use async_trait::async_trait;
use candle_core::{DType, Device, Tensor};
use inference_core::task::{Task, TaskResult as GrpcTaskResult};
use inference_core::Config;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

use crate::error::{TaskError, TaskResult};
use crate::utils;
use config::DetrConfig;
use detr::DetrForObjectDetection;

// ---------------------------------------------------------------------------
// I/O types
// ---------------------------------------------------------------------------

/// Input for object detection.
#[derive(Debug, Deserialize)]
pub struct ObjectDetectionInput {
    /// Base64-encoded image (JPEG/PNG) or data URI
    pub inputs: Option<serde_json::Value>,
    /// Alternative field name
    pub image: Option<serde_json::Value>,
    /// Confidence threshold (default 0.5)
    pub threshold: Option<f32>,
}

/// A single detected object in HF Inference API format.
#[derive(Debug, Serialize)]
pub struct DetectedObjectOutput {
    pub label: String,
    pub score: f32,
    #[serde(rename = "box")]
    pub bbox: BoundingBox,
}

/// Bounding box in pixel coordinates.
#[derive(Debug, Serialize)]
pub struct BoundingBox {
    pub xmin: i32,
    pub ymin: i32,
    pub xmax: i32,
    pub ymax: i32,
}

// ---------------------------------------------------------------------------
// Task struct
// ---------------------------------------------------------------------------

/// Candle-based object detection task (DETR / Table-Transformer).
pub struct CandleObjectDetectionTask {
    name: String,
    model: DetrForObjectDetection,
    device: Device,
    #[allow(dead_code)]
    dtype: DType,
    id2label: HashMap<usize, String>,
    /// Input image size used during preprocessing
    image_size: usize,
}

impl CandleObjectDetectionTask {
    /// Create from a model directory containing config.json and safetensors.
    pub fn from_model_dir(
        model_dir: impl AsRef<Path>,
        name: impl Into<String>,
        config: &Config,
    ) -> TaskResult<Self> {
        let model_dir = model_dir.as_ref();
        let name = name.into();

        info!(
            model_dir = %model_dir.display(),
            task_name = %name,
            "Loading candle object detection model"
        );

        let device = utils::resolve_device(&config.device)?;
        info!(device = ?device, "Using device");

        // Parse config.json
        let config_path = model_dir.join("config.json");
        let config_content = std::fs::read_to_string(&config_path)
            .map_err(|e| TaskError::ModelLoad(format!("Cannot read config.json: {e}")))?;
        let model_json: serde_json::Value = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Invalid config.json: {e}")))?;

        let model_type = model_json
            .get("model_type")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown");

        match model_type {
            "detr" | "table-transformer" => {}
            other => {
                return Err(TaskError::ModelLoad(format!(
                    "Unsupported object detection architecture: '{other}'. \
                     Supported: detr, table-transformer."
                )));
            }
        }

        let detr_config: DetrConfig = serde_json::from_str(&config_content)
            .map_err(|e| TaskError::ModelLoad(format!("Failed to parse DetrConfig: {e}")))?;

        // Extract id2label
        let id2label = model_json
            .get("id2label")
            .and_then(|v| v.as_object())
            .map(|obj| {
                obj.iter()
                    .filter_map(|(k, v)| {
                        let idx: usize = k.parse().ok()?;
                        let label = v.as_str()?.to_string();
                        Some((idx, label))
                    })
                    .collect::<HashMap<usize, String>>()
            })
            .unwrap_or_default();

        let num_labels = id2label.len();

        info!(
            model_type,
            backbone = %detr_config.backbone,
            d_model = detr_config.d_model,
            encoder_layers = detr_config.encoder_layers,
            decoder_layers = detr_config.decoder_layers,
            num_queries = detr_config.num_queries,
            num_labels,
            "Object detection model configuration"
        );

        // Load weights
        let weight_files = utils::find_weight_files(model_dir)?;
        info!(num_files = weight_files.len(), "Found weight files");

        let model_dtype = utils::read_model_dtype(&config_path);
        let dtype = match device {
            Device::Cpu => DType::F32,
            _ => model_dtype.unwrap_or(DType::F32),
        };
        info!(dtype = ?dtype, "Compute dtype");

        let vb = utils::load_safetensors_safe(&weight_files, dtype, &device)?;

        let model = if num_labels > 0 {
            DetrForObjectDetection::load_with_num_labels(&vb, &detr_config, num_labels)?
        } else {
            DetrForObjectDetection::load(&vb, &detr_config)?
        };
        info!("DETR model loaded");

        Ok(Self {
            name,
            model,
            device,
            dtype,
            id2label,
            image_size: 800, // DETR default
        })
    }

    /// Parse input image from base64 or data URI, return pixel tensor [1, 3, H, W].
    fn parse_image(&self, input: &ObjectDetectionInput) -> TaskResult<Tensor> {
        let value = input
            .inputs
            .as_ref()
            .or(input.image.as_ref())
            .ok_or_else(|| {
                TaskError::InvalidInput("Missing 'inputs' or 'image' field in request".into())
            })?;

        let b64_str = match value {
            serde_json::Value::String(s) => {
                // Strip data URI prefix if present
                if let Some(pos) = s.find(";base64,") {
                    s[pos + 8..].to_string()
                } else {
                    s.clone()
                }
            }
            _ => {
                return Err(TaskError::InvalidInput(
                    "Expected base64-encoded image string".into(),
                ))
            }
        };

        let image_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &b64_str,
        )
        .map_err(|e| TaskError::InvalidInput(format!("Invalid base64: {e}")))?;

        // Decode image and resize
        let img = image::load_from_memory(&image_bytes)
            .map_err(|e| TaskError::InvalidInput(format!("Invalid image: {e}")))?;

        let (orig_w, orig_h) = (img.width(), img.height());
        debug!(orig_w, orig_h, target_size = self.image_size, "Image loaded");

        // Resize to target size (maintaining aspect ratio with padding would be better,
        // but DETR in HF typically just resizes to a square)
        let img = img.resize_exact(
            self.image_size as u32,
            self.image_size as u32,
            image::imageops::FilterType::Triangle,
        );

        // Convert to tensor [1, 3, H, W] with ImageNet normalization
        let rgb = img.to_rgb8();
        let (w, h) = (rgb.width() as usize, rgb.height() as usize);
        let pixels = rgb.as_raw();

        // HWC → CHW, normalize with ImageNet mean/std
        let mean = [0.485f32, 0.456, 0.406];
        let std = [0.229f32, 0.224, 0.225];

        let mut data = vec![0f32; 3 * h * w];
        for y in 0..h {
            for x in 0..w {
                let idx = (y * w + x) * 3;
                for c in 0..3 {
                    let pixel = f32::from(pixels[idx + c]) / 255.0;
                    data[c * h * w + y * w + x] = (pixel - mean[c]) / std[c];
                }
            }
        }

        Tensor::from_vec(data, (1, 3, h, w), &self.device)
            .map_err(|e| TaskError::Inference(format!("Failed to create image tensor: {e}")))
    }

    /// Post-process model outputs into detected objects.
    ///
    /// DETR outputs:
    /// - `logits`: [1, num_queries, num_classes+1] — last class = "no object"
    /// - `pred_boxes`: [1, num_queries, 4] — normalized (cx, cy, w, h)
    fn postprocess(
        &self,
        logits: &Tensor,
        pred_boxes: &Tensor,
        threshold: f32,
        img_width: u32,
        img_height: u32,
    ) -> TaskResult<Vec<DetectedObjectOutput>> {
        // Squeeze batch dim: [num_queries, num_classes+1]
        let logits = logits.squeeze(0).map_err(|e| {
            TaskError::Inference(format!("logits squeeze: {e}"))
        })?;
        let boxes = pred_boxes.squeeze(0).map_err(|e| {
            TaskError::Inference(format!("boxes squeeze: {e}"))
        })?;

        // Softmax over classes
        let probs = candle_nn::ops::softmax(&logits, 1).map_err(|e| {
            TaskError::Inference(format!("softmax: {e}"))
        })?;

        let num_queries = probs.dim(0).map_err(|e| {
            TaskError::Inference(format!("dim: {e}"))
        })?;
        let num_classes_plus_one = probs.dim(1).map_err(|e| {
            TaskError::Inference(format!("dim: {e}"))
        })?;
        let num_classes = num_classes_plus_one - 1; // last class = "no object"

        let probs_vec: Vec<f32> = probs
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(|e| TaskError::Inference(format!("probs to vec: {e}")))?;

        let boxes_vec: Vec<f32> = boxes
            .to_dtype(DType::F32)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(|e| TaskError::Inference(format!("boxes to vec: {e}")))?;

        let mut results = Vec::new();

        for q in 0..num_queries {
            // Find best class (excluding "no object" at index num_classes)
            let mut best_score = 0.0f32;
            let mut best_class = 0usize;
            for c in 0..num_classes {
                let score = probs_vec[q * num_classes_plus_one + c];
                if score > best_score {
                    best_score = score;
                    best_class = c;
                }
            }

            if best_score < threshold {
                continue;
            }

            // Convert center-format boxes to corner-format
            let cx = boxes_vec[q * 4];
            let cy = boxes_vec[q * 4 + 1];
            let w = boxes_vec[q * 4 + 2];
            let h = boxes_vec[q * 4 + 3];

            let xmin = ((cx - w / 2.0) * img_width as f32).round() as i32;
            let ymin = ((cy - h / 2.0) * img_height as f32).round() as i32;
            let xmax = ((cx + w / 2.0) * img_width as f32).round() as i32;
            let ymax = ((cy + h / 2.0) * img_height as f32).round() as i32;

            let label = self
                .id2label
                .get(&best_class)
                .cloned()
                .unwrap_or_else(|| format!("class_{best_class}"));

            results.push(DetectedObjectOutput {
                label,
                score: best_score,
                bbox: BoundingBox {
                    xmin: xmin.max(0),
                    ymin: ymin.max(0),
                    xmax: xmax.min(img_width.cast_signed()),
                    ymax: ymax.min(img_height.cast_signed()),
                },
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }
}

#[async_trait]
impl Task for CandleObjectDetectionTask {
    fn name(&self) -> &str {
        &self.name
    }

    async fn execute(&self, payload: &str, request_id: &str) -> GrpcTaskResult {
        debug!(request_id, "Object detection execute");

        let input: ObjectDetectionInput = match serde_json::from_str(payload) {
            Ok(i) => i,
            Err(e) => return GrpcTaskResult::err(format!("Invalid input JSON: {e}")),
        };

        let threshold = input.threshold.unwrap_or(0.5);

        // Get original image dimensions before preprocessing
        let (orig_w, orig_h) = match Self::get_image_dims(&input) {
            Ok(dims) => dims,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        // Parse and preprocess image
        let pixel_values = match self.parse_image(&input) {
            Ok(t) => t,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        // Forward pass
        let (logits, pred_boxes) = match self.model.forward(&pixel_values) {
            Ok(r) => r,
            Err(e) => return GrpcTaskResult::err(format!("Model forward pass failed: {e}")),
        };

        // Post-process
        let detections = match self.postprocess(&logits, &pred_boxes, threshold, orig_w, orig_h) {
            Ok(d) => d,
            Err(e) => return GrpcTaskResult::err(e.to_string()),
        };

        debug!(num_detections = detections.len(), "Object detection complete");

        match serde_json::to_string(&detections) {
            Ok(json) => GrpcTaskResult::ok(json),
            Err(e) => GrpcTaskResult::err(format!("JSON serialization failed: {e}")),
        }
    }

    fn supports_streaming(&self) -> bool {
        false
    }
}

impl CandleObjectDetectionTask {
    /// Get original image dimensions without full decode.
    fn get_image_dims(input: &ObjectDetectionInput) -> TaskResult<(u32, u32)> {
        let value = input
            .inputs
            .as_ref()
            .or(input.image.as_ref())
            .ok_or_else(|| {
                TaskError::InvalidInput("Missing 'inputs' or 'image' field".into())
            })?;

        let b64_str = match value {
            serde_json::Value::String(s) => {
                if let Some(pos) = s.find(";base64,") {
                    s[pos + 8..].to_string()
                } else {
                    s.clone()
                }
            }
            _ => {
                return Err(TaskError::InvalidInput(
                    "Expected base64-encoded image string".into(),
                ))
            }
        };

        let image_bytes = base64::Engine::decode(
            &base64::engine::general_purpose::STANDARD,
            &b64_str,
        )
        .map_err(|e| TaskError::InvalidInput(format!("Invalid base64: {e}")))?;

        let img = image::load_from_memory(&image_bytes)
            .map_err(|e| TaskError::InvalidInput(format!("Invalid image: {e}")))?;

        Ok((img.width(), img.height()))
    }
}
