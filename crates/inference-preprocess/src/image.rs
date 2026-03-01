//! Image preprocessing for vision models.
//!
//! Handles:
//! - Loading from S3, base64, local files
//! - Resizing to model input size
//! - Normalization (ImageNet mean/std or custom)
//! - Converting to tensor format [batch, channels, height, width]

use crate::error::{PreprocessError, PreprocessResult};
use image::DynamicImage;
use ndarray::{Array4, Axis};
use std::path::Path;
use tracing::debug;

/// Default ImageNet normalization values.
pub const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// CLIP normalization values.
pub const CLIP_MEAN: [f32; 3] = [0.481_454_66, 0.457_827_5, 0.408_210_73];
pub const CLIP_STD: [f32; 3] = [0.268_629_54, 0.261_302_6, 0.275_777_1];

/// Processed image ready for ONNX inference.
#[derive(Debug, Clone)]
pub struct ImageOutput {
    /// Pixel values [batch_size, channels, height, width]
    pub pixel_values: Array4<f32>,
    /// Original image dimensions (width, height)
    pub original_size: (u32, u32),
    /// Pixel mask for models like DETR [batch_size, height, width]
    /// 1 = valid pixel, 0 = padding (for variable-size batching)
    pub pixel_mask: Option<ndarray::Array3<i64>>,
}

impl ImageOutput {
    /// Convert to JSON-compatible format for ONNX input.
    pub fn to_json_inputs(&self) -> serde_json::Value {
        // Convert ndarray to nested Vec for JSON serialization
        let shape = self.pixel_values.shape();
        let mut result = Vec::with_capacity(shape[0]);

        for b in 0..shape[0] {
            let mut channels = Vec::with_capacity(shape[1]);
            for c in 0..shape[1] {
                let mut rows = Vec::with_capacity(shape[2]);
                for h in 0..shape[2] {
                    let row: Vec<f32> = (0..shape[3])
                        .map(|w| self.pixel_values[[b, c, h, w]])
                        .collect();
                    rows.push(row);
                }
                channels.push(rows);
            }
            result.push(channels);
        }

        let mut json = serde_json::json!({
            "pixel_values": result
        });

        // Add pixel_mask if present (for DETR-style models)
        if let Some(ref mask) = self.pixel_mask {
            let mask_shape = mask.shape();
            let mut mask_result = Vec::with_capacity(mask_shape[0]);
            for b in 0..mask_shape[0] {
                let mut rows = Vec::with_capacity(mask_shape[1]);
                for h in 0..mask_shape[1] {
                    let row: Vec<i64> = (0..mask_shape[2]).map(|w| mask[[b, h, w]]).collect();
                    rows.push(row);
                }
                mask_result.push(rows);
            }
            json["pixel_mask"] = serde_json::json!(mask_result);
        }

        json
    }
}

/// Image preprocessing configuration.
#[derive(Debug, Clone)]
pub struct ImageConfig {
    /// Target image size (width, height)
    pub size: (u32, u32),
    /// Normalization mean (RGB)
    pub mean: [f32; 3],
    /// Normalization std (RGB)
    pub std: [f32; 3],
    /// Whether to rescale pixel values to [0, 1] before normalization
    pub rescale: bool,
    /// Resampling filter for resizing
    pub filter: image::imageops::FilterType,
    /// Whether to generate pixel_mask (for DETR-style models)
    pub generate_pixel_mask: bool,
    /// Size of the pixel_mask (if different from image size).
    /// DETR models downsample the mask (e.g., 800x800 image -> 64x64 mask).
    /// If None, uses the same size as the image.
    pub pixel_mask_size: Option<(usize, usize)>,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            size: (224, 224),
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: false,
            pixel_mask_size: None,
        }
    }
}

impl ImageConfig {
    /// Create config for CLIP-style models.
    pub fn clip() -> Self {
        Self {
            size: (224, 224),
            mean: CLIP_MEAN,
            std: CLIP_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: false,
            pixel_mask_size: None,
        }
    }

    /// Create config for ViT-style models with custom size.
    pub fn vit(size: u32) -> Self {
        Self {
            size: (size, size),
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: false,
            pixel_mask_size: None,
        }
    }

    /// Create config for Florence-2 (768x768).
    pub fn florence2() -> Self {
        Self {
            size: (768, 768),
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: false,
            pixel_mask_size: None,
        }
    }

    /// Create config for DETR-style object detection models.
    /// These models require a pixel_mask input that is downsampled from the image size.
    /// DETR uses 800x800 images with 64x64 masks (downsampling factor of ~12.5).
    pub fn detr() -> Self {
        Self {
            size: (800, 800), // DETR default
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: true,
            pixel_mask_size: Some((64, 64)), // DETR expects downsampled mask
        }
    }

    /// Create config for TrOCR models (384x384 with 0.5 mean/std normalization).
    ///
    /// TrOCR uses a different normalization than ImageNet:
    /// - Image size: 384x384
    /// - Normalization: mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
    ///
    /// This effectively maps pixel values from [0, 1] to [-1, 1].
    pub fn trocr() -> Self {
        Self {
            size: (384, 384),
            mean: [0.5, 0.5, 0.5],
            std: [0.5, 0.5, 0.5],
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: false,
            pixel_mask_size: None,
        }
    }

    /// Create config for ViLT (Visual-Language Transformer) models.
    ///
    /// ViLT VQA models expect:
    /// - Image size: 384x384
    /// - ImageNet normalization (mean/std)
    /// - Optional pixel_mask at same resolution (for padding handling)
    pub fn vilt() -> Self {
        Self {
            size: (384, 384),
            mean: IMAGENET_MEAN,
            std: IMAGENET_STD,
            rescale: true,
            filter: image::imageops::FilterType::Lanczos3,
            generate_pixel_mask: true,
            pixel_mask_size: None, // Same size as image
        }
    }
}

/// Image processor.
pub struct ImageProcessor {
    config: ImageConfig,
}

impl ImageProcessor {
    /// Create a new image processor with default config.
    pub fn new() -> Self {
        Self {
            config: ImageConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: ImageConfig) -> Self {
        Self { config }
    }

    /// Load and process an image from bytes.
    pub fn process_bytes(&self, bytes: &[u8]) -> PreprocessResult<ImageOutput> {
        let img = image::load_from_memory(bytes)
            .map_err(|e| PreprocessError::Image(format!("Failed to decode image: {e}")))?;

        Ok(self.process_image(&img))
    }

    /// Load and process an image from a file path.
    pub fn process_file(&self, path: impl AsRef<Path>) -> PreprocessResult<ImageOutput> {
        let path = path.as_ref();
        debug!(path = %path.display(), "Loading image from file");

        let img = image::open(path)
            .map_err(|e| PreprocessError::Image(format!("Failed to load image: {e}")))?;

        Ok(self.process_image(&img))
    }

    /// Load and process an image from base64.
    pub fn process_base64(&self, data: &str) -> PreprocessResult<ImageOutput> {
        debug!(len = data.len(), "Decoding base64 image");

        // Strip data URL prefix if present
        let data = if data.starts_with("data:") {
            data.split(',').nth(1).unwrap_or(data)
        } else {
            data
        };

        let bytes = base64::Engine::decode(&base64::engine::general_purpose::STANDARD, data)
            .map_err(|e| PreprocessError::Image(format!("Invalid base64: {e}")))?;

        self.process_bytes(&bytes)
    }

    /// Process a batch of images.
    pub fn process_batch(&self, images: &[DynamicImage]) -> PreprocessResult<ImageOutput> {
        if images.is_empty() {
            return Err(PreprocessError::InvalidInput("Empty image batch".into()));
        }

        let mut batch_pixels = Vec::with_capacity(images.len());
        let mut batch_masks = Vec::with_capacity(images.len());
        let mut original_size = (0, 0);
        let mut has_masks = false;

        for (i, img) in images.iter().enumerate() {
            if i == 0 {
                original_size = (img.width(), img.height());
            }
            let output = self.process_image(img);
            batch_pixels.push(output.pixel_values);
            if let Some(mask) = output.pixel_mask {
                batch_masks.push(mask);
                has_masks = true;
            }
        }

        // Stack along batch dimension
        let stacked = ndarray::concatenate(
            Axis(0),
            &batch_pixels.iter().map(|a| a.view()).collect::<Vec<_>>(),
        )
        .map_err(|e| PreprocessError::Image(format!("Failed to stack batch: {e}")))?;

        // Stack masks if present
        let pixel_mask = if has_masks && batch_masks.len() == images.len() {
            Some(
                ndarray::concatenate(
                    Axis(0),
                    &batch_masks.iter().map(|a| a.view()).collect::<Vec<_>>(),
                )
                .map_err(|e| PreprocessError::Image(format!("Failed to stack masks: {e}")))?,
            )
        } else {
            None
        };

        Ok(ImageOutput {
            pixel_values: stacked,
            original_size,
            pixel_mask,
        })
    }

    /// Process a single image.
    fn process_image(&self, img: &DynamicImage) -> ImageOutput {
        let original_size = (img.width(), img.height());
        debug!(
            original_size = ?original_size,
            target_size = ?self.config.size,
            "Processing image"
        );

        // Resize
        let (target_w, target_h) = self.config.size;
        let resized = img.resize_exact(target_w, target_h, self.config.filter);

        // Convert to RGB
        let rgb = resized.to_rgb8();

        // Create tensor [1, 3, H, W]
        let (width, height) = (target_w as usize, target_h as usize);
        let mut pixels = Array4::<f32>::zeros((1, 3, height, width));

        // Process pixels using contiguous memory access for better cache performance.
        // Instead of random-access indexing, we iterate over the raw pixel buffer
        // which is stored in row-major order (y, x, channel).
        let raw_pixels = rgb.as_raw();

        // Get mutable slices for each channel to enable direct memory writes
        let (r_slice, rest) = pixels
            .as_slice_mut()
            .expect("Array4 should be contiguous")
            .split_at_mut(height * width);
        let (g_slice, b_slice) = rest.split_at_mut(height * width);

        let (mean, std) = (self.config.mean, self.config.std);
        let rescale = self.config.rescale;

        for y in 0..height {
            for x in 0..width {
                let pixel_idx = (y * width + x) * 3;
                let r = raw_pixels[pixel_idx];
                let g = raw_pixels[pixel_idx + 1];
                let b = raw_pixels[pixel_idx + 2];

                // Scale to [0, 1] if rescale is enabled, then normalize
                let (r, g, b) = if rescale {
                    (
                        (f32::from(r) / 255.0 - mean[0]) / std[0],
                        (f32::from(g) / 255.0 - mean[1]) / std[1],
                        (f32::from(b) / 255.0 - mean[2]) / std[2],
                    )
                } else {
                    (
                        (f32::from(r) - mean[0]) / std[0],
                        (f32::from(g) - mean[1]) / std[1],
                        (f32::from(b) - mean[2]) / std[2],
                    )
                };

                let output_idx = y * width + x;
                r_slice[output_idx] = r;
                g_slice[output_idx] = g;
                b_slice[output_idx] = b;
            }
        }

        // Generate pixel_mask if configured (for DETR-style models)
        let pixel_mask = if self.config.generate_pixel_mask {
            // Use configured mask size or fall back to image size
            let (mask_h, mask_w) = self.config.pixel_mask_size.unwrap_or((height, width));
            // For single images resized to exact size, all pixels are valid
            Some(ndarray::Array3::<i64>::ones((1, mask_h, mask_w)))
        } else {
            None
        };

        ImageOutput {
            pixel_values: pixels,
            original_size,
            pixel_mask,
        }
    }
}

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for ImageProcessor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ImageProcessor")
            .field("config", &self.config)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ImageConfig::default();
        assert_eq!(config.size, (224, 224));
        assert!(config.rescale);
    }

    #[test]
    fn test_clip_config() {
        let config = ImageConfig::clip();
        // Compare element-wise with epsilon for float arrays
        for (actual, expected) in config.mean.iter().zip(CLIP_MEAN.iter()) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
        for (actual, expected) in config.std.iter().zip(CLIP_STD.iter()) {
            assert!((actual - expected).abs() < f32::EPSILON);
        }
    }
}
