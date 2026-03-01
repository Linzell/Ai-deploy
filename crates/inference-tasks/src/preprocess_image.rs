//! Image preprocessing utilities.
//!
//! Requires the `image` feature flag.

use base64::{engine::general_purpose::STANDARD, Engine};
use image::{DynamicImage, GenericImageView};

use crate::error::{TaskError, TaskResult};

/// Decode base64 image data.
pub fn decode_base64_image(data: &str) -> TaskResult<DynamicImage> {
    let bytes = STANDARD
        .decode(data)
        .map_err(|e| TaskError::InvalidInput(format!("Invalid base64: {}", e)))?;

    image::load_from_memory(&bytes)
        .map_err(|e| TaskError::InvalidInput(format!("Invalid image: {}", e)))
}

/// Resize image to target dimensions.
pub fn resize_image(img: &DynamicImage, width: u32, height: u32) -> DynamicImage {
    img.resize_exact(width, height, image::imageops::FilterType::Triangle)
}

/// Convert image to RGB f32 tensor data (CHW format).
pub fn image_to_chw_f32(img: &DynamicImage) -> Vec<f32> {
    let (width, height) = img.dimensions();
    let rgb = img.to_rgb8();

    let mut data = vec![0.0f32; 3 * (width as usize) * (height as usize)];

    for (x, y, pixel) in rgb.enumerate_pixels() {
        let idx = (y as usize) * (width as usize) + (x as usize);
        data[idx] = pixel[0] as f32 / 255.0; // R
        data[idx + (width * height) as usize] = pixel[1] as f32 / 255.0; // G
        data[idx + 2 * (width * height) as usize] = pixel[2] as f32 / 255.0; // B
    }

    data
}

/// Normalize image data with mean and std.
pub fn normalize(data: &mut [f32], mean: [f32; 3], std: [f32; 3], width: usize, height: usize) {
    let channel_size = width * height;

    for c in 0..3 {
        for i in 0..channel_size {
            data[c * channel_size + i] = (data[c * channel_size + i] - mean[c]) / std[c];
        }
    }
}
