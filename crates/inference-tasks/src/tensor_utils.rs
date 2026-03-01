//! Tensor conversion utilities.
//!
//! This module provides shared utilities for converting between JSON values
//! and ndarray tensors. It supports:
//! - JSON to ndarray conversion (i64 and f32)
//! - ndarray to JSON conversion
//! - Shape inference from nested JSON arrays
//! - Type detection (float vs integer)

use ndarray::{ArrayD, IxDyn};
use serde_json::Value;

use crate::error::{TaskError, TaskResult};

/// Represents a tensor value that can be either i64 or f32.
///
/// This is used for mixed-type tensor inputs where the type
/// is determined at runtime based on the JSON content.
pub enum TensorValue {
    Int64(ArrayD<i64>),
    Float32(ArrayD<f32>),
}

impl TensorValue {
    /// Returns true if this is an i64 tensor.
    pub fn is_int64(&self) -> bool {
        matches!(self, Self::Int64(_))
    }

    /// Returns true if this is an f32 tensor.
    pub fn is_float32(&self) -> bool {
        matches!(self, Self::Float32(_))
    }

    /// Get the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Int64(arr) => arr.shape(),
            Self::Float32(arr) => arr.shape(),
        }
    }
}

/// Detect if a JSON value contains any floats (decimal points or scientific notation).
///
/// This recursively checks arrays for any floating-point numbers.
///
/// # Example
///
/// ```rust,ignore
/// assert!(!contains_floats(&json!([1, 2, 3])));
/// assert!(contains_floats(&json!([1.0, 2.0, 3.0])));
/// ```
pub fn contains_floats(value: &Value) -> bool {
    match value {
        Value::Array(arr) => arr.iter().any(contains_floats),
        Value::Number(n) => n.is_f64() && !n.is_i64(),
        _ => false,
    }
}

/// Infer shape from nested JSON array.
///
/// Traverses the JSON structure to determine the dimensions.
/// Assumes the array is rectangular (all sub-arrays have same length).
///
/// # Example
///
/// ```rust,ignore
/// let value = json!([[1, 2, 3], [4, 5, 6]]);
/// assert_eq!(infer_shape(&value)?, vec![2, 3]);
/// ```
pub fn infer_shape(value: &Value) -> TaskResult<Vec<usize>> {
    let mut shape = Vec::new();
    let mut current = value;

    while let Value::Array(arr) = current {
        shape.push(arr.len());
        if let Some(first) = arr.first() {
            current = first;
        } else {
            break;
        }
    }

    Ok(shape)
}

/// Parse JSON value to i64 dynamic ndarray.
///
/// Recursively extracts integer values from nested JSON arrays
/// and constructs an ndarray with the inferred shape.
///
/// # Errors
///
/// Returns `TaskError::InvalidInput` if:
/// - The value contains non-integer numbers
/// - The shape doesn't match the data
pub fn json_to_array_i64(value: &Value) -> TaskResult<ArrayD<i64>> {
    fn get_data(v: &Value, data: &mut Vec<i64>) -> TaskResult<()> {
        match v {
            Value::Array(arr) => {
                for item in arr {
                    get_data(item, data)?;
                }
            }
            Value::Number(n) => {
                let i = n.as_i64().ok_or_else(|| {
                    TaskError::InvalidInput(format!("Expected integer, got: {n}"))
                })?;
                data.push(i);
            }
            _ => {
                return Err(TaskError::InvalidInput("Expected array or number".into()));
            }
        }
        Ok(())
    }

    let shape = infer_shape(value)?;
    let mut data = Vec::new();
    get_data(value, &mut data)?;

    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| TaskError::InvalidInput(format!("Shape mismatch: {e}")))
}

/// Parse JSON value to f32 dynamic ndarray.
///
/// Recursively extracts float values from nested JSON arrays
/// and constructs an ndarray with the inferred shape.
///
/// # Errors
///
/// Returns `TaskError::InvalidInput` if:
/// - The value contains non-numeric types
/// - The shape doesn't match the data
#[allow(clippy::cast_possible_truncation)]
pub fn json_to_array_f32(value: &Value) -> TaskResult<ArrayD<f32>> {
    fn get_data(v: &Value, data: &mut Vec<f32>) -> TaskResult<()> {
        match v {
            Value::Array(arr) => {
                for item in arr {
                    get_data(item, data)?;
                }
            }
            Value::Number(n) => {
                #[allow(clippy::cast_possible_truncation)]
                let f = n
                    .as_f64()
                    .ok_or_else(|| TaskError::InvalidInput(format!("Expected number, got: {n}")))?
                    as f32;
                data.push(f);
            }
            _ => {
                return Err(TaskError::InvalidInput("Expected array or number".into()));
            }
        }
        Ok(())
    }

    let shape = infer_shape(value)?;
    let mut data = Vec::new();
    get_data(value, &mut data)?;

    ArrayD::from_shape_vec(IxDyn(&shape), data)
        .map_err(|e| TaskError::InvalidInput(format!("Shape mismatch: {e}")))
}

/// Parse JSON value to i64 2D ndarray.
///
/// This is a specialized version for 2D arrays commonly used
/// for tokenized inputs (batch_size, sequence_length).
///
/// # Errors
///
/// Returns `TaskError::InvalidInput` if:
/// - The value is not a 2D array
/// - The shape doesn't match the data
pub fn json_to_array2_i64(value: &Value) -> TaskResult<ndarray::Array2<i64>> {
    fn get_data(v: &Value, data: &mut Vec<i64>) -> TaskResult<()> {
        match v {
            Value::Array(arr) => {
                for item in arr {
                    get_data(item, data)?;
                }
            }
            Value::Number(n) => {
                let i = n.as_i64().ok_or_else(|| {
                    TaskError::InvalidInput(format!("Expected integer, got: {n}"))
                })?;
                data.push(i);
            }
            _ => {
                return Err(TaskError::InvalidInput("Expected array or number".into()));
            }
        }
        Ok(())
    }

    // Get shape - assumes 2D
    let outer_len = value.as_array().map_or(0, Vec::len);
    let inner_len = value
        .as_array()
        .and_then(|a| a.first())
        .and_then(|v| v.as_array())
        .map_or(0, Vec::len);

    let mut data = Vec::new();
    get_data(value, &mut data)?;

    ndarray::Array2::from_shape_vec((outer_len, inner_len), data)
        .map_err(|e| TaskError::InvalidInput(format!("Shape mismatch: {e}")))
}

/// Parse JSON value to TensorValue (auto-detects type).
///
/// Inspects the JSON value to determine if it contains floats,
/// then parses accordingly.
pub fn json_to_tensor_value(value: &Value) -> TaskResult<TensorValue> {
    if contains_floats(value) {
        Ok(TensorValue::Float32(json_to_array_f32(value)?))
    } else {
        Ok(TensorValue::Int64(json_to_array_i64(value)?))
    }
}

/// Convert f32 ndarray to JSON value.
///
/// Recursively converts the ndarray to nested JSON arrays.
pub fn array_f32_to_json(array: &ArrayD<f32>) -> Value {
    #[allow(clippy::needless_pass_by_value)]
    fn to_json_recursive(slice: ndarray::ArrayViewD<f32>) -> Value {
        if slice.ndim() == 0 {
            serde_json::json!(slice.first().copied().unwrap_or(0.0))
        } else if slice.ndim() == 1 {
            let vec: Vec<f32> = slice.iter().copied().collect();
            serde_json::json!(vec)
        } else {
            let inner: Vec<Value> = slice.outer_iter().map(to_json_recursive).collect();
            serde_json::json!(inner)
        }
    }

    to_json_recursive(array.view())
}

/// Convert i64 ndarray to JSON value.
///
/// Recursively converts the ndarray to nested JSON arrays.
pub fn array_i64_to_json(array: &ArrayD<i64>) -> Value {
    #[allow(clippy::needless_pass_by_value)]
    fn to_json_recursive(slice: ndarray::ArrayViewD<i64>) -> Value {
        if slice.ndim() == 0 {
            serde_json::json!(slice.first().copied().unwrap_or(0))
        } else if slice.ndim() == 1 {
            let vec: Vec<i64> = slice.iter().copied().collect();
            serde_json::json!(vec)
        } else {
            let inner: Vec<Value> = slice.outer_iter().map(to_json_recursive).collect();
            serde_json::json!(inner)
        }
    }

    to_json_recursive(array.view())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_contains_floats() {
        assert!(!contains_floats(&json!([1, 2, 3])));
        assert!(contains_floats(&json!([1.5, 2.0, 3.0])));
        assert!(!contains_floats(&json!([[1, 2], [3, 4]])));
        assert!(contains_floats(&json!([[1.0, 2.0], [3.0, 4.0]])));
    }

    #[test]
    fn test_infer_shape() {
        assert_eq!(infer_shape(&json!([1, 2, 3])).unwrap(), vec![3]);
        assert_eq!(infer_shape(&json!([[1, 2], [3, 4]])).unwrap(), vec![2, 2]);
        assert_eq!(
            infer_shape(&json!([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])).unwrap(),
            vec![2, 2, 2]
        );
    }

    #[test]
    fn test_json_to_array_i64() {
        let value = json!([[1, 2, 3], [4, 5, 6]]);
        let array = json_to_array_i64(&value).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[0, 0]], 1);
        assert_eq!(array[[1, 2]], 6);
    }

    #[test]
    fn test_json_to_array_f32() {
        let value = json!([[1.0, 2.5], [3.5, 4.0]]);
        let array = json_to_array_f32(&value).unwrap();
        assert_eq!(array.shape(), &[2, 2]);
        assert!((array[[0, 1]] - 2.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_json_to_array2_i64() {
        let value = json!([[1, 2, 3], [4, 5, 6]]);
        let array = json_to_array2_i64(&value).unwrap();
        assert_eq!(array.shape(), &[2, 3]);
        assert_eq!(array[[0, 0]], 1);
    }

    #[test]
    fn test_roundtrip_f32() {
        let value = json!([[1.0, 2.0], [3.0, 4.0]]);
        let array = json_to_array_f32(&value).unwrap();
        let back = array_f32_to_json(&array);
        assert_eq!(value, back);
    }
}
