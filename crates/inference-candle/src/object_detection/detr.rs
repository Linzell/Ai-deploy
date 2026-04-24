//! Full DETR model assembly.
//!
//! Pipeline: ResNet backbone → 1×1 conv projection → flatten + sine pos encoding
//! → Transformer encoder → Transformer decoder → class head + bbox head
//!
//! Supports both `DetrForObjectDetection` and `TableTransformerForObjectDetection`.

use candle_core::{Module, Result, Tensor};
use candle_nn::{linear, Linear, VarBuilder};

use super::config::DetrConfig;
use super::position_encoding::sine_position_embedding;
use super::resnet::{Conv2d, ResNetBackbone};
use super::transformer::{DetrDecoder, DetrEncoder};

/// Bounding box MLP: 3-layer MLP (d_model → d_model → d_model → 4).
struct BboxPredictor {
    layers: [Linear; 3],
}

impl BboxPredictor {
    fn load(vb: &VarBuilder, d_model: usize) -> Result<Self> {
        let l0 = linear(d_model, d_model, vb.pp("layers.0"))?;
        let l1 = linear(d_model, d_model, vb.pp("layers.1"))?;
        let l2 = linear(d_model, 4, vb.pp("layers.2"))?;
        Ok(Self {
            layers: [l0, l1, l2],
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.layers[0].forward(x)?.relu()?;
        let x = self.layers[1].forward(&x)?.relu()?;
        let x = self.layers[2].forward(&x)?;
        // Sigmoid to normalize to [0, 1]
        candle_nn::ops::sigmoid(&x)
    }
}

/// Full DETR model for object detection.
pub struct DetrForObjectDetection {
    backbone: ResNetBackbone,
    input_projection: Conv2d,
    query_position_embeddings: Tensor,
    encoder: DetrEncoder,
    decoder: DetrDecoder,
    class_labels_classifier: Linear,
    bbox_predictor: BboxPredictor,
    config: DetrConfig,
}

impl DetrForObjectDetection {
    /// Load model from safetensors.
    ///
    /// `vb` should be the root VarBuilder (no prefix).
    pub fn load(vb: &VarBuilder, config: &DetrConfig) -> Result<Self> {
        let model_vb = vb.pp("model");

        // Backbone
        let backbone = ResNetBackbone::load(
            &model_vb.pp("backbone").pp("conv_encoder").pp("model"),
            config.is_resnet18(),
        )?;

        // Input projection: 1×1 conv from backbone channels to d_model (with bias)
        let input_projection = Conv2d::load_with_bias(
            &model_vb.pp("input_projection"),
            config.d_model,
            config.backbone_out_channels(),
            1,
            1,
            0,
        )?;

        // Learned query position embeddings: [num_queries, d_model]
        let query_position_embeddings = model_vb
            .pp("query_position_embeddings")
            .get((config.num_queries, config.d_model), "weight")?;

        // Transformer
        let encoder = DetrEncoder::load(&model_vb.pp("encoder"), config)?;
        let decoder = DetrDecoder::load(&model_vb.pp("decoder"), config)?;

        // Detection heads (at root level, not under "model.")
        let num_labels = config.num_hidden_layers; // id2label length, loaded separately
        let class_labels_classifier = linear(
            config.d_model,
            num_labels + 1,
            vb.pp("class_labels_classifier"),
        )?;
        let bbox_predictor = BboxPredictor::load(&vb.pp("bbox_predictor"), config.d_model)?;

        Ok(Self {
            backbone,
            input_projection,
            query_position_embeddings,
            encoder,
            decoder,
            class_labels_classifier,
            bbox_predictor,
            config: config.clone(),
        })
    }

    /// Load model with explicit number of class labels.
    ///
    /// Use this when `id2label` from config.json gives the real label count.
    pub fn load_with_num_labels(
        vb: &VarBuilder,
        config: &DetrConfig,
        num_labels: usize,
    ) -> Result<Self> {
        let model_vb = vb.pp("model");

        let backbone = ResNetBackbone::load(
            &model_vb.pp("backbone").pp("conv_encoder").pp("model"),
            config.is_resnet18(),
        )?;

        let input_projection = Conv2d::load_with_bias(
            &model_vb.pp("input_projection"),
            config.d_model,
            config.backbone_out_channels(),
            1,
            1,
            0,
        )?;

        let query_position_embeddings = model_vb
            .pp("query_position_embeddings")
            .get((config.num_queries, config.d_model), "weight")?;

        let encoder = DetrEncoder::load(&model_vb.pp("encoder"), config)?;
        let decoder = DetrDecoder::load(&model_vb.pp("decoder"), config)?;

        let class_labels_classifier = linear(
            config.d_model,
            num_labels + 1,
            vb.pp("class_labels_classifier"),
        )?;
        let bbox_predictor = BboxPredictor::load(&vb.pp("bbox_predictor"), config.d_model)?;

        Ok(Self {
            backbone,
            input_projection,
            query_position_embeddings,
            encoder,
            decoder,
            class_labels_classifier,
            bbox_predictor,
            config: config.clone(),
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    /// - `pixel_values`: `[batch, 3, H, W]` — preprocessed image tensor
    ///
    /// # Returns
    /// `(logits, pred_boxes)` where:
    /// - `logits`: `[batch, num_queries, num_classes+1]`
    /// - `pred_boxes`: `[batch, num_queries, 4]` (normalized cx, cy, w, h)
    pub fn forward(&self, pixel_values: &Tensor) -> Result<(Tensor, Tensor)> {
        let (batch, _, _, _) = pixel_values.dims4()?;
        let device = pixel_values.device();
        let dtype = pixel_values.dtype();

        // 1. Backbone: [B, 3, H, W] → [B, C, H', W']
        let features = self.backbone.forward(pixel_values)?;
        let (_, _, fh, fw) = features.dims4()?;

        // 2. Input projection: [B, C, H', W'] → [B, d_model, H', W']
        let projected = self.input_projection.forward(&features)?;

        // 3. Flatten spatial dims: [B, d_model, H', W'] → [B, H'*W', d_model]
        let flattened = projected
            .reshape((batch, self.config.d_model, fh * fw))?
            .transpose(1, 2)?
            .contiguous()?;

        // 4. Position encoding: [B, H'*W', d_model]
        let position_embeddings = sine_position_embedding(
            batch,
            fh,
            fw,
            self.config.d_model / 2,
            10000.0,
            device,
            dtype,
        )?;

        // 5. Encoder
        let encoder_output = self
            .encoder
            .forward(&flattened, Some(&position_embeddings))?;

        // 6. Decoder
        // Object queries: zeros [B, num_queries, d_model]
        let object_queries = Tensor::zeros(
            (batch, self.config.num_queries, self.config.d_model),
            dtype,
            device,
        )?;

        // Query position: expand learned embeddings to batch
        let query_pos = self
            .query_position_embeddings
            .unsqueeze(0)?
            .broadcast_as((batch, self.config.num_queries, self.config.d_model))?
            .contiguous()?;

        let decoder_output = self.decoder.forward(
            &object_queries,
            &encoder_output,
            Some(&query_pos),
            Some(&position_embeddings),
        )?;

        // 7. Classification head: [B, num_queries, num_classes+1]
        let logits = self.class_labels_classifier.forward(&decoder_output)?;

        // 8. Bbox head: [B, num_queries, 4]
        let pred_boxes = self.bbox_predictor.forward(&decoder_output)?;

        Ok((logits, pred_boxes))
    }
}
