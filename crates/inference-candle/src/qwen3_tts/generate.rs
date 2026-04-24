//! Two-stage generation loop for Qwen3 TTS.
//!
//! Stage 1: Talker generates CB0 tokens autoregressively from text input.
//! Stage 2: Code predictor generates CB1-CB15 for each CB0 token.
//!
//! Then the speech tokenizer converts all 16 codebooks to a waveform.

use candle_core::{DType, Device, IndexOp, Tensor};
use tracing::{debug, info};

use crate::error::{TaskError, TaskResult};
use crate::utils;

use super::code_predictor::CodePredictor;
use super::config::{Qwen3TtsConfig, TalkerConfig};
use super::speech_tokenizer::SpeechTokenizer;
use super::talker::Talker;

/// Full Qwen3 TTS model: talker + code predictor + speech tokenizer.
pub struct Qwen3TtsModel {
    talker: Talker,
    code_predictor: CodePredictor,
    speech_tokenizer: Option<SpeechTokenizer>,
    config: TalkerConfig,
    device: Device,
}

impl Qwen3TtsModel {
    /// Load the full model from a model directory.
    ///
    /// Expects:
    /// - `config.json` — top-level Qwen3TtsConfig
    /// - `model.safetensors` — talker + code predictor weights (prefixed `talker.`)
    /// - `speech_tokenizer/model.safetensors` — speech tokenizer weights (optional)
    /// - `speech_tokenizer/config.json` — speech tokenizer config (optional)
    pub fn from_model_dir(
        model_dir: &std::path::Path,
        device: &Device,
        dtype: DType,
    ) -> TaskResult<Self> {
        // Parse top-level config
        let config_path = model_dir.join("config.json");
        let raw: serde_json::Value = serde_json::from_reader(
            std::fs::File::open(&config_path)
                .map_err(|e| TaskError::ModelLoad(format!("config.json: {e}")))?,
        )
        .map_err(|e| TaskError::ModelLoad(format!("config.json parse: {e}")))?;

        let tts_config: Qwen3TtsConfig = serde_json::from_value(raw)
            .map_err(|e| TaskError::ModelLoad(format!("Qwen3TtsConfig parse: {e}")))?;

        let talker_config = &tts_config.talker_config;

        // Load model weights
        let weight_files = utils::find_weight_files(model_dir)?;
        let vb = utils::load_safetensors_safe(&weight_files, dtype, device)?;

        // Weights are prefixed with `talker.`
        let vb_talker = vb.pp("talker");

        info!(
            "Loading Qwen3 TTS talker ({} layers)",
            talker_config.num_hidden_layers
        );
        let talker = Talker::new(talker_config, vb_talker.clone())
            .map_err(|e| TaskError::ModelLoad(format!("Talker: {e}")))?;

        info!(
            "Loading Qwen3 TTS code predictor ({} layers, {} codebooks)",
            talker_config.code_predictor_config.num_hidden_layers,
            talker_config.code_predictor_config.num_codebooks()
        );
        let code_predictor = CodePredictor::new(
            &talker_config.code_predictor_config,
            talker_config.hidden_size,
            talker_config.codec_vocab_size(),
            vb_talker.pp("code_predictor"),
        )
        .map_err(|e| TaskError::ModelLoad(format!("CodePredictor: {e}")))?;

        // Try to load speech tokenizer (optional — may not be downloaded yet)
        let speech_tok_dir = model_dir.join("speech_tokenizer");
        let speech_tokenizer = if speech_tok_dir.exists() {
            match SpeechTokenizer::from_dir(&speech_tok_dir, device) {
                Ok(st) => {
                    info!(sample_rate = st.sample_rate, "Loaded speech tokenizer");
                    Some(st)
                }
                Err(e) => {
                    tracing::warn!(
                        "Failed to load speech tokenizer: {e}. Audio decoding will be unavailable."
                    );
                    None
                }
            }
        } else {
            tracing::warn!(
                "speech_tokenizer/ directory not found. Audio decoding will be unavailable. \
                 Codec tokens will be returned directly."
            );
            None
        };

        Ok(Self {
            talker,
            code_predictor,
            speech_tokenizer,
            config: talker_config.clone(),
            device: device.clone(),
        })
    }

    /// Generate speech from text token IDs.
    ///
    /// `text_token_ids`: tokenized text, shape `[seq_len]`.
    /// `speaker_id`: optional speaker token ID to prepend.
    /// `language_id`: optional language token ID to prepend.
    /// `max_frames`: maximum number of audio frames to generate.
    /// `temperature`: sampling temperature (0 = greedy).
    ///
    /// Returns audio samples as `Vec<f32>` (mono, 24kHz).
    pub fn generate(
        &mut self,
        text_token_ids: &[u32],
        speaker_id: Option<u32>,
        language_id: Option<u32>,
        max_frames: usize,
        temperature: f64,
    ) -> TaskResult<GenerateOutput> {
        self.talker.clear_kv_cache();

        // Build input sequence: [tts_bos, (speaker_id), (language_id), ...text_tokens..., tts_eos, codec_bos]
        let mut input_ids: Vec<u32> = Vec::new();
        // TTS framing tokens
        input_ids.push(self.config.codec_bos_id); // Start of codec generation

        // Prepend speaker/language if provided
        if let Some(spk) = speaker_id {
            input_ids.push(spk);
        }
        if let Some(lang) = language_id {
            input_ids.push(lang);
        }

        // The text tokens need to be projected through text_embedding → text_projection
        // while codec tokens use codec_embedding. So we process them in two phases.

        // Phase 1: Prefill with text tokens
        let text_ids = Tensor::new(text_token_ids, &self.device)
            .map_err(|e| TaskError::Inference(format!("text tensor: {e}")))?;
        let text_embeds = self
            .talker
            .model
            .embed_text(&text_ids)
            .map_err(|e| TaskError::Inference(format!("text embed: {e}")))?;
        let text_embeds = self
            .talker
            .text_projection
            .forward(&text_embeds)
            .map_err(|e| TaskError::Inference(format!("text projection: {e}")))?;

        // Add batch dimension: [seq_len, hidden] → [1, seq_len, hidden]
        let text_embeds = text_embeds
            .unsqueeze(0)
            .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?;

        // Prefill: run text through the talker
        let hidden = self
            .talker
            .model
            .forward(&text_embeds, 0)
            .map_err(|e| TaskError::Inference(format!("talker prefill: {e}")))?;

        let seq_offset = text_token_ids.len();

        // Get logits for the last position → first CB0 token
        let last_hidden = hidden
            .i((.., hidden.dim(1).unwrap() - 1.., ..))
            .map_err(|e| TaskError::Inference(format!("last hidden: {e}")))?;
        let logits = self
            .talker
            .codec_logits(&last_hidden)
            .map_err(|e| TaskError::Inference(format!("codec logits: {e}")))?;

        // Phase 2: Autoregressive codec generation
        let mut all_cb0_tokens: Vec<u32> = Vec::new();
        let mut all_codebook_tokens: Vec<Vec<u32>> = Vec::new(); // [num_frames][16]
        let codec_eos = self.config.codec_eos_token_id;

        let mut current_logits = logits;
        let mut current_hidden = last_hidden; // Track hidden state for code predictor
        let mut step_offset = seq_offset;

        for frame_idx in 0..max_frames {
            // Sample CB0 token
            // current_logits: [1, 1, codec_vocab] → flatten to [codec_vocab]
            let cb0_logits = current_logits
                .flatten_all()
                .map_err(|e| TaskError::Inference(format!("flatten logits: {e}")))?;
            let cb0_token = if temperature <= 0.0 {
                cb0_logits
                    .argmax(0)
                    .map_err(|e| TaskError::Inference(format!("argmax: {e}")))?
            } else {
                let scaled = (cb0_logits / temperature)
                    .map_err(|e| TaskError::Inference(format!("scale: {e}")))?;
                let probs = candle_nn::ops::softmax_last_dim(&scaled)
                    .map_err(|e| TaskError::Inference(format!("softmax: {e}")))?;
                probs
                    .argmax(0)
                    .map_err(|e| TaskError::Inference(format!("sample: {e}")))?
            };

            let cb0_val: u32 = cb0_token
                .to_scalar()
                .map_err(|e| TaskError::Inference(format!("scalar: {e}")))?;

            // Check EOS
            if cb0_val == codec_eos {
                debug!(frame = frame_idx, "CB0 EOS reached");
                break;
            }

            all_cb0_tokens.push(cb0_val);

            // Get semantic embedding of CB0 token (from talker's codec_embedding)
            // This is used as part of the code predictor prefill alongside talker_hidden
            let cb0_embed_for_cp = self
                .talker
                .model
                .embed_codec(
                    &Tensor::new(&[cb0_val], &self.device)
                        .map_err(|e| TaskError::Inference(format!("cb0 cp tensor: {e}")))?,
                )
                .map_err(|e| TaskError::Inference(format!("cb0 cp embed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TaskError::Inference(format!("cb0 cp unsqueeze: {e}")))?;

            // Predict CB1-CB15 using code predictor
            // current_hidden: [1, 1, hidden_size] — the talker hidden at this frame
            // cb0_embed_for_cp: [1, 1, hidden_size] — semantic embedding of CB0
            let cb_rest = self
                .code_predictor
                .predict_codebooks(&current_hidden, &cb0_embed_for_cp, temperature)
                .map_err(|e| TaskError::Inference(format!("code predictor: {e}")))?;

            let mut frame_codes = vec![cb0_val];
            frame_codes.extend_from_slice(&cb_rest);
            all_codebook_tokens.push(frame_codes);

            // Prepare next step: embed CB0 token through codec_embedding
            let cb0_tensor = Tensor::new(&[cb0_val], &self.device)
                .map_err(|e| TaskError::Inference(format!("cb0 tensor: {e}")))?;
            let cb0_embed = self
                .talker
                .model
                .embed_codec(&cb0_tensor)
                .map_err(|e| TaskError::Inference(format!("codec embed: {e}")))?
                .unsqueeze(0)
                .map_err(|e| TaskError::Inference(format!("unsqueeze: {e}")))?;

            step_offset += 1;
            let hidden = self
                .talker
                .model
                .forward(&cb0_embed, step_offset)
                .map_err(|e| TaskError::Inference(format!("talker step {frame_idx}: {e}")))?;

            // Update hidden state for next code predictor call
            current_hidden = hidden.clone();

            current_logits = self
                .talker
                .codec_logits(&hidden)
                .map_err(|e| TaskError::Inference(format!("codec logits step {frame_idx}: {e}")))?;

            if frame_idx % 50 == 0 {
                debug!(frame = frame_idx, cb0 = cb0_val, "Generating...");
            }
        }

        let num_frames = all_codebook_tokens.len();
        info!(
            num_frames,
            "Generated {} frames of codec tokens", num_frames
        );

        // Decode to audio if speech tokenizer is available
        if let Some(ref speech_tok) = self.speech_tokenizer {
            let num_codebooks = self.config.code_predictor_config.num_code_groups;
            // Build codes tensor: [16, num_frames]
            let mut codes_flat: Vec<u32> = vec![0u32; num_codebooks * num_frames];
            for (frame_idx, frame_codes) in all_codebook_tokens.iter().enumerate() {
                for (cb_idx, &code) in frame_codes.iter().enumerate() {
                    codes_flat[cb_idx * num_frames + frame_idx] = code;
                }
            }

            let codes = Tensor::new(codes_flat, &self.device)
                .map_err(|e| TaskError::Inference(format!("codes tensor: {e}")))?
                .reshape((num_codebooks, num_frames))
                .map_err(|e| TaskError::Inference(format!("codes reshape: {e}")))?;

            let waveform = speech_tok
                .decode(&codes)
                .map_err(|e| TaskError::Inference(format!("speech decode: {e}")))?;

            let samples: Vec<f32> = waveform
                .to_vec1()
                .map_err(|e| TaskError::Inference(format!("waveform to vec: {e}")))?;

            info!(
                samples = samples.len(),
                duration_secs = samples.len() as f32 / speech_tok.sample_rate as f32,
                "Decoded audio waveform"
            );

            Ok(GenerateOutput {
                waveform: Some(samples),
                sample_rate: speech_tok.sample_rate,
                codec_tokens: all_codebook_tokens,
            })
        } else {
            // Return codec tokens without audio
            Ok(GenerateOutput {
                waveform: None,
                sample_rate: 24000,
                codec_tokens: all_codebook_tokens,
            })
        }
    }

    pub fn clear_kv_cache(&mut self) {
        self.talker.clear_kv_cache();
        self.code_predictor.clear_kv_cache();
    }
}

/// Output from Qwen3 TTS generation.
pub struct GenerateOutput {
    /// Audio waveform samples (mono, sample_rate Hz). None if speech tokenizer unavailable.
    pub waveform: Option<Vec<f32>>,
    /// Audio sample rate.
    pub sample_rate: u32,
    /// Raw codec tokens: `[num_frames][16]` (CB0-CB15).
    pub codec_tokens: Vec<Vec<u32>>,
}
