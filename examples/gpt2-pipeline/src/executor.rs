use std::path::PathBuf;
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use candle_core::{Device, Tensor};
use confidential_ml_transport::{DType, OwnedTensor};
use tracing::info;

use confidential_ml_pipeline::{ForwardOutput, RequestId, StageError, StageExecutor, StageSpec};

use crate::model::{Gpt2Config, Gpt2Shard};

/// Converts an OwnedTensor with DType::U32 to a candle Tensor.
fn owned_to_candle_u32(t: &OwnedTensor, device: &Device) -> Result<Tensor, StageError> {
    let num_elems: usize = t.shape.iter().map(|&d| d as usize).product();
    if t.data.len() != num_elems * 4 {
        return Err(StageError::ForwardFailed {
            request_id: 0,
            micro_batch: 0,
            reason: format!(
                "U32 tensor size mismatch: {} bytes for {} elements",
                t.data.len(),
                num_elems
            ),
        });
    }
    let values: Vec<u32> = t
        .data
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
    Tensor::from_vec(values, shape.as_slice(), device).map_err(|e| StageError::ForwardFailed {
        request_id: 0,
        micro_batch: 0,
        reason: format!("candle tensor creation failed: {e}"),
    })
}

/// Converts an OwnedTensor with DType::F32 to a candle Tensor.
fn owned_to_candle_f32(t: &OwnedTensor, device: &Device) -> Result<Tensor, StageError> {
    let num_elems: usize = t.shape.iter().map(|&d| d as usize).product();
    if t.data.len() != num_elems * 4 {
        return Err(StageError::ForwardFailed {
            request_id: 0,
            micro_batch: 0,
            reason: format!(
                "F32 tensor size mismatch: {} bytes for {} elements",
                t.data.len(),
                num_elems
            ),
        });
    }
    let values: Vec<f32> = t
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    let shape: Vec<usize> = t.shape.iter().map(|&d| d as usize).collect();
    Tensor::from_vec(values, shape.as_slice(), device).map_err(|e| StageError::ForwardFailed {
        request_id: 0,
        micro_batch: 0,
        reason: format!("candle tensor creation failed: {e}"),
    })
}

/// Converts a candle Tensor to OwnedTensor, preserving the original shape.
fn candle_to_owned_f32_shaped(
    t: &Tensor,
    name: &str,
    orig_shape: &[usize],
) -> Result<OwnedTensor, StageError> {
    let flat = t.flatten_all().map_err(|e| StageError::ForwardFailed {
        request_id: 0,
        micro_batch: 0,
        reason: format!("flatten failed: {e}"),
    })?;
    let values = flat.to_vec1::<f32>().map_err(|e| StageError::ForwardFailed {
        request_id: 0,
        micro_batch: 0,
        reason: format!("to_vec1 failed: {e}"),
    })?;
    let shape: Vec<u32> = orig_shape.iter().map(|&d| d as u32).collect();
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    Ok(OwnedTensor {
        name: name.to_string(),
        dtype: DType::F32,
        shape,
        data: Bytes::from(data),
    })
}

pub struct Gpt2StageExecutor {
    model_dir: PathBuf,
    shard: Option<Arc<Gpt2Shard>>,
    cfg: Option<Gpt2Config>,
    is_first: bool,
    is_last: bool,
}

impl Gpt2StageExecutor {
    pub fn new(model_dir: PathBuf) -> Self {
        Self {
            model_dir,
            shard: None,
            cfg: None,
            is_first: false,
            is_last: false,
        }
    }
}

#[async_trait]
impl StageExecutor for Gpt2StageExecutor {
    async fn init(&mut self, stage_spec: &StageSpec) -> Result<(), StageError> {
        let cfg = Gpt2Config::from_json(&self.model_dir.join("config.json")).map_err(|e| {
            StageError::InitFailed(format!("failed to load config.json: {e}"))
        })?;

        self.is_first = stage_spec.layer_start == 0;
        self.is_last = stage_spec.layer_end == cfg.n_layer;

        info!(
            stage = stage_spec.stage_idx,
            layers = format!("{}-{}", stage_spec.layer_start, stage_spec.layer_end),
            is_first = self.is_first,
            is_last = self.is_last,
            "loading GPT-2 shard"
        );

        let shard = Gpt2Shard::load(
            &self.model_dir,
            &cfg,
            stage_spec.layer_start,
            stage_spec.layer_end,
            self.is_first,
            self.is_last,
            &Device::Cpu,
        )
        .map_err(|e| StageError::InitFailed(format!("failed to load model shard: {e}")))?;

        info!(
            stage = stage_spec.stage_idx,
            "GPT-2 shard loaded"
        );

        self.shard = Some(Arc::new(shard));
        self.cfg = Some(cfg);
        Ok(())
    }

    async fn forward(
        &self,
        request_id: RequestId,
        micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        let shard = self.shard.as_ref().ok_or_else(|| StageError::ForwardFailed {
            request_id,
            micro_batch,
            reason: "shard not initialized".to_string(),
        })?;

        let input_tensor = inputs.first().ok_or_else(|| StageError::ForwardFailed {
            request_id,
            micro_batch,
            reason: "no input tensor".to_string(),
        })?;

        // Check for cache-clear sentinel: a U32 tensor with shape [0] signals new prompt.
        if input_tensor.dtype == DType::U32 && input_tensor.shape == [0] {
            shard.clear_cache();
            // The actual input follows as the second tensor.
            let actual_input = inputs.get(1).ok_or_else(|| StageError::ForwardFailed {
                request_id,
                micro_batch,
                reason: "cache-clear sentinel without actual input".to_string(),
            })?;
            return self.run_forward(shard, actual_input, request_id, micro_batch);
        }

        self.run_forward(shard, input_tensor, request_id, micro_batch)
    }
}

impl Gpt2StageExecutor {
    fn run_forward(
        &self,
        shard: &Gpt2Shard,
        input_tensor: &OwnedTensor,
        request_id: RequestId,
        micro_batch: u32,
    ) -> Result<ForwardOutput, StageError> {
        let device = Device::Cpu;
        let candle_input = if self.is_first {
            owned_to_candle_u32(input_tensor, &device)?
        } else {
            owned_to_candle_f32(input_tensor, &device)?
        };

        let output = shard.forward(&candle_input).map_err(|e| StageError::ForwardFailed {
            request_id,
            micro_batch,
            reason: format!("forward failed: {e}"),
        })?;

        let output_dims: Vec<usize> = output.dims().to_vec();
        let output_tensor = candle_to_owned_f32_shaped(
            &output,
            if self.is_last { "logits" } else { "hidden_states" },
            &output_dims,
        )?;

        info!(
            stage_first = self.is_first,
            stage_last = self.is_last,
            micro_batch,
            output_shape = ?output_dims,
            "forward complete"
        );

        Ok(ForwardOutput {
            tensors: vec![output_tensor],
        })
    }
}
