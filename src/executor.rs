use async_trait::async_trait;
use confidential_ml_transport::OwnedTensor;

use crate::error::StageError;
use crate::manifest::StageSpec;

/// Unique identifier for an inference request.
pub type RequestId = u64;

/// Output from a single forward pass (one micro-batch through one stage).
pub struct ForwardOutput {
    /// Activation tensors to forward to the next stage (or final output for the last stage).
    pub tensors: Vec<OwnedTensor>,
}

/// User-implemented trait for the computation within a pipeline stage.
///
/// Each stage holds a shard of the model and executes forward passes
/// on incoming activation tensors.
#[async_trait]
pub trait StageExecutor: Send + Sync {
    /// Initialize the executor with its stage specification (load weights, etc.).
    async fn init(&mut self, stage_spec: &StageSpec) -> std::result::Result<(), StageError>;

    /// Return SHA-256 hashes (hex-encoded) of loaded model weights.
    ///
    /// Called after [`init`](Self::init) to verify weight integrity against
    /// the manifest's `weight_hashes`. Default returns an empty vec, which
    /// will fail verification if the manifest declares any weight hashes.
    fn weight_hashes(&self) -> Vec<String> {
        Vec::new()
    }

    /// Run a forward pass on one micro-batch of input tensors.
    ///
    /// - `request_id`: identifies the inference request.
    /// - `micro_batch`: index of this micro-batch within the request.
    /// - `inputs`: activation tensors from the previous stage (or input tensors for stage 0).
    ///
    /// Returns the output tensors to forward to the next stage.
    async fn forward(
        &self,
        request_id: RequestId,
        micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> std::result::Result<ForwardOutput, StageError>;
}
