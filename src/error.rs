/// Errors arising from manifest parsing and validation.
#[derive(Debug, thiserror::Error)]
pub enum ManifestError {
    #[error("non-contiguous layer coverage: stage {stage_idx} ends at {end}, next starts at {next_start}")]
    NonContiguousLayers {
        stage_idx: usize,
        end: usize,
        next_start: usize,
    },
    #[error("empty stages list")]
    EmptyStages,
    #[error("stage {stage_idx}: layer_start ({start}) >= layer_end ({end})")]
    InvalidLayerRange {
        stage_idx: usize,
        start: usize,
        end: usize,
    },
    #[error("stages cover {covered} layers but total_layers is {total}")]
    LayerCountMismatch { covered: usize, total: usize },
    #[error("first stage must start at layer 0, but starts at {start}")]
    LayerStartNotZero { start: usize },
    #[error("stage {stage_idx} has wrong stage_idx field: {actual}")]
    WrongStageIndex { stage_idx: usize, actual: usize },
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Errors from the scheduler.
#[derive(Debug, thiserror::Error)]
pub enum SchedulerError {
    #[error("zero stages")]
    ZeroStages,
    #[error("zero micro-batches")]
    ZeroMicroBatches,
}

/// Errors from a pipeline stage.
#[derive(Debug, thiserror::Error)]
pub enum StageError {
    #[error("executor init failed: {0}")]
    InitFailed(String),
    #[error("forward pass failed for request {request_id}, micro-batch {micro_batch}: {reason}")]
    ForwardFailed {
        request_id: u64,
        micro_batch: u32,
        reason: String,
    },
    #[error("unexpected control message: {0}")]
    UnexpectedMessage(String),
    #[error("transport error: {0}")]
    Transport(#[from] confidential_ml_transport::Error),
    #[error("channel closed")]
    ChannelClosed,
    #[error("protocol error: {0}")]
    Protocol(String),
}

/// Top-level pipeline error.
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("manifest error: {0}")]
    Manifest(#[from] ManifestError),
    #[error("scheduler error: {0}")]
    Scheduler(#[from] SchedulerError),
    #[error("stage error: {0}")]
    Stage(#[from] StageError),
    #[error("transport error: {0}")]
    Transport(#[from] confidential_ml_transport::Error),
    #[error("stage {stage_idx} failed: {reason}")]
    StageFailed { stage_idx: usize, reason: String },
    #[error("request {request_id} failed: {reason}")]
    RequestFailed { request_id: u64, reason: String },
    #[error("pipeline shutting down")]
    Shutdown,
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("timeout: {0}")]
    Timeout(String),
    #[error("pipeline tainted after unrecoverable timeout; re-initialize to continue")]
    Tainted,
    #[error("protocol error: {0}")]
    Protocol(String),
    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, PipelineError>;
