use serde::{Deserialize, Serialize};

/// Messages sent from the orchestrator to a stage over the control channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum OrchestratorMsg {
    /// Initialize stage with its spec and activation format.
    Init {
        stage_spec_json: String,
        activation_spec_json: String,
        num_stages: usize,
    },
    /// Tell stage to accept data channel connections.
    EstablishDataChannels {
        has_upstream: bool,
        has_downstream: bool,
    },
    /// Start processing a new inference request.
    StartRequest {
        request_id: u64,
        num_micro_batches: u32,
        seq_len: u32,
    },
    /// Abort an in-progress request.
    AbortRequest { request_id: u64, reason: String },
    /// Shut down the stage gracefully.
    Shutdown,
    /// Health check ping.
    Ping { seq: u64 },
}

/// Messages sent from a stage back to the orchestrator over the control channel.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum StageMsg {
    /// Stage has finished initialization and is ready.
    Ready { stage_idx: usize },
    /// Data channels have been established.
    DataChannelsReady { stage_idx: usize },
    /// Request completed successfully.
    RequestDone { request_id: u64 },
    /// Request failed with an error.
    RequestError { request_id: u64, error: String },
    /// Health check pong.
    Pong { seq: u64 },
    /// Stage is shutting down.
    ShuttingDown { stage_idx: usize },
}

impl OrchestratorMsg {
    /// Serialize to JSON bytes for sending over a SecureChannel.
    pub fn to_bytes(&self) -> Result<bytes::Bytes, serde_json::Error> {
        serde_json::to_vec(self).map(bytes::Bytes::from)
    }

    /// Deserialize from bytes received from a SecureChannel.
    pub fn from_bytes(data: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

impl StageMsg {
    /// Serialize to JSON bytes for sending over a SecureChannel.
    pub fn to_bytes(&self) -> Result<bytes::Bytes, serde_json::Error> {
        serde_json::to_vec(self).map(bytes::Bytes::from)
    }

    /// Deserialize from bytes received from a SecureChannel.
    pub fn from_bytes(data: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        serde_json::from_slice(data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn orchestrator_msg_roundtrip() {
        let msgs = vec![
            OrchestratorMsg::Init {
                stage_spec_json: r#"{"stage_idx":0}"#.into(),
                activation_spec_json: r#"{"dtype":"F32"}"#.into(),
                num_stages: 3,
            },
            OrchestratorMsg::EstablishDataChannels {
                has_upstream: false,
                has_downstream: true,
            },
            OrchestratorMsg::StartRequest {
                request_id: 42,
                num_micro_batches: 4,
                seq_len: 128,
            },
            OrchestratorMsg::AbortRequest {
                request_id: 42,
                reason: "stage 1 failed".into(),
            },
            OrchestratorMsg::Shutdown,
            OrchestratorMsg::Ping { seq: 1 },
        ];

        for msg in msgs {
            let bytes = msg.to_bytes().unwrap();
            let decoded = OrchestratorMsg::from_bytes(&bytes).unwrap();
            // Verify tag-based discrimination round-trips
            let re_bytes = decoded.to_bytes().unwrap();
            assert_eq!(bytes, re_bytes);
        }
    }

    #[test]
    fn stage_msg_roundtrip() {
        let msgs = vec![
            StageMsg::Ready { stage_idx: 0 },
            StageMsg::DataChannelsReady { stage_idx: 1 },
            StageMsg::RequestDone { request_id: 42 },
            StageMsg::RequestError {
                request_id: 42,
                error: "OOM".into(),
            },
            StageMsg::Pong { seq: 1 },
            StageMsg::ShuttingDown { stage_idx: 2 },
        ];

        for msg in msgs {
            let bytes = msg.to_bytes().unwrap();
            let decoded = StageMsg::from_bytes(&bytes).unwrap();
            let re_bytes = decoded.to_bytes().unwrap();
            assert_eq!(bytes, re_bytes);
        }
    }

    #[test]
    fn invalid_json_returns_error() {
        assert!(OrchestratorMsg::from_bytes(b"not json").is_err());
        assert!(StageMsg::from_bytes(b"{\"type\":\"Unknown\"}").is_err());
    }
}
