use serde::{Deserialize, Serialize};

use crate::error::PipelineError;

/// Current protocol version. Incremented on breaking wire-format changes.
pub const PROTOCOL_VERSION: u32 = 1;

/// Default maximum size for a control message in bytes (4 MiB).
pub const DEFAULT_MAX_CONTROL_MESSAGE_BYTES: usize = 4 * 1024 * 1024;

/// Wire envelope that wraps every control message with a protocol version.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Envelope<T> {
    /// Protocol version of the sender.
    pub version: u32,
    /// The inner message payload.
    pub msg: T,
}

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
    /// Serialize to JSON bytes inside a versioned envelope.
    pub fn to_bytes(&self) -> Result<bytes::Bytes, serde_json::Error> {
        let envelope = Envelope {
            version: PROTOCOL_VERSION,
            msg: self,
        };
        serde_json::to_vec(&envelope).map(bytes::Bytes::from)
    }

    /// Deserialize from a versioned envelope, checking protocol version and size.
    ///
    /// Returns `PipelineError::MessageTooLarge` if `data` exceeds `max_bytes`,
    /// `PipelineError::VersionMismatch` if the envelope version differs from
    /// `PROTOCOL_VERSION`, or a protocol error on malformed JSON.
    pub fn from_bytes_checked(data: &[u8], max_bytes: usize) -> crate::error::Result<Self> {
        if data.len() > max_bytes {
            return Err(PipelineError::MessageTooLarge {
                size: data.len(),
                limit: max_bytes,
            });
        }
        let envelope: Envelope<OrchestratorMsg> = serde_json::from_slice(data).map_err(|e| {
            PipelineError::Protocol(format!(
                "malformed orchestrator message ({} bytes): {e}",
                data.len()
            ))
        })?;
        if envelope.version != PROTOCOL_VERSION {
            return Err(PipelineError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                actual: envelope.version,
            });
        }
        Ok(envelope.msg)
    }

    /// Deserialize from bytes (legacy unversioned path, for backward compat in tests).
    pub fn from_bytes(data: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        // Try versioned envelope first, fall back to bare message.
        if let Ok(envelope) = serde_json::from_slice::<Envelope<OrchestratorMsg>>(data) {
            return Ok(envelope.msg);
        }
        serde_json::from_slice(data)
    }
}

impl StageMsg {
    /// Serialize to JSON bytes inside a versioned envelope.
    pub fn to_bytes(&self) -> Result<bytes::Bytes, serde_json::Error> {
        let envelope = Envelope {
            version: PROTOCOL_VERSION,
            msg: self,
        };
        serde_json::to_vec(&envelope).map(bytes::Bytes::from)
    }

    /// Deserialize from a versioned envelope, checking protocol version and size.
    ///
    /// Returns `PipelineError::MessageTooLarge` if `data` exceeds `max_bytes`,
    /// `PipelineError::VersionMismatch` if the envelope version differs from
    /// `PROTOCOL_VERSION`, or a protocol error on malformed JSON.
    pub fn from_bytes_checked(data: &[u8], max_bytes: usize) -> crate::error::Result<Self> {
        if data.len() > max_bytes {
            return Err(PipelineError::MessageTooLarge {
                size: data.len(),
                limit: max_bytes,
            });
        }
        let envelope: Envelope<StageMsg> = serde_json::from_slice(data).map_err(|e| {
            PipelineError::Protocol(format!(
                "malformed stage message ({} bytes): {e}",
                data.len()
            ))
        })?;
        if envelope.version != PROTOCOL_VERSION {
            return Err(PipelineError::VersionMismatch {
                expected: PROTOCOL_VERSION,
                actual: envelope.version,
            });
        }
        Ok(envelope.msg)
    }

    /// Deserialize from bytes (legacy unversioned path, for backward compat in tests).
    pub fn from_bytes(data: &[u8]) -> std::result::Result<Self, serde_json::Error> {
        // Try versioned envelope first, fall back to bare message.
        if let Ok(envelope) = serde_json::from_slice::<Envelope<StageMsg>>(data) {
            return Ok(envelope.msg);
        }
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

    #[test]
    fn envelope_contains_version() {
        let msg = OrchestratorMsg::Ping { seq: 42 };
        let bytes = msg.to_bytes().unwrap();
        let raw: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(raw["version"], PROTOCOL_VERSION);
        assert!(raw["msg"].is_object());
    }

    #[test]
    fn from_bytes_checked_rejects_wrong_version() {
        let envelope = Envelope {
            version: 999,
            msg: OrchestratorMsg::Ping { seq: 1 },
        };
        let data = serde_json::to_vec(&envelope).unwrap();
        let result = OrchestratorMsg::from_bytes_checked(&data, 4 * 1024 * 1024);
        match result {
            Err(crate::error::PipelineError::VersionMismatch { expected, actual }) => {
                assert_eq!(expected, PROTOCOL_VERSION);
                assert_eq!(actual, 999);
            }
            other => panic!("expected VersionMismatch, got {other:?}"),
        }

        // Stage messages too.
        let stage_envelope = Envelope {
            version: 0,
            msg: StageMsg::Pong { seq: 1 },
        };
        let data = serde_json::to_vec(&stage_envelope).unwrap();
        let result = StageMsg::from_bytes_checked(&data, 4 * 1024 * 1024);
        assert!(matches!(
            result,
            Err(crate::error::PipelineError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn from_bytes_checked_rejects_oversized() {
        let msg = OrchestratorMsg::Ping { seq: 1 };
        let data = msg.to_bytes().unwrap();
        // Set limit smaller than the message.
        let result = OrchestratorMsg::from_bytes_checked(&data, 5);
        match result {
            Err(crate::error::PipelineError::MessageTooLarge { size, limit }) => {
                assert_eq!(size, data.len());
                assert_eq!(limit, 5);
            }
            other => panic!("expected MessageTooLarge, got {other:?}"),
        }

        // Stage messages too.
        let stage_msg = StageMsg::Pong { seq: 1 };
        let data = stage_msg.to_bytes().unwrap();
        let result = StageMsg::from_bytes_checked(&data, 2);
        assert!(matches!(
            result,
            Err(crate::error::PipelineError::MessageTooLarge { .. })
        ));
    }

    #[test]
    fn from_bytes_checked_rejects_malformed_json() {
        let result = OrchestratorMsg::from_bytes_checked(b"not json at all", 1024);
        assert!(matches!(
            result,
            Err(crate::error::PipelineError::Protocol(_))
        ));

        let result = StageMsg::from_bytes_checked(b"{truncated", 1024);
        assert!(matches!(
            result,
            Err(crate::error::PipelineError::Protocol(_))
        ));
    }

    #[test]
    fn from_bytes_checked_rejects_truncated_envelope() {
        // Valid JSON but not a valid envelope.
        let result = OrchestratorMsg::from_bytes_checked(b"{\"version\":1}", 1024);
        assert!(matches!(
            result,
            Err(crate::error::PipelineError::Protocol(_))
        ));
    }

    #[test]
    fn from_bytes_checked_accepts_valid_message() {
        let msg = OrchestratorMsg::StartRequest {
            request_id: 42,
            num_micro_batches: 4,
            seq_len: 128,
        };
        let data = msg.to_bytes().unwrap();
        let decoded = OrchestratorMsg::from_bytes_checked(&data, 4 * 1024 * 1024).unwrap();
        match decoded {
            OrchestratorMsg::StartRequest {
                request_id,
                num_micro_batches,
                seq_len,
            } => {
                assert_eq!(request_id, 42);
                assert_eq!(num_micro_batches, 4);
                assert_eq!(seq_len, 128);
            }
            other => panic!("expected StartRequest, got {other:?}"),
        }
    }

    #[test]
    fn from_bytes_checked_accepts_exact_size_limit() {
        let msg = StageMsg::Pong { seq: 1 };
        let data = msg.to_bytes().unwrap();
        // Set limit exactly to the message size — should pass.
        let result = StageMsg::from_bytes_checked(&data, data.len());
        assert!(result.is_ok());
    }

    #[test]
    fn from_bytes_legacy_accepts_versioned() {
        // to_bytes() now produces versioned envelopes; from_bytes() should decode them.
        let msg = OrchestratorMsg::Shutdown;
        let data = msg.to_bytes().unwrap();
        let decoded = OrchestratorMsg::from_bytes(&data).unwrap();
        assert!(matches!(decoded, OrchestratorMsg::Shutdown));
    }

    #[test]
    fn from_bytes_legacy_accepts_bare() {
        // Bare (unversioned) JSON should still decode via fallback.
        let bare = br#"{"type":"Shutdown"}"#;
        let decoded = OrchestratorMsg::from_bytes(bare).unwrap();
        assert!(matches!(decoded, OrchestratorMsg::Shutdown));
    }
}
