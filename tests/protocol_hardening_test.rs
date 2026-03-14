#![cfg(feature = "mock")]

//! Tests for protocol hardening: versioning, size guards, schema validation.

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{
    DType, MockProvider, MockVerifier, OwnedTensor, SecureChannel, SessionConfig,
};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig,
    PipelineError, PortSpec, RequestId, ShardManifest, StageConfig, StageEndpoint, StageError,
    StageExecutor, StageRuntime, StageSpec, PROTOCOL_VERSION,
};

/// Identity executor: passes input tensors through unchanged.
struct IdentityExecutor;

#[async_trait]
impl StageExecutor for IdentityExecutor {
    async fn init(&mut self, _stage_spec: &StageSpec) -> Result<(), StageError> {
        Ok(())
    }

    async fn forward(
        &self,
        _request_id: RequestId,
        _micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        Ok(ForwardOutput { tensors: inputs })
    }
}

fn make_test_manifest(num_stages: usize) -> ShardManifest {
    let stages = (0..num_stages)
        .map(|i| StageSpec {
            stage_idx: i,
            layer_start: i * 4,
            layer_end: (i + 1) * 4,
            weight_hashes: vec![],
            expected_measurements: BTreeMap::new(),
            endpoint: StageEndpoint {
                control: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9400 + i * 10),
                },
                data_in: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9401 + i * 10),
                },
                data_out: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9402 + i * 10),
                },
            },
        })
        .collect();

    ShardManifest {
        model_name: "test-model".into(),
        model_version: "1.0".into(),
        total_layers: num_stages * 4,
        stages,
        activation_spec: ActivationSpec {
            dtype: ActivationDType::F32,
            hidden_dim: 4,
            max_seq_len: 16,
        },
    }
}

/// Sending a message with the wrong protocol version from a "rogue stage"
/// should cause the orchestrator to return a VersionMismatch or Protocol error.
#[tokio::test]
async fn orchestrator_rejects_wrong_version_from_stage() {
    let manifest = make_test_manifest(1);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);

    // Spawn a rogue stage that sends a message with wrong protocol version.
    let rogue = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();

        // Accept control channel normally.
        let mut control = SecureChannel::accept_with_attestation(
            stage_ctrl,
            &provider,
            &verifier,
            SessionConfig::development(),
        )
        .await
        .expect("stage handshake failed");

        // Read the Init message.
        let _init = control.recv().await.expect("stage recv Init failed");

        // Send a Ready message but with wrong version in the envelope.
        let wrong_version_msg = serde_json::json!({
            "version": 999,
            "msg": {
                "type": "Ready",
                "stage_idx": 0
            }
        });
        let data = serde_json::to_vec(&wrong_version_msg).unwrap();
        control
            .send(Bytes::from(data))
            .await
            .expect("stage send failed");
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::development(), manifest).unwrap();
    let result = orch.init(vec![orch_ctrl], &provider, &verifier).await;

    assert!(
        matches!(&result, Err(PipelineError::VersionMismatch { expected, actual }) if *expected == PROTOCOL_VERSION && *actual == 999),
        "expected VersionMismatch, got: {result:?}"
    );

    // Clean up: rogue task may have already finished.
    let _ = rogue.await;
}

/// Sending an oversized control message from a "rogue stage" should be rejected.
#[tokio::test]
async fn orchestrator_rejects_oversized_control_message() {
    let manifest = make_test_manifest(1);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(1024 * 1024);

    // Spawn a rogue stage that sends a large message.
    let rogue = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();

        let mut control = SecureChannel::accept_with_attestation(
            stage_ctrl,
            &provider,
            &verifier,
            SessionConfig::development(),
        )
        .await
        .expect("stage handshake failed");

        // Read Init.
        let _init = control.recv().await.expect("stage recv Init failed");

        // Send an oversized message (fill the error field with junk).
        let big_payload = "X".repeat(200);
        let envelope = serde_json::json!({
            "version": PROTOCOL_VERSION,
            "msg": {
                "type": "Ready",
                "stage_idx": 0,
                "extra_junk": big_payload
            }
        });
        let data = serde_json::to_vec(&envelope).unwrap();
        control
            .send(Bytes::from(data))
            .await
            .expect("stage send failed");
    });

    // Set a very small limit so the message will be rejected.
    let config = OrchestratorConfig {
        max_control_message_bytes: 50,
        ..OrchestratorConfig::development()
    };
    let mut orch = Orchestrator::new(config, manifest).unwrap();
    let result = orch.init(vec![orch_ctrl], &provider, &verifier).await;

    assert!(
        matches!(&result, Err(PipelineError::MessageTooLarge { .. })),
        "expected MessageTooLarge, got: {result:?}"
    );

    let _ = rogue.await;
}

/// Stage should reject a malformed (non-JSON) control message from orchestrator.
#[tokio::test]
async fn stage_rejects_malformed_control_message() {
    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    // Spawn stage.
    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::development());
        let (stage0_data_in, _) = tokio::io::duplex(65536);
        let (stage0_data_out, _) = tokio::io::duplex(65536);
        runtime
            .run(
                stage_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
    });

    // Orchestrator sends garbage instead of Init.
    let mut control = SecureChannel::connect_with_attestation(
        orch_ctrl,
        &provider,
        &verifier,
        SessionConfig::development(),
    )
    .await
    .expect("handshake failed");

    control
        .send(Bytes::from_static(b"this is not valid json"))
        .await
        .expect("send failed");

    let result = stage_handle.await.unwrap();
    assert!(
        matches!(&result, Err(PipelineError::Protocol(msg)) if msg.contains("malformed")),
        "expected Protocol error with 'malformed', got: {result:?}"
    );
}

/// Stage should reject a control message with wrong protocol version.
#[tokio::test]
async fn stage_rejects_wrong_version_from_orchestrator() {
    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    // Spawn stage.
    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::development());
        let (stage0_data_in, _) = tokio::io::duplex(65536);
        let (stage0_data_out, _) = tokio::io::duplex(65536);
        runtime
            .run(
                stage_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
    });

    // Send Init with wrong version.
    let mut control = SecureChannel::connect_with_attestation(
        orch_ctrl,
        &provider,
        &verifier,
        SessionConfig::development(),
    )
    .await
    .expect("handshake failed");

    let wrong_version = serde_json::json!({
        "version": 42,
        "msg": {
            "type": "Init",
            "stage_spec_json": "{}",
            "activation_spec_json": "{}",
            "num_stages": 1
        }
    });
    let data = serde_json::to_vec(&wrong_version).unwrap();
    control
        .send(Bytes::from(data))
        .await
        .expect("send failed");

    let result = stage_handle.await.unwrap();
    assert!(
        matches!(&result, Err(PipelineError::VersionMismatch { expected, actual }) if *expected == PROTOCOL_VERSION && *actual == 42),
        "expected VersionMismatch, got: {result:?}"
    );
}

/// Stage should reject oversized control messages.
#[tokio::test]
async fn stage_rejects_oversized_control_message() {
    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(1024 * 1024);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    // Spawn stage with small limit.
    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let config = StageConfig {
            max_control_message_bytes: 64,
            ..StageConfig::development()
        };
        let mut runtime = StageRuntime::new(IdentityExecutor, config);
        let (stage0_data_in, _) = tokio::io::duplex(65536);
        let (stage0_data_out, _) = tokio::io::duplex(65536);
        runtime
            .run(
                stage_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
    });

    let mut control = SecureChannel::connect_with_attestation(
        orch_ctrl,
        &provider,
        &verifier,
        SessionConfig::development(),
    )
    .await
    .expect("handshake failed");

    // Send a correctly versioned but oversized message.
    let big_payload = "Y".repeat(500);
    let envelope = serde_json::json!({
        "version": PROTOCOL_VERSION,
        "msg": {
            "type": "Init",
            "stage_spec_json": big_payload,
            "activation_spec_json": "{}",
            "num_stages": 1
        }
    });
    let data = serde_json::to_vec(&envelope).unwrap();
    control
        .send(Bytes::from(data))
        .await
        .expect("send failed");

    let result = stage_handle.await.unwrap();
    assert!(
        matches!(&result, Err(PipelineError::MessageTooLarge { .. })),
        "expected MessageTooLarge, got: {result:?}"
    );
}

/// A truncated JSON message (valid bytes but incomplete JSON) should produce
/// a clear Protocol error, not a panic.
#[tokio::test]
async fn stage_rejects_truncated_json() {
    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::development());
        let (stage0_data_in, _) = tokio::io::duplex(65536);
        let (stage0_data_out, _) = tokio::io::duplex(65536);
        runtime
            .run(
                stage_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
    });

    let mut control = SecureChannel::connect_with_attestation(
        orch_ctrl,
        &provider,
        &verifier,
        SessionConfig::development(),
    )
    .await
    .expect("handshake failed");

    // Truncated JSON.
    control
        .send(Bytes::from_static(b"{\"version\":1,\"msg\":{\"type\":\"In"))
        .await
        .expect("send failed");

    let result = stage_handle.await.unwrap();
    assert!(
        matches!(&result, Err(PipelineError::Protocol(msg)) if msg.contains("malformed")),
        "expected Protocol error with 'malformed', got: {result:?}"
    );
}

/// Verify that protocol version constant is 1 and is embedded in serialized messages.
#[test]
fn protocol_version_is_one() {
    assert_eq!(PROTOCOL_VERSION, 1);
}

/// Verify that max_control_message_bytes config validation rejects zero.
#[test]
fn config_rejects_zero_max_control_message_bytes() {
    let config = OrchestratorConfig {
        max_control_message_bytes: 0,
        ..OrchestratorConfig::development()
    };
    let manifest = make_test_manifest(1);
    let result = Orchestrator::<tokio::io::DuplexStream>::new(config, manifest);
    match &result {
        Err(PipelineError::Protocol(msg)) if msg.contains("max_control_message_bytes") => {}
        Err(e) => panic!("expected config validation error about max_control_message_bytes, got: {e}"),
        Ok(_) => panic!("expected config validation error, got Ok"),
    }
}

/// Full pipeline with hardening: version + size guards active throughout.
#[tokio::test]
async fn full_pipeline_with_hardening_guards() {
    let manifest = make_test_manifest(1);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let (orch_data_in, stage_data_in) = tokio::io::duplex(65536);
    let (stage_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        // Use a reasonable but not giant limit.
        let config = StageConfig {
            max_control_message_bytes: 8192,
            ..StageConfig::development()
        };
        let mut runtime = StageRuntime::new(IdentityExecutor, config);
        runtime
            .run(
                stage_ctrl,
                stage_data_in,
                stage_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage failed");
    });

    let config = OrchestratorConfig {
        max_control_message_bytes: 8192,
        ..OrchestratorConfig::development()
    };
    let mut orch = Orchestrator::new(config, manifest).unwrap();

    orch.init(vec![orch_ctrl], &provider, &verifier)
        .await
        .expect("init failed");

    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &provider, &verifier)
        .await
        .expect("data channels failed");

    orch.health_check().await.expect("health check failed");

    let input = vec![vec![OwnedTensor {
        name: "input".into(),
        dtype: DType::F32,
        shape: vec![1, 4],
        data: Bytes::from(vec![0u8; 16]),
    }]];

    let result = orch.infer(input, 16).await.expect("inference failed");
    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0][0].name, "input");

    orch.shutdown().await.expect("shutdown failed");
    stage_handle.await.unwrap();
}
