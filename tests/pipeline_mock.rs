use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig, PortSpec,
    RequestId, ShardManifest, StageConfig, StageEndpoint, StageError, StageExecutor, StageRuntime,
    StageSpec,
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
                    addr: format!("127.0.0.1:{}", 9000 + i * 10),
                },
                data_in: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9001 + i * 10),
                },
                data_out: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9002 + i * 10),
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

fn make_test_tensor(name: &str) -> OwnedTensor {
    OwnedTensor {
        name: name.to_string(),
        dtype: DType::F32,
        shape: vec![1, 4],
        data: Bytes::from(vec![0u8; 16]),
    }
}

/// End-to-end 2-stage pipeline: input -> identity stage 0 -> identity stage 1 -> output.
///
/// Channel roles:
/// - control: orchestrator connects (initiator), stage accepts (responder)
/// - data_in: orchestrator/upstream stage connects (initiator), stage accepts (responder)
/// - data_out: stage connects (initiator), orchestrator/downstream accepts (responder)
#[tokio::test]
async fn two_stage_identity_pipeline() {
    let manifest = make_test_manifest(2);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    // Control channels: orchestrator initiates, stages accept.
    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(65536);

    // Data_in for stage 0: orchestrator initiates, stage 0 accepts.
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);

    // Inter-stage: stage 0 data_out (initiator) <-> stage 1 data_in (responder).
    // stage0_data_out initiates, stage1_data_in accepts.
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(65536);

    // Data_out for stage 1: stage 1 data_out (initiator) <-> orchestrator (responder).
    let (stage1_data_out, orch_data_out) = tokio::io::duplex(65536);

    // Spawn stage 0.
    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage 0 failed");
    });

    // Spawn stage 1.
    let stage1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage 1 failed");
    });

    // Run orchestrator.
    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();

    // Phase 1: Init.
    orch.init(vec![orch_ctrl0, orch_ctrl1], &verifier)
        .await
        .expect("orchestrator init failed");

    // Phase 2: Establish data channels.
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .expect("data channels failed");

    // Phase 3: Health check.
    orch.health_check().await.expect("health check failed");

    // Phase 4: Inference with 1 micro-batch.
    let input = vec![vec![make_test_tensor("input")]];
    let result = orch.infer(input, 16).await.expect("inference failed");

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].len(), 1);
    assert_eq!(result.outputs[0][0].name, "input");
    assert_eq!(result.outputs[0][0].shape, vec![1, 4]);

    // Phase 5: Shutdown.
    orch.shutdown().await.expect("shutdown failed");

    stage0_handle.await.unwrap();
    stage1_handle.await.unwrap();
}

/// Test with 2 micro-batches through a 2-stage pipeline.
#[tokio::test]
async fn two_stage_two_micro_batches() {
    let manifest = make_test_manifest(2);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(65536);
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(65536);
    let (stage1_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let stage1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    orch.init(vec![orch_ctrl0, orch_ctrl1], &verifier)
        .await
        .unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .unwrap();

    let input = vec![vec![make_test_tensor("mb0")], vec![make_test_tensor("mb1")]];
    let result = orch.infer(input, 16).await.unwrap();

    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.outputs[0][0].name, "mb0");
    assert_eq!(result.outputs[1][0].name, "mb1");

    orch.shutdown().await.unwrap();
    stage0_handle.await.unwrap();
    stage1_handle.await.unwrap();
}

/// 10 sequential inference requests through a 2-stage duplex pipeline.
#[tokio::test]
async fn sequential_inference_ten_requests() {
    let manifest = make_test_manifest(2);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(65536);
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(65536);
    let (stage1_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let stage1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    orch.init(vec![orch_ctrl0, orch_ctrl1], &verifier)
        .await
        .unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .unwrap();

    for i in 0..10 {
        let name = format!("req_{i}");
        let input = vec![vec![make_test_tensor(&name)]];
        let result = orch
            .infer(input, 16)
            .await
            .unwrap_or_else(|e| panic!("inference {i} failed: {e}"));

        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0].len(), 1);
        assert_eq!(result.outputs[0][0].name, name);
        assert_eq!(result.outputs[0][0].shape, vec![1, 4]);
    }

    orch.shutdown().await.unwrap();
    stage0_handle.await.unwrap();
    stage1_handle.await.unwrap();
}

/// 3-stage pipeline over duplex channels with 2 micro-batches.
#[tokio::test]
async fn three_stage_identity_pipeline() {
    let manifest = make_test_manifest(3);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    // Control channels.
    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(65536);
    let (orch_ctrl2, stage2_ctrl) = tokio::io::duplex(65536);

    // Data channels.
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(65536);
    let (stage1_data_out, stage2_data_in) = tokio::io::duplex(65536);
    let (stage2_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let stage1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let stage2_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(IdentityExecutor, StageConfig::default());
        runtime
            .run(
                stage2_ctrl,
                stage2_data_in,
                stage2_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap();
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    orch.init(vec![orch_ctrl0, orch_ctrl1, orch_ctrl2], &verifier)
        .await
        .unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .unwrap();

    // 2 micro-batches through 3 stages.
    let input = vec![vec![make_test_tensor("mb0")], vec![make_test_tensor("mb1")]];
    let result = orch.infer(input, 16).await.unwrap();

    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.outputs[0][0].name, "mb0");
    assert_eq!(result.outputs[1][0].name, "mb1");

    orch.shutdown().await.unwrap();
    stage0_handle.await.unwrap();
    stage1_handle.await.unwrap();
    stage2_handle.await.unwrap();
}
