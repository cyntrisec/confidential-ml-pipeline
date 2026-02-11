#![cfg(feature = "mock")]

use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig, PortSpec,
    RequestId, ShardManifest, StageConfig, StageEndpoint, StageError, StageExecutor, StageRuntime,
    StageSpec,
};

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
        model_name: "stress-model".into(),
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

/// Helper: set up a 2-stage duplex pipeline ready for inference.
/// Returns (orchestrator, stage handles).
async fn setup_two_stage() -> (
    Orchestrator<tokio::io::DuplexStream>,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
) {
    let manifest = make_test_manifest(2);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(262144);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(262144);
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(262144);
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(262144);
    let (stage1_data_out, orch_data_out) = tokio::io::duplex(262144);

    let s0 = tokio::spawn(async move {
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

    let s1 = tokio::spawn(async move {
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

    (orch, s0, s1)
}

/// Helper: set up a 3-stage duplex pipeline ready for inference.
async fn setup_three_stage() -> (
    Orchestrator<tokio::io::DuplexStream>,
    Vec<tokio::task::JoinHandle<()>>,
) {
    let manifest = make_test_manifest(3);
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(262144);
    let (orch_ctrl1, stage1_ctrl) = tokio::io::duplex(262144);
    let (orch_ctrl2, stage2_ctrl) = tokio::io::duplex(262144);

    let (orch_data_in, stage0_data_in) = tokio::io::duplex(262144);
    let (stage0_data_out, stage1_data_in) = tokio::io::duplex(262144);
    let (stage1_data_out, stage2_data_in) = tokio::io::duplex(262144);
    let (stage2_data_out, orch_data_out) = tokio::io::duplex(262144);

    let s0 = tokio::spawn(async move {
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

    let s1 = tokio::spawn(async move {
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

    let s2 = tokio::spawn(async move {
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

    (orch, vec![s0, s1, s2])
}

/// 100 sequential inference requests through a 2-stage pipeline.
#[tokio::test]
async fn stress_sequential_100_requests() {
    let (mut orch, s0, s1) = setup_two_stage().await;

    for i in 0..100 {
        let name = format!("req_{i}");
        let input = vec![vec![make_test_tensor(&name)]];
        let result = orch
            .infer(input, 16)
            .await
            .unwrap_or_else(|e| panic!("inference {i} failed: {e}"));

        assert_eq!(result.outputs.len(), 1);
        assert_eq!(result.outputs[0][0].name, name);
    }

    orch.shutdown().await.unwrap();
    s0.await.unwrap();
    s1.await.unwrap();
}

/// Single inference with a 1.5 MiB tensor through a 2-stage pipeline.
#[tokio::test]
async fn stress_large_tensors() {
    let (mut orch, s0, s1) = setup_two_stage().await;

    let large_data = vec![0u8; 1536 * 1024]; // 1.5 MiB = 384*1024 F32 elements
    let tensor = OwnedTensor {
        name: "large".to_string(),
        dtype: DType::F32,
        shape: vec![1, 384 * 1024],
        data: Bytes::from(large_data),
    };

    let input = vec![vec![tensor]];
    let result = orch
        .infer(input, 16)
        .await
        .expect("large tensor inference failed");

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].len(), 1);
    assert_eq!(result.outputs[0][0].name, "large");
    assert_eq!(result.outputs[0][0].data.len(), 1536 * 1024);

    orch.shutdown().await.unwrap();
    s0.await.unwrap();
    s1.await.unwrap();
}

/// Single inference with 16 micro-batches through a 2-stage pipeline.
#[tokio::test]
async fn stress_many_micro_batches() {
    let (mut orch, s0, s1) = setup_two_stage().await;

    let input: Vec<Vec<OwnedTensor>> = (0..16)
        .map(|i| vec![make_test_tensor(&format!("mb_{i}"))])
        .collect();

    let result = orch.infer(input, 16).await.expect("16-mb inference failed");

    assert_eq!(result.outputs.len(), 16);
    for (i, mb_out) in result.outputs.iter().enumerate() {
        assert_eq!(mb_out.len(), 1);
        assert_eq!(mb_out[0].name, format!("mb_{i}"));
    }

    orch.shutdown().await.unwrap();
    s0.await.unwrap();
    s1.await.unwrap();
}

/// 3-stage pipeline, 5 sequential requests with 4 micro-batches each.
#[tokio::test]
async fn stress_three_stage_multi_request_multi_batch() {
    let (mut orch, handles) = setup_three_stage().await;

    for req in 0..5 {
        let input: Vec<Vec<OwnedTensor>> = (0..4)
            .map(|mb| vec![make_test_tensor(&format!("r{req}_mb{mb}"))])
            .collect();

        let result = orch
            .infer(input, 16)
            .await
            .unwrap_or_else(|e| panic!("request {req} failed: {e}"));

        assert_eq!(result.outputs.len(), 4);
        for (mb, mb_out) in result.outputs.iter().enumerate() {
            assert_eq!(mb_out.len(), 1);
            assert_eq!(mb_out[0].name, format!("r{req}_mb{mb}"));
        }
    }

    orch.shutdown().await.unwrap();
    for h in handles {
        h.await.unwrap();
    }
}
