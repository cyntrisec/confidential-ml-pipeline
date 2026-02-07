use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig,
    PipelineError, PortSpec, RequestId, ShardManifest, StageConfig, StageEndpoint, StageError,
    StageExecutor, StageRuntime, StageSpec,
};

/// Executor that fails on the first forward call.
struct FailingExecutor;

#[async_trait]
impl StageExecutor for FailingExecutor {
    async fn init(&mut self, _stage_spec: &StageSpec) -> Result<(), StageError> {
        Ok(())
    }

    async fn forward(
        &self,
        request_id: RequestId,
        micro_batch: u32,
        _inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        Err(StageError::ForwardFailed {
            request_id,
            micro_batch,
            reason: "intentional test failure".into(),
        })
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
                    addr: format!("127.0.0.1:{}", 9100 + i * 10),
                },
                data_in: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9101 + i * 10),
                },
                data_out: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 9102 + i * 10),
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

/// When a stage's executor fails, the orchestrator should receive a RequestFailed error.
#[tokio::test]
async fn stage_failure_returns_request_error() {
    let manifest = make_test_manifest(1);

    let (orch_ctrl0, stage0_ctrl) = tokio::io::duplex(65536);
    // data_in: orchestrator initiates, stage accepts
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);
    // data_out: stage initiates, orchestrator accepts
    let (stage0_data_out, orch_data_out) = tokio::io::duplex(65536);

    // Spawn stage 0 with FailingExecutor.
    let stage0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(FailingExecutor, StageConfig::default());
        runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await
    });

    let verifier = MockVerifier::new();
    let provider = MockProvider::new();
    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();

    orch.init(vec![orch_ctrl0], &verifier).await.unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .unwrap();

    let input = vec![vec![OwnedTensor {
        name: "input".into(),
        dtype: DType::F32,
        shape: vec![1, 4],
        data: Bytes::from(vec![0u8; 16]),
    }]];

    let result = orch.infer(input, 16).await;

    assert!(
        matches!(&result, Err(PipelineError::RequestFailed { .. })),
        "expected RequestFailed, got: {result:?}"
    );

    orch.shutdown().await.unwrap();

    let stage_result = stage0_handle.await.unwrap();
    assert!(stage_result.is_ok());
}
