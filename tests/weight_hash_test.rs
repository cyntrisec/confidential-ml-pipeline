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

/// Executor that returns configurable weight hashes.
struct HashableExecutor {
    hashes: Vec<String>,
}

#[async_trait]
impl StageExecutor for HashableExecutor {
    async fn init(&mut self, _stage_spec: &StageSpec) -> Result<(), StageError> {
        Ok(())
    }

    fn weight_hashes(&self) -> Vec<String> {
        self.hashes.clone()
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

fn make_manifest_with_hashes(hashes: Vec<String>) -> ShardManifest {
    ShardManifest {
        model_name: "test-model".into(),
        model_version: "1.0".into(),
        total_layers: 4,
        stages: vec![StageSpec {
            stage_idx: 0,
            layer_start: 0,
            layer_end: 4,
            weight_hashes: hashes,
            expected_measurements: BTreeMap::new(),
            endpoint: StageEndpoint {
                control: PortSpec::Tcp {
                    addr: "127.0.0.1:9000".to_string(),
                },
                data_in: PortSpec::Tcp {
                    addr: "127.0.0.1:9001".to_string(),
                },
                data_out: PortSpec::Tcp {
                    addr: "127.0.0.1:9002".to_string(),
                },
            },
        }],
        activation_spec: ActivationSpec {
            dtype: ActivationDType::F32,
            hidden_dim: 4,
            max_seq_len: 16,
        },
    }
}

/// Weight hashes match — stage should initialize successfully.
#[tokio::test]
async fn weight_hashes_match_passes() {
    let hash = "abc123".to_string();
    let manifest = make_manifest_with_hashes(vec![hash.clone()]);

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let (orch_data_in, stage_data_in) = tokio::io::duplex(65536);
    let (stage_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let executor = HashableExecutor {
            hashes: vec![hash],
        };
        let mut runtime = StageRuntime::new(executor, StageConfig::default());
        runtime
            .run(
                stage_ctrl,
                stage_data_in,
                stage_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage should succeed with matching hashes");
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    orch.init(vec![orch_ctrl], &provider, &verifier).await.unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &provider, &verifier)
        .await
        .unwrap();

    let input = vec![vec![OwnedTensor {
        name: "x".into(),
        dtype: DType::F32,
        shape: vec![1, 4],
        data: Bytes::from(vec![0u8; 16]),
    }]];
    let result = orch.infer(input, 16).await.unwrap();
    assert_eq!(result.outputs.len(), 1);

    orch.shutdown().await.unwrap();
    stage_handle.await.unwrap();
}

/// Weight hashes mismatch — stage should fail during control phase.
#[tokio::test]
async fn weight_hashes_mismatch_fails() {
    let manifest = make_manifest_with_hashes(vec!["expected_hash".to_string()]);

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let executor = HashableExecutor {
            hashes: vec!["wrong_hash".to_string()],
        };
        let mut runtime = StageRuntime::new(executor, StageConfig::default());
        let result = runtime.run_control_phase(stage_ctrl, &provider, &verifier).await;
        assert!(result.is_err(), "stage should fail with mismatched hashes");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("weight hash mismatch"),
            "error should mention weight hash mismatch, got: {err}"
        );
    });

    // Orchestrator init will fail because stage errors out.
    let mut orch =
        Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    let result = orch.init(vec![orch_ctrl], &provider, &verifier).await;
    // This should fail because the stage side closes the channel.
    assert!(result.is_err());

    stage_handle.await.unwrap();
}

/// Weight hash count mismatch — stage should fail.
#[tokio::test]
async fn weight_hashes_count_mismatch_fails() {
    let manifest = make_manifest_with_hashes(vec!["h1".into(), "h2".into()]);

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let executor = HashableExecutor {
            hashes: vec!["h1".into()], // only 1 hash, manifest expects 2
        };
        let mut runtime = StageRuntime::new(executor, StageConfig::default());
        let result = runtime.run_control_phase(stage_ctrl, &provider, &verifier).await;
        assert!(result.is_err(), "stage should fail with count mismatch");
        let err = result.err().unwrap().to_string();
        assert!(
            err.contains("count mismatch"),
            "error should mention count mismatch, got: {err}"
        );
    });

    let mut orch =
        Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    let result = orch.init(vec![orch_ctrl], &provider, &verifier).await;
    assert!(result.is_err());

    stage_handle.await.unwrap();
}

/// Empty weight_hashes in manifest — verification skipped, stage passes.
#[tokio::test]
async fn empty_weight_hashes_skips_verification() {
    let manifest = make_manifest_with_hashes(vec![]); // no hashes declared

    let provider = MockProvider::new();
    let verifier = MockVerifier::new();

    let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
    let (orch_data_in, stage_data_in) = tokio::io::duplex(65536);
    let (stage_data_out, orch_data_out) = tokio::io::duplex(65536);

    let stage_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        // Default weight_hashes() returns empty — should be fine.
        let executor = HashableExecutor { hashes: vec![] };
        let mut runtime = StageRuntime::new(executor, StageConfig::default());
        runtime
            .run(
                stage_ctrl,
                stage_data_in,
                stage_data_out,
                &provider,
                &verifier,
            )
            .await
            .expect("stage should pass with no hashes declared");
    });

    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();
    orch.init(vec![orch_ctrl], &provider, &verifier).await.unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &provider, &verifier)
        .await
        .unwrap();

    orch.shutdown().await.unwrap();
    stage_handle.await.unwrap();
}
