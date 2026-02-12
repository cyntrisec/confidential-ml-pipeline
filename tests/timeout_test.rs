#![cfg(feature = "mock")]

use std::collections::BTreeMap;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig,
    PipelineError, PortSpec, RequestId, ShardManifest, StageConfig, StageEndpoint, StageError,
    StageExecutor, StageRuntime, StageSpec,
};

/// Executor that sleeps for a configurable duration before returning inputs unchanged.
struct SlowExecutor(Duration);

#[async_trait]
impl StageExecutor for SlowExecutor {
    async fn init(&mut self, _stage_spec: &StageSpec) -> Result<(), StageError> {
        Ok(())
    }

    async fn forward(
        &self,
        _request_id: RequestId,
        _micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        tokio::time::sleep(self.0).await;
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
        model_name: "timeout-test".into(),
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

/// Set up a 2-stage pipeline with the given executor delay, ready for inference.
async fn setup_slow_pipeline(
    delay: Duration,
) -> (
    Orchestrator<tokio::io::DuplexStream>,
    tokio::task::JoinHandle<()>,
    tokio::task::JoinHandle<()>,
) {
    setup_slow_pipeline_with_config(delay, OrchestratorConfig::default()).await
}

/// Set up a 2-stage pipeline with custom config and executor delay.
async fn setup_slow_pipeline_with_config(
    delay: Duration,
    config: OrchestratorConfig,
) -> (
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

    let delay0 = delay;
    let s0 = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(SlowExecutor(delay0), StageConfig::default());
        let _ = runtime
            .run(
                stage0_ctrl,
                stage0_data_in,
                stage0_data_out,
                &provider,
                &verifier,
            )
            .await;
    });

    let delay1 = delay;
    let s1 = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(SlowExecutor(delay1), StageConfig::default());
        let _ = runtime
            .run(
                stage1_ctrl,
                stage1_data_in,
                stage1_data_out,
                &provider,
                &verifier,
            )
            .await;
    });

    let mut orch = Orchestrator::new(config, manifest).unwrap();
    orch.init(vec![orch_ctrl0, orch_ctrl1], &provider, &verifier)
        .await
        .unwrap();
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &provider, &verifier)
        .await
        .unwrap();

    (orch, s0, s1)
}

/// Test 1: Infer timeout triggers drain, then recovery with a longer timeout.
///
/// SlowExecutor(500ms) with 100ms infer_timeout: first infer times out.
/// After drain (~500ms), pipeline recovers with a 5s timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_timeout_drain_and_recovery() {
    let (mut orch, s0, s1) = setup_slow_pipeline(Duration::from_millis(500)).await;

    // Set a very short timeout to trigger timeout.
    orch.set_infer_timeout(Duration::from_millis(100));

    let input = vec![vec![make_test_tensor("timeout_req")]];
    let result = orch.infer(input, 16).await;

    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(
        matches!(err, PipelineError::Timeout(_)),
        "expected Timeout, got {err:?}"
    );

    // Pipeline should NOT be tainted (stages finish within 5s drain timeout).
    assert!(
        !orch.is_tainted(),
        "pipeline should not be tainted after successful drain"
    );

    // Increase timeout for recovery.
    orch.set_infer_timeout(Duration::from_secs(5));

    let input = vec![vec![make_test_tensor("recovery_req")]];
    let result = orch.infer(input, 16).await;
    assert!(result.is_ok(), "recovery infer failed: {result:?}");

    let output = result.unwrap();
    assert_eq!(output.outputs.len(), 1);
    assert_eq!(output.outputs[0][0].name, "recovery_req");

    orch.shutdown().await.unwrap();
    s0.await.unwrap();
    s1.await.unwrap();
}

/// Test 2: Infer timeout with stages stuck beyond drain timeout -> pipeline tainted.
///
/// SlowExecutor(60s) with 50ms infer_timeout: first infer times out.
/// Drain waits 5s per stage, stages still stuck -> tainted.
/// Second infer returns PipelineError::Tainted immediately.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn infer_timeout_tainted_pipeline() {
    let (mut orch, _s0, _s1) = setup_slow_pipeline(Duration::from_secs(60)).await;

    orch.set_infer_timeout(Duration::from_millis(50));

    let input = vec![vec![make_test_tensor("stuck_req")]];
    let result = orch.infer(input, 16).await;

    assert!(result.is_err());
    assert!(
        matches!(result.unwrap_err(), PipelineError::Timeout(_)),
        "expected Timeout"
    );

    // Pipeline should be tainted (stages stuck for > 5s drain timeout).
    assert!(
        orch.is_tainted(),
        "pipeline should be tainted after drain failure"
    );

    // Any further operation should return Tainted.
    let input = vec![vec![make_test_tensor("after_taint")]];
    let result = orch.infer(input, 16).await;
    assert!(
        matches!(result, Err(PipelineError::Tainted)),
        "expected Tainted, got {result:?}"
    );

    let hc = orch.health_check().await;
    assert!(
        matches!(hc, Err(PipelineError::Tainted)),
        "expected Tainted from health_check, got {hc:?}"
    );

    // Stage tasks are still running (60s sleep); they'll be dropped when handles drop.
}

/// Test 3: Health check succeeds after an infer timeout, skipping stale messages.
///
/// SlowExecutor(500ms) with 100ms infer_timeout: first infer times out.
/// After drain, health check works (tolerant reader skips stale messages).
/// Then a normal infer also succeeds.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn health_check_after_infer_timeout() {
    let (mut orch, s0, s1) = setup_slow_pipeline(Duration::from_millis(500)).await;

    orch.set_infer_timeout(Duration::from_millis(100));

    let input = vec![vec![make_test_tensor("timeout_req")]];
    let result = orch.infer(input, 16).await;
    assert!(matches!(result, Err(PipelineError::Timeout(_))));

    assert!(!orch.is_tainted());

    // Increase timeout for subsequent operations.
    orch.set_infer_timeout(Duration::from_secs(5));

    // Health check should succeed, skipping any stale messages.
    orch.health_check()
        .await
        .expect("health check should succeed after timeout drain");

    // Normal infer should also succeed.
    let input = vec![vec![make_test_tensor("after_hc")]];
    let result = orch.infer(input, 16).await;
    assert!(
        result.is_ok(),
        "infer after health check failed: {result:?}"
    );
    assert_eq!(result.unwrap().outputs[0][0].name, "after_hc");

    orch.shutdown().await.unwrap();
    s0.await.unwrap();
    s1.await.unwrap();
}

/// Test 4: Data-out drain timeout taints the pipeline.
///
/// Stages complete quickly (control drain succeeds), but data_quiet_period is
/// set very long (5s) so the drain blocks waiting for more data after reading
/// the buffered output. The short data_drain_timeout (10ms) fires during this
/// wait, triggering the taint path.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn data_out_drain_timeout_taints_pipeline() {
    let config = OrchestratorConfig {
        infer_timeout: Duration::from_millis(50),
        // Stages finish within this window.
        stage_drain_timeout: Duration::from_secs(5),
        // Long quiet period: drain blocks after reading buffered output.
        data_quiet_period: Duration::from_secs(5),
        // Short outer bound: fires while drain is blocked in quiet-period wait.
        data_drain_timeout: Duration::from_millis(10),
        ..OrchestratorConfig::default()
    };

    // SlowExecutor(300ms): stages complete in ~300ms per micro-batch (within
    // stage_drain_timeout), but after control drain succeeds, the data_out
    // drain reads the buffered output then blocks for 5s quiet period. The
    // 10ms data_drain_timeout fires, tainting the pipeline.
    let (mut orch, _s0, _s1) =
        setup_slow_pipeline_with_config(Duration::from_millis(300), config).await;

    let input = vec![vec![make_test_tensor("data_drain_req")]];
    let result = orch.infer(input, 16).await;

    assert!(
        matches!(result, Err(PipelineError::Timeout(_))),
        "expected Timeout, got {result:?}"
    );

    // Pipeline should be tainted because data_out drain timed out.
    assert!(
        orch.is_tainted(),
        "pipeline should be tainted after data_out drain timeout"
    );

    // Further operations should be rejected.
    let input = vec![vec![make_test_tensor("after_data_taint")]];
    let result = orch.infer(input, 16).await;
    assert!(
        matches!(result, Err(PipelineError::Tainted)),
        "expected Tainted, got {result:?}"
    );

    let hc = orch.health_check().await;
    assert!(
        matches!(hc, Err(PipelineError::Tainted)),
        "expected Tainted from health_check, got {hc:?}"
    );
}
