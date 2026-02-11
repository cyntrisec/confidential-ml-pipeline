#![cfg(all(feature = "tcp", feature = "mock"))]

use std::collections::BTreeMap;
use std::net::SocketAddr;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::tcp;
use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, OrchestratorConfig, PortSpec, RequestId,
    ShardManifest, StageConfig, StageEndpoint, StageError, StageExecutor, StageSpec,
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

fn make_test_tensor(name: &str) -> OwnedTensor {
    OwnedTensor {
        name: name.to_string(),
        dtype: DType::F32,
        shape: vec![1, 4],
        data: Bytes::from(vec![1u8; 16]),
    }
}

/// Build a manifest whose endpoint addresses match the actual bound listeners.
fn make_manifest_with_addrs(
    stage_addrs: &[(SocketAddr, SocketAddr)], // (control_addr, data_in_addr) per stage
) -> ShardManifest {
    let num_stages = stage_addrs.len();
    let stages = stage_addrs
        .iter()
        .enumerate()
        .map(|(i, (ctrl, din))| StageSpec {
            stage_idx: i,
            layer_start: i * 4,
            layer_end: (i + 1) * 4,
            weight_hashes: vec![],
            expected_measurements: BTreeMap::new(),
            endpoint: StageEndpoint {
                control: PortSpec::Tcp {
                    addr: ctrl.to_string(),
                },
                data_in: PortSpec::Tcp {
                    addr: din.to_string(),
                },
                // data_out is stage-initiated, not used in manifest for connection
                data_out: PortSpec::Tcp {
                    addr: "0.0.0.0:0".to_string(),
                },
            },
        })
        .collect();

    ShardManifest {
        model_name: "tcp-test-model".into(),
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

/// Two-stage pipeline over real TCP with IdentityExecutor.
#[tokio::test]
async fn two_stage_tcp_pipeline() {
    let localhost: SocketAddr = "127.0.0.1:0".parse().unwrap();

    // Bind listeners for both stages.
    let (s0_ctrl_lis, s0_ctrl_addr, s0_din_lis, s0_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();
    let (s1_ctrl_lis, s1_ctrl_addr, s1_din_lis, s1_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();

    let manifest =
        make_manifest_with_addrs(&[(s0_ctrl_addr, s0_din_addr), (s1_ctrl_addr, s1_din_addr)]);

    // Bind orchestrator data_out listener.
    let orch_dout_lis = tokio::net::TcpListener::bind(localhost).await.unwrap();
    let orch_dout_addr = orch_dout_lis.local_addr().unwrap();

    // Stage 0: data_out goes to stage 1's data_in.
    let s0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s0_ctrl_lis,
            s0_din_lis,
            s1_din_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 0 failed");
    });

    // Stage 1: data_out goes to orchestrator's data_out listener.
    let s1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s1_ctrl_lis,
            s1_din_lis,
            orch_dout_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 1 failed");
    });

    // Give stages a moment to start their accept loops.
    tokio::time::sleep(Duration::from_millis(20)).await;

    // Orchestrator.
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let mut orch = tcp::init_orchestrator_tcp(
        OrchestratorConfig::default(),
        manifest,
        orch_dout_lis,
        &verifier,
        &provider,
    )
    .await
    .expect("orchestrator init failed");

    // Health check.
    orch.health_check().await.expect("health check failed");

    // Inference.
    let input = vec![vec![make_test_tensor("tcp_input")]];
    let result = orch.infer(input, 16).await.expect("inference failed");

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0].len(), 1);
    assert_eq!(result.outputs[0][0].name, "tcp_input");

    // Shutdown.
    orch.shutdown().await.expect("shutdown failed");
    s0_handle.await.unwrap();
    s1_handle.await.unwrap();
}

/// Single-stage degenerate pipeline over TCP.
#[tokio::test]
async fn single_stage_tcp_pipeline() {
    let localhost: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let (s0_ctrl_lis, s0_ctrl_addr, s0_din_lis, s0_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();

    let manifest = make_manifest_with_addrs(&[(s0_ctrl_addr, s0_din_addr)]);

    let orch_dout_lis = tokio::net::TcpListener::bind(localhost).await.unwrap();
    let orch_dout_addr = orch_dout_lis.local_addr().unwrap();

    // Stage 0: data_out goes directly to orchestrator's data_out listener.
    let s0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s0_ctrl_lis,
            s0_din_lis,
            orch_dout_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 0 failed");
    });

    tokio::time::sleep(Duration::from_millis(20)).await;

    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let mut orch = tcp::init_orchestrator_tcp(
        OrchestratorConfig::default(),
        manifest,
        orch_dout_lis,
        &verifier,
        &provider,
    )
    .await
    .expect("orchestrator init failed");

    let input = vec![vec![make_test_tensor("single")]];
    let result = orch.infer(input, 16).await.expect("inference failed");

    assert_eq!(result.outputs.len(), 1);
    assert_eq!(result.outputs[0][0].name, "single");

    orch.shutdown().await.expect("shutdown failed");
    s0_handle.await.unwrap();
}

/// Three-stage pipeline over TCP — validates cascade through 2 intermediate data links.
#[tokio::test]
async fn three_stage_tcp_pipeline() {
    let localhost: SocketAddr = "127.0.0.1:0".parse().unwrap();

    let (s0_ctrl_lis, s0_ctrl_addr, s0_din_lis, s0_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();
    let (s1_ctrl_lis, s1_ctrl_addr, s1_din_lis, s1_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();
    let (s2_ctrl_lis, s2_ctrl_addr, s2_din_lis, s2_din_addr) =
        tcp::bind_stage_listeners(localhost, localhost)
            .await
            .unwrap();

    let manifest = make_manifest_with_addrs(&[
        (s0_ctrl_addr, s0_din_addr),
        (s1_ctrl_addr, s1_din_addr),
        (s2_ctrl_addr, s2_din_addr),
    ]);

    let orch_dout_lis = tokio::net::TcpListener::bind(localhost).await.unwrap();
    let orch_dout_addr = orch_dout_lis.local_addr().unwrap();

    // Stage 0 → stage 1 data_in.
    let s0_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s0_ctrl_lis,
            s0_din_lis,
            s1_din_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 0 failed");
    });

    // Stage 1 → stage 2 data_in.
    let s1_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s1_ctrl_lis,
            s1_din_lis,
            s2_din_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 1 failed");
    });

    // Stage 2 → orchestrator data_out.
    let s2_handle = tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        tcp::run_stage_with_listeners(
            IdentityExecutor,
            StageConfig::default(),
            s2_ctrl_lis,
            s2_din_lis,
            orch_dout_addr,
            &provider,
            &verifier,
        )
        .await
        .expect("stage 2 failed");
    });

    tokio::time::sleep(Duration::from_millis(20)).await;

    let verifier = MockVerifier::new();
    let provider = MockProvider::new();

    let mut orch = tcp::init_orchestrator_tcp(
        OrchestratorConfig::default(),
        manifest,
        orch_dout_lis,
        &verifier,
        &provider,
    )
    .await
    .expect("orchestrator init failed");

    orch.health_check().await.expect("health check failed");

    // First inference: 2 micro-batches through 3 stages.
    let input = vec![vec![make_test_tensor("mb0")], vec![make_test_tensor("mb1")]];
    let result = orch.infer(input, 16).await.expect("inference failed");

    assert_eq!(result.outputs.len(), 2);
    assert_eq!(result.outputs[0][0].name, "mb0");
    assert_eq!(result.outputs[1][0].name, "mb1");

    // Second inference: verify sequential requests work over TCP.
    let input2 = vec![
        vec![make_test_tensor("seq0")],
        vec![make_test_tensor("seq1")],
    ];
    let result2 = orch
        .infer(input2, 16)
        .await
        .expect("second inference failed");

    assert_eq!(result2.outputs.len(), 2);
    assert_eq!(result2.outputs[0][0].name, "seq0");
    assert_eq!(result2.outputs[1][0].name, "seq1");

    orch.shutdown().await.expect("shutdown failed");
    s0_handle.await.unwrap();
    s1_handle.await.unwrap();
    s2_handle.await.unwrap();
}
