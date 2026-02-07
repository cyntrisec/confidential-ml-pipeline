use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, Orchestrator, OrchestratorConfig, PortSpec,
    RequestId, ShardManifest, StageConfig, StageEndpoint, StageError, StageExecutor, StageRuntime,
    StageSpec,
};

/// Executor that multiplies all tensor data bytes by 2 (simulates a "computation").
struct DoubleExecutor {
    stage_idx: usize,
}

#[async_trait]
impl StageExecutor for DoubleExecutor {
    async fn init(&mut self, stage_spec: &StageSpec) -> Result<(), StageError> {
        self.stage_idx = stage_spec.stage_idx;
        println!(
            "  [stage {}] initialized (layers {}-{})",
            stage_spec.stage_idx, stage_spec.layer_start, stage_spec.layer_end
        );
        Ok(())
    }

    async fn forward(
        &self,
        _request_id: RequestId,
        micro_batch: u32,
        inputs: Vec<OwnedTensor>,
    ) -> Result<ForwardOutput, StageError> {
        let outputs: Vec<OwnedTensor> = inputs
            .into_iter()
            .map(|t| {
                let doubled: Vec<u8> = t.data.iter().map(|b| b.wrapping_mul(2)).collect();
                println!(
                    "  [stage {}] forward mb={} tensor=\"{}\" ({} bytes)",
                    self.stage_idx,
                    micro_batch,
                    t.name,
                    doubled.len()
                );
                OwnedTensor {
                    name: t.name,
                    dtype: t.dtype,
                    shape: t.shape,
                    data: Bytes::from(doubled),
                }
            })
            .collect();
        Ok(ForwardOutput { tensors: outputs })
    }
}

fn make_manifest(num_stages: usize) -> ShardManifest {
    let layers_per_stage = 4;
    let stages = (0..num_stages)
        .map(|i| StageSpec {
            stage_idx: i,
            layer_start: i * layers_per_stage,
            layer_end: (i + 1) * layers_per_stage,
            weight_hashes: vec![],
            expected_measurements: BTreeMap::new(),
            endpoint: StageEndpoint {
                control: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 10000 + i * 10),
                },
                data_in: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 10001 + i * 10),
                },
                data_out: PortSpec::Tcp {
                    addr: format!("127.0.0.1:{}", 10002 + i * 10),
                },
            },
        })
        .collect();

    ShardManifest {
        model_name: "mock-model".into(),
        model_version: "1.0".into(),
        total_layers: num_stages * layers_per_stage,
        stages,
        activation_spec: ActivationSpec {
            dtype: ActivationDType::F32,
            hidden_dim: 4,
            max_seq_len: 16,
        },
    }
}

#[tokio::main]
async fn main() {
    let num_stages = 3;
    let num_micro_batches = 2;

    println!("=== Mock Pipeline Demo ===");
    println!("Stages: {num_stages}, Micro-batches: {num_micro_batches}");
    println!();

    let manifest = make_manifest(num_stages);

    // Create all duplex pairs.
    // Control channels.
    let mut orch_ctrls = Vec::new();
    let mut stage_ctrls = Vec::new();
    for _ in 0..num_stages {
        let (orch, stage) = tokio::io::duplex(65536);
        orch_ctrls.push(orch);
        stage_ctrls.push(stage);
    }

    // Data channels.
    // orch -> stage 0 data_in
    let (orch_data_in, stage0_data_in) = tokio::io::duplex(65536);

    // Inter-stage: stage[i] data_out -> stage[i+1] data_in
    let mut inter_stage_data_outs = Vec::new();
    let mut inter_stage_data_ins = Vec::new();
    for _ in 0..num_stages - 1 {
        let (out_side, in_side) = tokio::io::duplex(65536);
        inter_stage_data_outs.push(out_side);
        inter_stage_data_ins.push(in_side);
    }

    // last stage data_out -> orch
    let (last_stage_data_out, orch_data_out) = tokio::io::duplex(65536);

    // Spawn stages.
    let mut stage_handles = Vec::new();

    // Stage 0.
    let ctrl0 = stage_ctrls.remove(0);
    let data_out0 = inter_stage_data_outs.remove(0);
    stage_handles.push(tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(
            DoubleExecutor { stage_idx: 0 },
            StageConfig::default(),
        );
        runtime
            .run(ctrl0, stage0_data_in, data_out0, &provider, &verifier)
            .await
            .expect("stage 0 failed");
    }));

    // Middle stages.
    for i in 1..num_stages - 1 {
        let ctrl = stage_ctrls.remove(0);
        let data_in = inter_stage_data_ins.remove(0);
        let data_out = inter_stage_data_outs.remove(0);
        stage_handles.push(tokio::spawn(async move {
            let provider = MockProvider::new();
            let verifier = MockVerifier::new();
            let mut runtime = StageRuntime::new(
                DoubleExecutor { stage_idx: i },
                StageConfig::default(),
            );
            runtime
                .run(ctrl, data_in, data_out, &provider, &verifier)
                .await
                .unwrap_or_else(|e| panic!("stage {i} failed: {e}"));
        }));
    }

    // Last stage.
    let ctrl_last = stage_ctrls.remove(0);
    let data_in_last = inter_stage_data_ins.remove(0);
    let last_idx = num_stages - 1;
    stage_handles.push(tokio::spawn(async move {
        let provider = MockProvider::new();
        let verifier = MockVerifier::new();
        let mut runtime = StageRuntime::new(
            DoubleExecutor {
                stage_idx: last_idx,
            },
            StageConfig::default(),
        );
        runtime
            .run(
                ctrl_last,
                data_in_last,
                last_stage_data_out,
                &provider,
                &verifier,
            )
            .await
            .unwrap_or_else(|e| panic!("stage {last_idx} failed: {e}"));
    }));

    // Run orchestrator.
    let verifier = MockVerifier::new();
    let provider = MockProvider::new();
    let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest).unwrap();

    println!("[orch] Initializing pipeline...");
    orch.init(orch_ctrls, &verifier).await.unwrap();

    println!("[orch] Establishing data channels...");
    orch.establish_data_channels(orch_data_in, orch_data_out, vec![], &verifier, &provider)
        .await
        .unwrap();

    println!("[orch] Health check...");
    orch.health_check().await.unwrap();
    println!("[orch] All stages healthy.");
    println!();

    // Create input tensors.
    let input: Vec<Vec<OwnedTensor>> = (0..num_micro_batches)
        .map(|mb| {
            vec![OwnedTensor {
                name: format!("input_mb{mb}"),
                dtype: DType::F32,
                shape: vec![1, 4],
                data: Bytes::from(vec![1u8; 16]), // all ones
            }]
        })
        .collect();

    println!("[orch] Running inference ({num_micro_batches} micro-batches)...");
    let result = orch.infer(input, 16).await.unwrap();

    println!();
    println!("=== Results ===");
    for (mb, tensors) in result.outputs.iter().enumerate() {
        for t in tensors {
            // After 3 stages of doubling: 1 * 2 * 2 * 2 = 8
            let first_byte = t.data.first().copied().unwrap_or(0);
            println!(
                "  mb={mb} tensor=\"{}\" shape={:?} first_byte={first_byte} (expected: {})",
                t.name,
                t.shape,
                1u8.wrapping_mul(2).wrapping_mul(2).wrapping_mul(2)
            );
        }
    }

    println!();
    println!("[orch] Shutting down...");
    orch.shutdown().await.unwrap();

    for h in stage_handles {
        h.await.unwrap();
    }

    println!("[orch] Done.");
}
