use std::collections::BTreeMap;

use async_trait::async_trait;
use bytes::Bytes;
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use tokio::io::{AsyncReadExt, AsyncWriteExt};

use confidential_ml_transport::{DType, MockProvider, MockVerifier, OwnedTensor};

use confidential_ml_pipeline::{
    ActivationDType, ActivationSpec, ForwardOutput, InferenceSchedule, Orchestrator,
    OrchestratorConfig, OrchestratorMsg, PortSpec, RequestId, ShardManifest, StageConfig,
    StageEndpoint, StageError, StageExecutor, StageMsg, StageRuntime, StageSpec,
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
        model_name: "bench-model".into(),
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

fn make_tensor(name: &str, data_size: usize) -> OwnedTensor {
    OwnedTensor {
        name: name.to_string(),
        dtype: DType::F32,
        shape: vec![1, (data_size / 4) as u32],
        data: Bytes::from(vec![0u8; data_size]),
    }
}

/// Build a fully-initialized 2-stage duplex pipeline ready for inference.
async fn setup_two_stage_pipeline() -> (
    Orchestrator<tokio::io::DuplexStream>,
    Vec<tokio::task::JoinHandle<()>>,
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

    (orch, vec![s0, s1])
}

/// Build a fully-initialized 3-stage duplex pipeline ready for inference.
async fn setup_three_stage_pipeline() -> (
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

// ---------------------------------------------------------------------------
// 1. Pipeline throughput: single-request latency at different tensor sizes
// ---------------------------------------------------------------------------

fn bench_pipeline_throughput(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("pipeline_throughput");

    for &size in &[64, 1024, 16384, 131072] {
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(
            BenchmarkId::new("2stage_1mb", format!("{size}B")),
            &size,
            |b, &size| {
                // Setup pipeline once per benchmark group iteration.
                let (mut orch, handles) = rt.block_on(setup_two_stage_pipeline());

                b.iter(|| {
                    rt.block_on(async {
                        let input = vec![vec![make_tensor("bench", size)]];
                        let result = orch.infer(input, 16).await.unwrap();
                        black_box(result);
                    })
                });

                rt.block_on(async {
                    orch.shutdown().await.unwrap();
                    for h in handles {
                        h.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 2. Latency per stage: compare 2-stage vs 3-stage to isolate per-stage cost
// ---------------------------------------------------------------------------

fn bench_latency_per_stage(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("latency_per_stage");

    // 2-stage pipeline, 1KB tensor
    group.bench_function("2_stages_1KB", |b| {
        let (mut orch, handles) = rt.block_on(setup_two_stage_pipeline());

        b.iter(|| {
            rt.block_on(async {
                let input = vec![vec![make_tensor("bench", 1024)]];
                let result = orch.infer(input, 16).await.unwrap();
                black_box(result);
            })
        });

        rt.block_on(async {
            orch.shutdown().await.unwrap();
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    // 3-stage pipeline, 1KB tensor
    group.bench_function("3_stages_1KB", |b| {
        let (mut orch, handles) = rt.block_on(setup_three_stage_pipeline());

        b.iter(|| {
            rt.block_on(async {
                let input = vec![vec![make_tensor("bench", 1024)]];
                let result = orch.infer(input, 16).await.unwrap();
                black_box(result);
            })
        });

        rt.block_on(async {
            orch.shutdown().await.unwrap();
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 3. Scheduling overhead: pure schedule generation (no I/O)
// ---------------------------------------------------------------------------

fn bench_scheduling_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("scheduling");

    for &(stages, batches) in &[(2, 4), (4, 16), (8, 64), (16, 128)] {
        group.bench_with_input(
            BenchmarkId::new("generate", format!("{stages}s_{batches}mb")),
            &(stages, batches),
            |b, &(stages, batches)| {
                b.iter(|| {
                    let schedule = InferenceSchedule::generate(stages, batches).unwrap();
                    black_box(schedule);
                })
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 4. Relay overhead: measure raw relay forwarding throughput
// ---------------------------------------------------------------------------

fn bench_relay_overhead(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("relay");

    for &chunk_size in &[1024, 16384, 131072] {
        group.throughput(Throughput::Bytes(chunk_size as u64));
        group.bench_with_input(
            BenchmarkId::new("forward", format!("{chunk_size}B")),
            &chunk_size,
            |b, &chunk_size| {
                b.iter(|| {
                    rt.block_on(async {
                        let (client, relay_left) = tokio::io::duplex(chunk_size * 2);
                        let (relay_right, server) = tokio::io::duplex(chunk_size * 2);

                        let handle =
                            confidential_ml_pipeline::start_relay_link(relay_left, relay_right);

                        let data = vec![0xABu8; chunk_size];
                        let (client_read, mut client_write) = tokio::io::split(client);
                        let (mut server_read, mut server_write) = tokio::io::split(server);

                        // client → relay → server
                        let send_task = tokio::spawn(async move {
                            client_write.write_all(&data).await.unwrap();
                            client_write.shutdown().await.unwrap();
                        });

                        let recv_task = tokio::spawn(async move {
                            let mut buf = vec![0u8; chunk_size];
                            let mut total = 0;
                            while total < chunk_size {
                                let n = server_read.read(&mut buf[total..]).await.unwrap();
                                if n == 0 {
                                    break;
                                }
                                total += n;
                            }
                            black_box(total);
                            // Clean up server side
                            server_write.shutdown().await.ok();
                        });

                        send_task.await.unwrap();
                        recv_task.await.unwrap();

                        drop(client_read);
                        handle.abort();
                    })
                });
            },
        );
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// 5. Protocol message serialization overhead
// ---------------------------------------------------------------------------

fn bench_protocol_serde(c: &mut Criterion) {
    let mut group = c.benchmark_group("protocol_serde");

    let start_req = OrchestratorMsg::StartRequest {
        request_id: 12345678,
        num_micro_batches: 16,
        seq_len: 512,
    };
    let start_req_bytes = start_req.to_bytes();

    group.bench_function("serialize_StartRequest", |b| {
        b.iter(|| {
            let bytes = start_req.to_bytes();
            black_box(bytes);
        })
    });

    group.bench_function("deserialize_StartRequest", |b| {
        b.iter(|| {
            let msg = OrchestratorMsg::from_bytes(&start_req_bytes).unwrap();
            black_box(msg);
        })
    });

    let request_done = StageMsg::RequestDone {
        request_id: 12345678,
    };
    let request_done_bytes = request_done.to_bytes();

    group.bench_function("serialize_RequestDone", |b| {
        b.iter(|| {
            let bytes = request_done.to_bytes();
            black_box(bytes);
        })
    });

    group.bench_function("deserialize_RequestDone", |b| {
        b.iter(|| {
            let msg = StageMsg::from_bytes(&request_done_bytes).unwrap();
            black_box(msg);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 6. Health check latency
// ---------------------------------------------------------------------------

fn bench_health_check(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("health_check");

    group.bench_function("2_stages", |b| {
        let (mut orch, handles) = rt.block_on(setup_two_stage_pipeline());

        b.iter(|| {
            rt.block_on(async {
                orch.health_check().await.unwrap();
            })
        });

        rt.block_on(async {
            orch.shutdown().await.unwrap();
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.bench_function("3_stages", |b| {
        let (mut orch, handles) = rt.block_on(setup_three_stage_pipeline());

        b.iter(|| {
            rt.block_on(async {
                orch.health_check().await.unwrap();
            })
        });

        rt.block_on(async {
            orch.shutdown().await.unwrap();
            for h in handles {
                h.await.unwrap();
            }
        });
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// 7. Multi-micro-batch throughput
// ---------------------------------------------------------------------------

fn bench_multi_micro_batch(c: &mut Criterion) {
    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .unwrap();

    let mut group = c.benchmark_group("multi_micro_batch");

    for &num_mb in &[1, 4, 8, 16] {
        group.bench_with_input(
            BenchmarkId::new("2stage", format!("{num_mb}mb")),
            &num_mb,
            |b, &num_mb| {
                let (mut orch, handles) = rt.block_on(setup_two_stage_pipeline());

                b.iter(|| {
                    rt.block_on(async {
                        let input: Vec<Vec<OwnedTensor>> = (0..num_mb)
                            .map(|i| vec![make_tensor(&format!("mb_{i}"), 1024)])
                            .collect();
                        let result = orch.infer(input, 16).await.unwrap();
                        black_box(result);
                    })
                });

                rt.block_on(async {
                    orch.shutdown().await.unwrap();
                    for h in handles {
                        h.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_pipeline_throughput,
    bench_latency_per_stage,
    bench_scheduling_overhead,
    bench_relay_overhead,
    bench_protocol_serde,
    bench_health_check,
    bench_multi_micro_batch,
);
criterion_main!(benches);
