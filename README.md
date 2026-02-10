# confidential-ml-pipeline

Multi-enclave pipeline orchestration for confidential ML inference.

> **Disclaimer:** This project is under development. All source code and features are not production ready.

## Overview

`confidential-ml-pipeline` builds on top of [`confidential-ml-transport`](https://github.com/cyntrisec/confidential-ml-transport) to orchestrate model-parallel inference across multiple TEE enclaves. A model is sharded across N stages, each running inside a separate enclave, and the orchestrator drives input tensors through the pipeline while maintaining encrypted, attestation-bound channels between all participants.

**Key properties:**

- **Pipeline parallelism** -- 1F1B (one forward, one backward) fill-drain scheduling with configurable micro-batching to minimize pipeline bubbles
- **Shard manifest** -- JSON-based model sharding specification with layer ranges, weight hashes, and expected attestation measurements per stage
- **Two-phase APIs** -- `StageRuntime` and `Orchestrator` expose split control/data phases for TCP deployment where connections arrive at different times
- **Configurable timeouts** -- per-operation timeouts for health checks (default 10s) and inference requests (default 60s), surfaced as `PipelineError::Timeout`
- **Retry policy** -- TCP connection retries use the transport crate's `RetryPolicy` with exponential backoff and jitter, configurable on both `OrchestratorConfig` and `StageConfig`
- **TCP deployment helpers** -- `tcp` module with retry-connect, listener binding, and full stage/orchestrator lifecycle over real TCP
- **Pluggable transports** -- TCP and VSock backends via feature flags, with `tokio::io::duplex` for in-process testing
- **Pluggable attestation** -- trait-based attestation, mock for development, Nitro for production
- **Relay mesh** -- transparent bidirectional byte relay for inter-stage data channels through the host
- **Error propagation** -- stage failures send error sentinels on data channels to unblock the pipeline, with detailed error reporting on control channels

## Architecture

```
                    control channels (SecureChannel)
                    ┌──────────┬──────────┐
                    │          │          │
              ┌─────▼───┐ ┌───▼─────┐ ┌──▼──────┐
  input ────► │ Stage 0  │ │ Stage 1 │ │ Stage 2 │ ────► output
  tensors     │ layers   │ │ layers  │ │ layers  │       tensors
              │  0..3    │ │  4..7   │ │  8..11  │
              └────┬─────┘ └──┬──────┘ └─────────┘
                   │          │
                   └──────────┘
              data channels (SecureChannel)
              activation tensors flow L→R
```

The **orchestrator** runs on the host and:
1. Connects control channels to each stage, sends `Init` with shard specs, waits for `Ready`
2. Sends `EstablishDataChannels`, then connects/accepts data channels
3. Dispatches `StartRequest` with micro-batch scheduling, sends input tensors to stage 0, receives output tensors from the last stage

Each **stage** runs inside an enclave and:
1. Accepts a control channel, receives its `StageSpec` and `ActivationSpec`
2. Accepts a data-in channel, connects a data-out channel
3. Executes forward passes per the 1F1B schedule, streaming activation tensors to the next stage

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `mock` | Yes | Mock attestation provider/verifier for local development |
| `tcp` | Yes | TCP transport backend + TCP deployment helpers |
| `vsock` | No | VSock transport backend for Nitro Enclaves |
| `nitro` | No | AWS Nitro attestation provider/verifier |
| `sev-snp` | No | AMD SEV-SNP attestation provider/verifier |
| `tdx` | No | Intel TDX attestation provider/verifier |

## Quick Start

### In-process (duplex transport)

```rust
use confidential_ml_pipeline::*;
use confidential_ml_transport::{MockProvider, MockVerifier};

// Create duplex pairs for control and data channels
let (orch_ctrl, stage_ctrl) = tokio::io::duplex(65536);
let (orch_data_in, stage_data_in) = tokio::io::duplex(65536);
let (stage_data_out, orch_data_out) = tokio::io::duplex(65536);

// Spawn stage
tokio::spawn(async move {
    let mut runtime = StageRuntime::new(MyExecutor::new(), StageConfig::default());
    runtime.run(stage_ctrl, stage_data_in, stage_data_out,
                &MockProvider::new(), &MockVerifier::new()).await.unwrap();
});

// Run orchestrator
let mut orch = Orchestrator::new(OrchestratorConfig::default(), manifest)?;
orch.init(vec![orch_ctrl], &MockVerifier::new()).await?;
orch.establish_data_channels(orch_data_in, orch_data_out, vec![],
                              &MockVerifier::new(), &MockProvider::new()).await?;
let result = orch.infer(input_tensors, seq_len).await?;
```

### TCP multi-process

```rust
use confidential_ml_pipeline::tcp;

// Stage worker (each process)
let (ctrl_lis, _, din_lis, _) = tcp::bind_stage_listeners(ctrl_addr, din_addr).await?;
tcp::run_stage_with_listeners(executor, config, ctrl_lis, din_lis,
                               data_out_target, &provider, &verifier).await?;

// Orchestrator
let dout_listener = TcpListener::bind(dout_addr).await?;
let mut orch = tcp::init_orchestrator_tcp(config, manifest, dout_listener,
                                           &verifier, &provider).await?;
let result = orch.infer(input_tensors, seq_len).await?;
```

See `examples/tcp-pipeline/` for a complete multi-binary example.

## Examples

### Mock pipeline (in-process)

```bash
cargo run --example mock-pipeline --manifest-path examples/mock-pipeline/Cargo.toml
```

### TCP pipeline (multi-process)

```bash
# Terminal 1: Stage 0
cargo run --bin stage-worker --manifest-path examples/tcp-pipeline/Cargo.toml -- \
  --manifest manifest.json --stage-idx 0 --data-out-target 127.0.0.1:9011

# Terminal 2: Stage 1
cargo run --bin stage-worker --manifest-path examples/tcp-pipeline/Cargo.toml -- \
  --manifest manifest.json --stage-idx 1 --data-out-target 127.0.0.1:9020

# Terminal 3: Orchestrator
cargo run --bin pipeline-orch --manifest-path examples/tcp-pipeline/Cargo.toml -- \
  --manifest manifest.json --data-out-listen 127.0.0.1:9020
```

## Testing

```bash
# All tests (51 total: 23 unit + 18 integration + 3 TCP + 4 stress + 4 timeout)
cargo test

# Stress tests only (100 sequential requests, 1.5 MiB tensors, 16 micro-batches, 3-stage multi-request)
cargo test --test stress_test

# TCP integration tests only
cargo test --test tcp_pipeline

# With logging
RUST_LOG=debug cargo test --test tcp_pipeline -- --nocapture
```

## Benchmarks

### Synthetic (in-process, mock transport)

```bash
cargo bench --bench pipeline_bench
```

| Benchmark | Result |
|-----------|--------|
| Pipeline throughput (2-stage, 128KB tensor) | 923 µs, 135 MiB/s |
| Latency per stage (1KB tensor) | ~43 µs/stage |
| Schedule generation (4 stages, 16 micro-batches) | 4.0 µs |
| Relay forwarding (128KB) | 57 µs, 2.1 GiB/s |
| Protocol serde (StartRequest roundtrip) | 205 ns serialize, 419 ns deserialize |
| Health check (2 stages) | 39 µs |
| Multi micro-batch (2-stage, 16 micro-batches) | 598 µs |

### Nitro Enclave (GPT-2 124M, m6i.2xlarge, N=5)

End-to-end GPT-2 inference across real Nitro Enclaves with encrypted VSock transport (ChaCha20-Poly1305). 5 independent cold-boot runs per configuration. Mean +/- 95% CI (t-distribution, df=4).

| Metric | 1-Stage (12 layers) | 2-Stage (6+6) | 3-Stage (4+4+4) |
|--------|---------------------|----------------|------------------|
| TTFT | 92.5 +/- 1.8ms | 97.5 +/- 5.4ms | 107.1 +/- 13.7ms |
| Gen avg | 41.9 +/- 1.8ms/tok | 44.1 +/- 3.3ms/tok | 50.0 +/- 8.3ms/tok |
| Gen p50 | 41.9 +/- 1.8ms/tok | 44.1 +/- 3.4ms/tok | 49.9 +/- 8.2ms/tok |
| Gen p95 | 42.9 +/- 1.9ms/tok | 45.4 +/- 3.6ms/tok | 51.9 +/- 11.0ms/tok |
| Throughput | 23.9 +/- 1.0 tok/s | 22.7 +/- 1.6 tok/s | 20.3 +/- 2.9 tok/s |
| Overhead vs 1-stage | -- | +5.2% | +19.2% |

See [`examples/gpt2-pipeline/`](examples/gpt2-pipeline/) for the full example and [`benchmark_results/`](benchmark_results/) for raw data and detailed analysis.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
