# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.2] - 2026-02-12

### Changed

- **Bumped `confidential-ml-transport` dependency** from `0.2` to `0.3` — picks up the `peer_attestation()` accessor and Azure SEV-SNP backend.
- **Azure SEV-SNP feature flag** (`azure-sev-snp`) — forwards to `confidential-ml-transport/azure-sev-snp` for Azure Confidential VM attestation.
- Feature count: 6 → 7 (added `azure-sev-snp`).

## [0.2.1] - 2026-02-12

### Added

- **GCP cross-VM benchmarks** — end-to-end GPT-2 pipeline across separate GCP c3-standard-4 VMs with encrypted TCP transport. Standard VMs, TDX VMs (mock attestation), and TDX VMs with real TDX attestation (configfs-tsm). 5 runs per configuration, statistical 95% CI.
- **Real TDX attestation benchmarks** — Phase 3 results showing ~4ms one-time attestation overhead, no per-token impact. TDX compute overhead ~15% vs standard VMs.
- **`tcp-tdx` feature** in GPT-2 pipeline example for real TDX attestation deployment.
- GCP benchmark scripts (`run_gcp.sh`, `bench_gcp.sh`, `cleanup_gcp.sh`).

## [0.2.0] - 2026-02-11

### Added

- **AMD SEV-SNP feature flag** (`sev-snp`) — forwards to `confidential-ml-transport/sev-snp` for SEV-SNP attestation backends.
- **Intel TDX feature flag** (`tdx`) — forwards to `confidential-ml-transport/tdx` for TDX attestation backends.
- CI feature matrix testing 7 feature combinations (expanded from 4).
- CI doc build check with `-D warnings` to catch broken doc links.
- `CHANGELOG.md`.

### Fixed

- **Test files missing feature gates** — added `#![cfg(feature = "mock")]` to all mock-dependent test files and `#![cfg(all(feature = "tcp", feature = "mock"))]` to `tcp_pipeline.rs`, so tests compile under any feature combination.
- **Broken doc links** — added `Self::` prefix to intra-doc method links in `orchestrator.rs` and `stage.rs`; escaped `[i]` index syntax in `relay.rs`.
- **`input_tensors.len() as u32` silent truncation** — replaced with `u32::try_from()` to return an explicit error when the micro-batch count exceeds `u32::MAX`.
- **`StageFailed` hardcoded `stage_idx: 0`** — both `recv_output_tensors` (orchestrator) and `recv_tensors` (stage) now report `stage_idx: usize::MAX` to indicate unknown origin, instead of falsely blaming stage 0.
- **`rand_request_id()` timestamp collisions** — replaced nanosecond timestamp with an atomic counter seeded from the clock, guaranteeing unique request IDs within a process.
- **Health check timeout did not taint pipeline** — `health_check()` now marks the pipeline as tainted on timeout, preventing subsequent requests from hitting a desynchronized protocol state.
- **Recursive async stack overflow in `wait_for_establish_data_channels`** — converted `Box::pin` recursion to a loop, eliminating stack overflow risk from many consecutive Ping messages.
- **Dead code cleanup** — removed unused `PipelineError::Relay(String)` variant and the error-discarding `From<StageError> for fmt::Error` impl.

### Changed

- Feature count: 4 → 6 (added `sev-snp`, `tdx`).
- Test count: 47 → 51.

## [0.1.0] - 2026-02-09

Initial release.

### Features

- Multi-enclave pipeline orchestration with 1F1B fill-drain scheduling.
- JSON shard manifest with layer ranges, weight hashes, and expected attestation measurements.
- Two-phase `StageRuntime` and `Orchestrator` APIs for split control/data channel establishment.
- Configurable timeouts for health checks and inference requests with drain-on-timeout recovery.
- TCP deployment helpers with retry-connect and exponential backoff.
- Transparent bidirectional relay mesh for inter-stage data channels.
- Error sentinel propagation on data channels for stage failure unblocking.
- Pluggable `StageExecutor` trait for user-defined forward passes.
- Mock, TCP, VSock, and Nitro attestation feature flags (forwarded to `confidential-ml-transport`).

[0.2.2]: https://github.com/cyntrisec/confidential-ml-pipeline/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cyntrisec/confidential-ml-pipeline/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cyntrisec/confidential-ml-pipeline/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/cyntrisec/confidential-ml-pipeline/releases/tag/v0.1.0
