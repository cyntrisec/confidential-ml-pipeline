# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **AMD SEV-SNP feature flag** (`sev-snp`) — forwards to `confidential-ml-transport/sev-snp` for SEV-SNP attestation backends.
- **Intel TDX feature flag** (`tdx`) — forwards to `confidential-ml-transport/tdx` for TDX attestation backends.
- CI feature matrix testing 4 feature combinations (`--all-features`, `--no-default-features --features mock,tcp`, `--features mock`, `--features tcp`).
- `CHANGELOG.md`.

### Fixed

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

[Unreleased]: https://github.com/cyntrisec/confidential-ml-pipeline/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/cyntrisec/confidential-ml-pipeline/releases/tag/v0.1.0
