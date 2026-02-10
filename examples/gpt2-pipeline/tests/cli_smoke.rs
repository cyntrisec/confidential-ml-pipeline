//! CLI smoke tests — catch flag drift between binaries and docs.

use std::process::Command;

fn cargo_bin(name: &str) -> std::path::PathBuf {
    // cargo test builds binaries into the deps directory;
    // `cargo_bin` equivalent via env var set by cargo.
    let mut path = std::path::PathBuf::from(env!("CARGO_BIN_EXE_pipeline-orch"));
    path.pop(); // remove binary name
    path.push(name);
    path
}

/// Assert that `--help` output contains all expected flags for pipeline-orch.
#[test]
fn pipeline_orch_help_contains_documented_flags() {
    let output = Command::new(cargo_bin("pipeline-orch"))
        .arg("--help")
        .output()
        .expect("failed to run pipeline-orch --help");

    let help = String::from_utf8_lossy(&output.stdout);

    // Core flags that must exist regardless of feature gate.
    for flag in &["--manifest", "--tokenizer", "--text", "--max-tokens", "--latency-out"] {
        assert!(
            help.contains(flag),
            "pipeline-orch --help missing documented flag: {flag}\n--- help output ---\n{help}"
        );
    }

    // --data-out-port must NOT appear (was removed; VSock reads from manifest).
    assert!(
        !help.contains("--data-out-port"),
        "pipeline-orch --help still contains removed flag --data-out-port\n--- help output ---\n{help}"
    );
}

/// Assert that `--help` output contains all expected flags for stage-worker.
#[test]
fn stage_worker_help_contains_documented_flags() {
    let output = Command::new(cargo_bin("stage-worker"))
        .arg("--help")
        .output()
        .expect("failed to run stage-worker --help");

    let help = String::from_utf8_lossy(&output.stdout);

    for flag in &["--manifest", "--stage-idx", "--model-dir"] {
        assert!(
            help.contains(flag),
            "stage-worker --help missing documented flag: {flag}\n--- help output ---\n{help}"
        );
    }
}

/// stage-worker with out-of-range --stage-idx exits non-zero with a clear error.
#[test]
fn stage_worker_invalid_stage_idx() {
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("manifests")
        .join("manifest_2stage.json");

    // stage_idx=99 is out of range for a 2-stage manifest.
    // In tcp-mock mode, --data-out-target is required but won't be reached.
    let output = Command::new(cargo_bin("stage-worker"))
        .args([
            "--manifest",
            manifest.to_str().unwrap(),
            "--stage-idx",
            "99",
            "--model-dir",
            "/nonexistent",
            "--data-out-target",
            "127.0.0.1:1",
        ])
        .output()
        .expect("failed to run stage-worker");

    assert!(
        !output.status.success(),
        "stage-worker should exit non-zero for invalid stage_idx"
    );

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("stage_idx 99 out of range"),
        "expected clear error message about stage_idx, got:\n{stderr}"
    );
}
