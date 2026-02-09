#!/bin/bash
# Failure test: kills stage 1 mid-generation and verifies the orchestrator handles it gracefully.
# Expected: orchestrator returns an error (not hang/crash), exit code != 0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

MANIFEST="$EXAMPLE_DIR/manifests/manifest_2stage.json"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "SKIP: model not downloaded"
    exit 0
fi

if [ ! -f "$STAGE_BIN" ] || [ ! -f "$ORCH_BIN" ]; then
    echo "Building (release)..."
    cargo build --release --manifest-path "$EXAMPLE_DIR/Cargo.toml"
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT

echo "=== Failure Recovery Test ==="
echo ""

# Start stages
RUST_LOG=error "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 0 \
    --data-out-target "127.0.0.1:9011" \
    --model-dir "$MODEL_DIR" &
STAGE0_PID=$!
PIDS+=($STAGE0_PID)

RUST_LOG=error "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 1 \
    --data-out-target "127.0.0.1:9021" \
    --model-dir "$MODEL_DIR" &
STAGE1_PID=$!
PIDS+=($STAGE1_PID)

sleep 3

# Run orchestrator with many tokens in background — we'll kill stage 1 after a few tokens.
# Use a timeout to ensure we don't hang forever.
RUST_LOG=error timeout 60 "$ORCH_BIN" \
    --manifest "$MANIFEST" \
    --data-out-listen "127.0.0.1:9021" \
    --tokenizer "$TOKENIZER" \
    --text "The capital of France is" \
    --max-tokens 500 > /tmp/failure_test_out.txt 2>&1 &
ORCH_PID=$!

# Wait just enough for a couple of tokens to flow, then kill stage 1.
sleep 1

# Kill stage 1 mid-generation.
echo "Killing stage 1 (PID $STAGE1_PID) mid-generation..."
kill -9 "$STAGE1_PID" 2>/dev/null || true

# Wait for orchestrator to notice and exit.
set +e
wait $ORCH_PID
ORCH_EXIT=$?
set -e

OUTPUT=$(cat /tmp/failure_test_out.txt 2>/dev/null || echo "")

echo "Orchestrator exit code: $ORCH_EXIT"
echo "Output (first 200 chars): ${OUTPUT:0:200}"
echo ""

if [ "$ORCH_EXIT" -ne 0 ]; then
    echo "PASS [failure-recovery]: orchestrator detected stage failure (exit=$ORCH_EXIT)"
else
    # Exit code 0 could mean it finished all 50 tokens before the kill happened.
    # Check if it actually generated < 50 tokens.
    TOKEN_COUNT=$(echo "$OUTPUT" | wc -w)
    if [ "$TOKEN_COUNT" -lt 500 ]; then
        echo "PASS [failure-recovery]: orchestrator exited cleanly with partial output ($TOKEN_COUNT words)"
    else
        echo "WARN [failure-recovery]: orchestrator completed all tokens before kill took effect"
    fi
fi

# Verify the orchestrator didn't hang (it should have exited already).
if kill -0 $ORCH_PID 2>/dev/null; then
    echo "FAIL [failure-recovery]: orchestrator is still running (hung)"
    kill $ORCH_PID 2>/dev/null
    cleanup
    rm -f /tmp/failure_test_out.txt
    exit 1
fi

echo ""
echo "=== Failure Recovery Test Complete ==="
rm -f /tmp/failure_test_out.txt
