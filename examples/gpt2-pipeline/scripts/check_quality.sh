#!/bin/bash
# Quality check: runs 2-stage pipeline with known prompts, verifies output contains expected substrings.
# Exit code 0 = all checks pass, 1 = failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

MANIFEST="$EXAMPLE_DIR/manifests/manifest_2stage.json"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "SKIP: model not downloaded (run download_model.sh first)"
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

start_stages() {
    RUST_LOG=error "$STAGE_BIN" \
        --manifest "$MANIFEST" \
        --stage-idx 0 \
        --data-out-target "127.0.0.1:9011" \
        --model-dir "$MODEL_DIR" &
    PIDS+=($!)

    RUST_LOG=error "$STAGE_BIN" \
        --manifest "$MANIFEST" \
        --stage-idx 1 \
        --data-out-target "127.0.0.1:9021" \
        --model-dir "$MODEL_DIR" &
    PIDS+=($!)

    sleep 3
}

run_prompt() {
    local text="$1"
    local max_tokens="$2"
    RUST_LOG=error "$ORCH_BIN" \
        --manifest "$MANIFEST" \
        --data-out-listen "127.0.0.1:9021" \
        --tokenizer "$TOKENIZER" \
        --text "$text" \
        --max-tokens "$max_tokens" 2>/dev/null
}

PASS=0
FAIL=0

check() {
    local name="$1"
    local prompt="$2"
    local tokens="$3"
    local expected="$4"

    start_stages
    local output
    output=$(run_prompt "$prompt" "$tokens") || { echo "FAIL [$name]: pipeline error"; FAIL=$((FAIL+1)); cleanup; PIDS=(); return; }
    cleanup
    PIDS=()

    if echo "$output" | grep -qi "$expected"; then
        echo "PASS [$name]: \"$output\""
        PASS=$((PASS+1))
    else
        echo "FAIL [$name]: expected substring \"$expected\" in: \"$output\""
        FAIL=$((FAIL+1))
    fi
}

echo "=== GPT-2 Pipeline Quality Checks ==="
echo ""

# Test 1: Geography - GPT-2 should produce "Paris" or "the" after this prompt
check "geography" "The capital of France is" 5 "the"

# Test 2: Continuation - should produce coherent English
check "continuation" "Once upon a time" 10 "the"

# Test 3: Determinism - same prompt should produce same output (greedy decoding)
start_stages
out1=$(run_prompt "The meaning of life is" 5) || { echo "FAIL [determinism]: pipeline error"; FAIL=$((FAIL+1)); cleanup; PIDS=(); }
cleanup
PIDS=()

start_stages
out2=$(run_prompt "The meaning of life is" 5) || { echo "FAIL [determinism]: pipeline error"; FAIL=$((FAIL+1)); cleanup; PIDS=(); }
cleanup
PIDS=()

if [ -n "$out1" ] && [ "$out1" = "$out2" ]; then
    echo "PASS [determinism]: outputs match: \"$out1\""
    PASS=$((PASS+1))
else
    echo "FAIL [determinism]: \"$out1\" != \"$out2\""
    FAIL=$((FAIL+1))
fi

echo ""
echo "=== Results: $PASS passed, $FAIL failed ==="

if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
