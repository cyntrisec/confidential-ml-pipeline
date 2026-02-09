#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

MANIFEST="$EXAMPLE_DIR/manifests/manifest_3stage.json"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
TEXT="${1:-The capital of France is}"
MAX_TOKENS="${2:-20}"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Model not found. Run download_model.sh first."
    exit 1
fi

echo "Building (release)..."
cargo build --release --manifest-path "$EXAMPLE_DIR/Cargo.toml"

STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

PIDS=()

cleanup() {
    echo ""
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
}
trap cleanup EXIT

echo "Starting stage 0 (layers 0-4)..."
RUST_LOG=info "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 0 \
    --data-out-target "127.0.0.1:9011" \
    --model-dir "$MODEL_DIR" &
PIDS+=($!)

echo "Starting stage 1 (layers 4-8)..."
RUST_LOG=info "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 1 \
    --data-out-target "127.0.0.1:9021" \
    --model-dir "$MODEL_DIR" &
PIDS+=($!)

echo "Starting stage 2 (layers 8-12)..."
RUST_LOG=info "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 2 \
    --data-out-target "127.0.0.1:9031" \
    --model-dir "$MODEL_DIR" &
PIDS+=($!)

sleep 2

echo "Starting orchestrator..."
echo "Prompt: \"$TEXT\""
echo "---"

RUST_LOG=info "$ORCH_BIN" \
    --manifest "$MANIFEST" \
    --data-out-listen "127.0.0.1:9031" \
    --tokenizer "$TOKENIZER" \
    --text "$TEXT" \
    --max-tokens "$MAX_TOKENS"

echo "---"
echo "Done."
