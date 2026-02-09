#!/bin/bash
# Verify Rust pipeline output matches HuggingFace transformers reference (greedy decoding).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

PROMPT="The capital of France is"
MAX_TOKENS=20

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: model not downloaded (run download_model.sh first)"
    exit 1
fi

# --- Step 1: Python reference ---
echo "=== Python Reference (HuggingFace transformers, greedy) ==="

PYTHON_OUT=$(python3 -c "
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained('$MODEL_DIR')
model = GPT2LMHeadModel.from_pretrained('$MODEL_DIR', torch_dtype=torch.float32)
model.eval()

input_ids = tokenizer.encode('$PROMPT', return_tensors='pt')
prompt_len = input_ids.shape[1]

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=$MAX_TOKENS,
        do_sample=False,
        temperature=1.0,
    )

generated_ids = output[0][prompt_len:].tolist()
generated_text = tokenizer.decode(generated_ids)

# Print token IDs and text for comparison
print('TOKEN_IDS:' + ','.join(str(t) for t in generated_ids))
print('TEXT:' + generated_text)
" 2>/dev/null)

PYTHON_IDS=$(echo "$PYTHON_OUT" | grep '^TOKEN_IDS:' | cut -d: -f2)
PYTHON_TEXT=$(echo "$PYTHON_OUT" | grep '^TEXT:' | cut -d: -f2-)

echo "  Token IDs: $PYTHON_IDS"
echo "  Text: $PYTHON_TEXT"

# --- Step 2: Rust pipeline ---
echo ""
echo "=== Rust Pipeline (2-stage, KV-cache, greedy) ==="

MANIFEST="$EXAMPLE_DIR/manifests/manifest_2stage.json"

if [ ! -f "$STAGE_BIN" ] || [ ! -f "$ORCH_BIN" ]; then
    echo "Building (release)..."
    cargo build --release --manifest-path "$EXAMPLE_DIR/Cargo.toml" 2>/dev/null
fi

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    PIDS=()
}
trap cleanup EXIT

RUST_LOG=error "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 0 \
    --data-out-target 127.0.0.1:9011 \
    --model-dir "$MODEL_DIR" &
PIDS+=($!)

RUST_LOG=error "$STAGE_BIN" \
    --manifest "$MANIFEST" \
    --stage-idx 1 \
    --data-out-target 127.0.0.1:9021 \
    --model-dir "$MODEL_DIR" &
PIDS+=($!)

sleep 3

RUST_OUT=$(RUST_LOG=error "$ORCH_BIN" \
    --manifest "$MANIFEST" \
    --data-out-listen 127.0.0.1:9021 \
    --tokenizer "$TOKENIZER" \
    --text "$PROMPT" \
    --max-tokens "$MAX_TOKENS" \
    --latency-out /tmp/verify_latency.json 2>/dev/null)

cleanup

# Extract Rust token IDs from latency JSON (token count) and generated text
RUST_TEXT=$(echo "$RUST_OUT" | sed "s/^${PROMPT}//")
echo "  Text: $RUST_TEXT"

# --- Step 3: Compare ---
echo ""
echo "=== Comparison ==="

if [ "$PYTHON_TEXT" = "$RUST_TEXT" ]; then
    echo "PASS: Rust pipeline output matches Python reference exactly."
    echo ""
    echo "  Prompt:    $PROMPT"
    echo "  Generated: $PYTHON_TEXT"
    rm -f /tmp/verify_latency.json
    exit 0
else
    echo "MISMATCH detected."
    echo ""
    echo "  Prompt:  $PROMPT"
    echo "  Python:  $PYTHON_TEXT"
    echo "  Rust:    $RUST_TEXT"
    echo ""

    # Find first divergence point
    py_words=($PYTHON_TEXT)
    rs_words=($RUST_TEXT)
    for i in "${!py_words[@]}"; do
        if [ "${py_words[$i]}" != "${rs_words[$i]:-}" ]; then
            echo "  First difference at word $i: Python='${py_words[$i]}' Rust='${rs_words[$i]:-<missing>}'"
            break
        fi
    done

    rm -f /tmp/verify_latency.json
    exit 1
fi
