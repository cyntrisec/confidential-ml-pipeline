#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$EXAMPLE_DIR")")"

EIF_PATH="$EXAMPLE_DIR/gpt2-pipeline.eif"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
TEXT="${1:-The capital of France is}"
MAX_TOKENS="${2:-20}"

# Well-known ports (must match manifest template)
CTRL_PORT=5000
DIN_PORT=5001
DOUT_PORT=5002

# Memory per enclave (GPT-2 ~500MB model + runtime)
ENCLAVE_MEM=1024
ENCLAVE_CPUS=2

if [ ! -f "$EIF_PATH" ]; then
    echo "EIF not found at $EIF_PATH"
    echo "Run scripts/build_nitro.sh first."
    exit 1
fi

ENCLAVE_IDS=()

cleanup() {
    echo ""
    echo "Shutting down enclaves..."
    for eid in "${ENCLAVE_IDS[@]}"; do
        nitro-cli terminate-enclave --enclave-id "$eid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# 1. Launch enclave 0 (stage 0, layers 0-6)
echo "Launching enclave 0 (stage 0)..."
RESULT0=$(nitro-cli run-enclave \
    --eif-path "$EIF_PATH" \
    --memory "$ENCLAVE_MEM" \
    --cpu-count "$ENCLAVE_CPUS" \
    --enclave-name gpt2-stage0)
CID0=$(echo "$RESULT0" | jq -r '.EnclaveCID')
EID0=$(echo "$RESULT0" | jq -r '.EnclaveID')
ENCLAVE_IDS+=("$EID0")
echo "  Enclave 0: CID=$CID0, ID=$EID0"

# 2. Launch enclave 1 (stage 1, layers 6-12)
echo "Launching enclave 1 (stage 1)..."
RESULT1=$(nitro-cli run-enclave \
    --eif-path "$EIF_PATH" \
    --memory "$ENCLAVE_MEM" \
    --cpu-count "$ENCLAVE_CPUS" \
    --enclave-name gpt2-stage1)
CID1=$(echo "$RESULT1" | jq -r '.EnclaveCID')
EID1=$(echo "$RESULT1" | jq -r '.EnclaveID')
ENCLAVE_IDS+=("$EID1")
echo "  Enclave 1: CID=$CID1, ID=$EID1"

# 3. Generate manifest with actual CIDs
#    Host CID is 3 (parent instance)
HOST_CID=3
MANIFEST_FILE=$(mktemp /tmp/manifest_vsock_XXXXXX.json)

cat > "$MANIFEST_FILE" <<EOF
{
  "model_name": "gpt2",
  "model_version": "1.0",
  "total_layers": 12,
  "stages": [
    {
      "stage_idx": 0,
      "layer_start": 0,
      "layer_end": 6,
      "weight_hashes": [],
      "expected_measurements": {},
      "endpoint": {
        "control": { "type": "vsock", "cid": $CID0, "port": $CTRL_PORT },
        "data_in": { "type": "vsock", "cid": $CID0, "port": $DIN_PORT },
        "data_out": { "type": "vsock", "cid": $CID1, "port": $DIN_PORT }
      }
    },
    {
      "stage_idx": 1,
      "layer_start": 6,
      "layer_end": 12,
      "weight_hashes": [],
      "expected_measurements": {},
      "endpoint": {
        "control": { "type": "vsock", "cid": $CID1, "port": $CTRL_PORT },
        "data_in": { "type": "vsock", "cid": $CID1, "port": $DIN_PORT },
        "data_out": { "type": "vsock", "cid": $HOST_CID, "port": $DOUT_PORT }
      }
    }
  ],
  "activation_spec": {
    "dtype": "F32",
    "hidden_dim": 768,
    "max_seq_len": 1024
  }
}
EOF

echo "Generated manifest: $MANIFEST_FILE"

# 4. Send stage config to each enclave via vsock-proxy or env vars
#    (nitro-cli doesn't support env vars directly; pass via --enclave-cid args
#    or use a config endpoint. For now, enclaves read from baked-in manifest
#    and the init.sh uses env vars set by the run-enclave command.)

# Wait for enclaves to boot
echo "Waiting for enclaves to boot..."
sleep 5

# 5. Build and run orchestrator on host
echo "Building orchestrator (host)..."
cargo build --release --bin pipeline-orch \
    --manifest-path "$EXAMPLE_DIR/Cargo.toml" \
    --no-default-features --features vsock-nitro

ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

echo "Starting orchestrator..."
echo "Prompt: \"$TEXT\""
echo "---"

RUST_LOG=info "$ORCH_BIN" \
    --manifest "$MANIFEST_FILE" \
    --data-out-port "$DOUT_PORT" \
    --tokenizer "$TOKENIZER" \
    --text "$TEXT" \
    --max-tokens "$MAX_TOKENS"

echo "---"
echo "Done."

# Cleanup happens via trap
