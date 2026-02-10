#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$EXAMPLE_DIR")")"

# Per-stage EIF files (built by build_nitro.sh with --build-arg STAGE_IDX=N)
EIF_S0="$EXAMPLE_DIR/gpt2-pipeline-s0.eif"
EIF_S1="$EXAMPLE_DIR/gpt2-pipeline-s1.eif"
MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
TEXT="${1:-The capital of France is}"
MAX_TOKENS="${2:-20}"

# Well-known ports (must match manifest template)
CTRL_PORT=5000
DIN_PORT=5001
DOUT_PORT=5002
RELAY_PORT=6001   # host relay for stage 0 → stage 1 data

# Memory per enclave (EIF ~614MB + model ~500MB + runtime headroom)
ENCLAVE_MEM=3072
ENCLAVE_CPUS=2

for f in "$EIF_S0" "$EIF_S1"; do
    if [ ! -f "$f" ]; then
        echo "EIF not found: $f"
        echo "Run 'scripts/build_nitro.sh 2' first."
        exit 1
    fi
done

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
nitro-cli run-enclave \
    --eif-path "$EIF_S0" \
    --memory "$ENCLAVE_MEM" \
    --cpu-count "$ENCLAVE_CPUS" \
    --enclave-name gpt2-stage0 >/dev/null
CID0=$(nitro-cli describe-enclaves | jq -r '.[] | select(.EnclaveName == "gpt2-stage0") | .EnclaveCID')
EID0=$(nitro-cli describe-enclaves | jq -r '.[] | select(.EnclaveName == "gpt2-stage0") | .EnclaveID')
ENCLAVE_IDS+=("$EID0")
echo "  Enclave 0: CID=$CID0, ID=$EID0"

# 2. Launch enclave 1 (stage 1, layers 6-12)
echo "Launching enclave 1 (stage 1)..."
nitro-cli run-enclave \
    --eif-path "$EIF_S1" \
    --memory "$ENCLAVE_MEM" \
    --cpu-count "$ENCLAVE_CPUS" \
    --enclave-name gpt2-stage1 >/dev/null
CID1=$(nitro-cli describe-enclaves | jq -r '.[] | select(.EnclaveName == "gpt2-stage1") | .EnclaveCID')
EID1=$(nitro-cli describe-enclaves | jq -r '.[] | select(.EnclaveName == "gpt2-stage1") | .EnclaveID')
ENCLAVE_IDS+=("$EID1")
echo "  Enclave 1: CID=$CID1, ID=$EID1"

# 3. Generate manifest with actual CIDs
#    Host CID is 3 (parent instance).
#    Stage data_out endpoints route through the host:
#      stage 0 → host:RELAY_PORT (relay) → stage 1
#      stage 1 → host:DOUT_PORT  (final output to orchestrator)
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
        "data_out": { "type": "vsock", "cid": $HOST_CID, "port": $RELAY_PORT }
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

# Wait for enclaves to boot and load model
echo "Waiting for enclaves to boot..."
sleep 15

# 4. Build and run orchestrator on host
# Build with vsock-mock: mock attestation over real VSock transport.
# The host has no NSM device so NitroProvider would fail. Transport encryption
# (X25519 + ChaCha20-Poly1305) is still fully active — only attestation
# identity is mocked. Matches the enclave-side feature in Dockerfile.
echo "Building orchestrator (host)..."
cargo build --release --bin pipeline-orch \
    --manifest-path "$EXAMPLE_DIR/Cargo.toml" \
    --no-default-features --features vsock-mock

ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"

echo "Starting orchestrator..."
echo "Prompt: \"$TEXT\""
echo "---"

RUST_LOG=info "$ORCH_BIN" \
    --manifest "$MANIFEST_FILE" \
    --tokenizer "$TOKENIZER" \
    --text "$TEXT" \
    --max-tokens "$MAX_TOKENS"

echo "---"
echo "Done."

# Cleanup happens via trap
