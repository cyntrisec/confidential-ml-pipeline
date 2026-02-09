#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$EXAMPLE_DIR")")"
WORKSPACE_DIR="$(dirname "$PIPELINE_DIR")"

MODEL_DIR="$EXAMPLE_DIR/model"

# 1. Download model if needed
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Downloading GPT-2 model..."
    bash "$SCRIPT_DIR/download_model.sh"
else
    echo "Model already present at $MODEL_DIR"
fi

# 2. Build Docker image from workspace root (needs both crates as context)
echo "Building Docker image..."
cd "$WORKSPACE_DIR"
docker build \
    -f "$EXAMPLE_DIR/Dockerfile" \
    -t gpt2-pipeline-enclave:latest \
    .

# 3. Build EIF (requires nitro-cli)
echo "Building EIF..."
EIF_PATH="$EXAMPLE_DIR/gpt2-pipeline.eif"
nitro-cli build-enclave \
    --docker-uri gpt2-pipeline-enclave:latest \
    --output-file "$EIF_PATH"

echo ""
echo "EIF built: $EIF_PATH"
echo "PCR values above can be used for expected_measurements in the manifest."
echo ""
echo "Next: run scripts/run_nitro.sh to launch enclaves and orchestrator."
