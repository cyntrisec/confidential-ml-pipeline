#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$EXAMPLE_DIR")")"
WORKSPACE_DIR="$(dirname "$PIPELINE_DIR")"

MODEL_DIR="$EXAMPLE_DIR/model"
NUM_STAGES="${1:-2}"

# 1. Download model if needed
if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "Downloading GPT-2 model..."
    bash "$SCRIPT_DIR/download_model.sh"
else
    echo "Model already present at $MODEL_DIR"
fi

cd "$WORKSPACE_DIR"

# 2. Build one EIF per stage, each with its own STAGE_IDX baked in.
#    Nitro Enclaves cannot pass env vars at runtime, so STAGE_IDX must
#    be set at Docker build time via --build-arg.
for ((i=0; i<NUM_STAGES; i++)); do
    TAG="gpt2-pipeline-s${i}:latest"
    EIF_PATH="$EXAMPLE_DIR/gpt2-pipeline-s${i}.eif"

    echo "Building Docker image for stage $i..."
    docker build \
        -f "$EXAMPLE_DIR/Dockerfile" \
        --build-arg STAGE_IDX="$i" \
        -t "$TAG" \
        .

    echo "Building EIF for stage $i..."
    nitro-cli build-enclave \
        --docker-uri "$TAG" \
        --output-file "$EIF_PATH"

    echo "  Stage $i EIF: $EIF_PATH"
done

echo ""
echo "Built $NUM_STAGES EIF(s). PCR values above can be used for expected_measurements."
echo ""
echo "Next: run scripts/run_nitro.sh to launch enclaves and orchestrator."
