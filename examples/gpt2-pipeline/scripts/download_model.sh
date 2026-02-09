#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="${EXAMPLE_DIR}/model"

mkdir -p "$MODEL_DIR"

BASE_URL="https://huggingface.co/openai-community/gpt2/resolve/main"

for FILE in model.safetensors tokenizer.json config.json; do
    if [ ! -f "$MODEL_DIR/$FILE" ]; then
        echo "Downloading $FILE..."
        curl -L -o "$MODEL_DIR/$FILE" "$BASE_URL/$FILE"
    else
        echo "$FILE already exists, skipping."
    fi
done

echo "Model downloaded to $MODEL_DIR"
ls -lh "$MODEL_DIR"
