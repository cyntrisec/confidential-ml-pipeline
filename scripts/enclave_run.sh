#!/bin/bash
# Runs on the EC2 instance. Launches enclaves, writes manifest, runs benchmark.
# Usage: bash enclave_run.sh <config> <prompt> <max_tokens>
set -euo pipefail

CONFIG="$1"
PROMPT="$2"
MAX_TOKENS="$3"

case "$CONFIG" in
  1stage)
    nitro-cli run-enclave --eif-path ~/gpt2-1stage-s0.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s0 --debug-mode >/dev/null
    CID0=$(nitro-cli describe-enclaves | jq -r '.[0].EnclaveCID')
    echo "CID0=$CID0"
    jq -n --argjson cid0 "$CID0" '{
      model_name: "gpt2", model_version: "1.0", total_layers: 12,
      stages: [{
        stage_idx: 0, layer_start: 0, layer_end: 12, weight_hashes: [], expected_measurements: {},
        endpoint: {
          control: {type: "vsock", cid: $cid0, port: 5000},
          data_in: {type: "vsock", cid: $cid0, port: 5001},
          data_out: {type: "vsock", cid: 3, port: 5002}
        }
      }],
      activation_spec: {dtype: "F32", hidden_dim: 768, max_seq_len: 1024}
    }' > ~/manifest_run.json
    ;;
  2stage)
    nitro-cli run-enclave --eif-path ~/gpt2-2stage-s0.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s0 --debug-mode >/dev/null
    nitro-cli run-enclave --eif-path ~/gpt2-2stage-s1.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s1 --debug-mode >/dev/null
    CIDS=$(nitro-cli describe-enclaves | jq -r 'sort_by(.EnclaveName) | .[].EnclaveCID')
    CID0=$(echo "$CIDS" | sed -n '1p')
    CID1=$(echo "$CIDS" | sed -n '2p')
    echo "CID0=$CID0 CID1=$CID1"
    jq -n --argjson cid0 "$CID0" --argjson cid1 "$CID1" '{
      model_name: "gpt2", model_version: "1.0", total_layers: 12,
      stages: [
        {stage_idx: 0, layer_start: 0, layer_end: 6, weight_hashes: [], expected_measurements: {},
         endpoint: {control: {type: "vsock", cid: $cid0, port: 5000}, data_in: {type: "vsock", cid: $cid0, port: 5001}, data_out: {type: "vsock", cid: 3, port: 6001}}},
        {stage_idx: 1, layer_start: 6, layer_end: 12, weight_hashes: [], expected_measurements: {},
         endpoint: {control: {type: "vsock", cid: $cid1, port: 5000}, data_in: {type: "vsock", cid: $cid1, port: 5001}, data_out: {type: "vsock", cid: 3, port: 5002}}}
      ],
      activation_spec: {dtype: "F32", hidden_dim: 768, max_seq_len: 1024}
    }' > ~/manifest_run.json
    ;;
  3stage)
    nitro-cli run-enclave --eif-path ~/gpt2-3stage-s0.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s0 --debug-mode >/dev/null
    nitro-cli run-enclave --eif-path ~/gpt2-3stage-s1.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s1 --debug-mode >/dev/null
    nitro-cli run-enclave --eif-path ~/gpt2-3stage-s2.eif --memory 2560 --cpu-count 2 --enclave-name gpt2-s2 --debug-mode >/dev/null
    CIDS=$(nitro-cli describe-enclaves | jq -r 'sort_by(.EnclaveName) | .[].EnclaveCID')
    CID0=$(echo "$CIDS" | sed -n '1p')
    CID1=$(echo "$CIDS" | sed -n '2p')
    CID2=$(echo "$CIDS" | sed -n '3p')
    echo "CID0=$CID0 CID1=$CID1 CID2=$CID2"
    jq -n --argjson cid0 "$CID0" --argjson cid1 "$CID1" --argjson cid2 "$CID2" '{
      model_name: "gpt2", model_version: "1.0", total_layers: 12,
      stages: [
        {stage_idx: 0, layer_start: 0, layer_end: 4, weight_hashes: [], expected_measurements: {},
         endpoint: {control: {type: "vsock", cid: $cid0, port: 5000}, data_in: {type: "vsock", cid: $cid0, port: 5001}, data_out: {type: "vsock", cid: 3, port: 6001}}},
        {stage_idx: 1, layer_start: 4, layer_end: 8, weight_hashes: [], expected_measurements: {},
         endpoint: {control: {type: "vsock", cid: $cid1, port: 5000}, data_in: {type: "vsock", cid: $cid1, port: 5001}, data_out: {type: "vsock", cid: 3, port: 6002}}},
        {stage_idx: 2, layer_start: 8, layer_end: 12, weight_hashes: [], expected_measurements: {},
         endpoint: {control: {type: "vsock", cid: $cid2, port: 5000}, data_in: {type: "vsock", cid: $cid2, port: 5001}, data_out: {type: "vsock", cid: 3, port: 5002}}}
      ],
      activation_spec: {dtype: "F32", hidden_dim: 768, max_seq_len: 1024}
    }' > ~/manifest_run.json
    ;;
esac

echo "Waiting 15s for model load..."
sleep 15

source ~/.cargo/env
cd ~/cmp/examples/gpt2-pipeline

RUST_LOG=info target/release/pipeline-orch \
  --manifest ~/manifest_run.json \
  --tokenizer model/tokenizer.json \
  --text "$PROMPT" \
  --max-tokens "$MAX_TOKENS" \
  --latency-out ~/latency_run.json 2>&1 | grep -E 'Latency Summary|Prompt|Generation|Tokens'

echo "=== RUN COMPLETE ==="
