#!/bin/bash
# Benchmark: runs 2-stage and 3-stage pipelines, measures per-token latency, outputs markdown table.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"

MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"
MAX_TOKENS=20
PROMPT="The capital of France is"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: model not downloaded (run download_model.sh first)"
    exit 1
fi

echo "Building (release)..."
cargo build --release --manifest-path "$EXAMPLE_DIR/Cargo.toml" 2>/dev/null

CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
COMMIT=$(git -C "$EXAMPLE_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    PIDS=()
}
trap cleanup EXIT

run_bench() {
    local stages="$1"
    local manifest="$2"
    local dout_listen="$3"
    local latency_file="$4"

    cleanup

    for i in $(seq 0 $((stages - 1))); do
        local ctrl_port=$((9000 + i * 10))
        local din_port=$((9001 + i * 10))
        local dout_port=$((9011 + i * 10))

        RUST_LOG=error "$STAGE_BIN" \
            --manifest "$manifest" \
            --stage-idx "$i" \
            --data-out-target "127.0.0.1:$dout_port" \
            --model-dir "$MODEL_DIR" &
        PIDS+=($!)
    done

    sleep 3

    RUST_LOG=error "$ORCH_BIN" \
        --manifest "$manifest" \
        --data-out-listen "$dout_listen" \
        --tokenizer "$TOKENIZER" \
        --text "$PROMPT" \
        --max-tokens "$MAX_TOKENS" \
        --latency-out "$latency_file" > /dev/null 2>&1

    cleanup
}

echo "Running 2-stage benchmark..."
run_bench 2 "$EXAMPLE_DIR/manifests/manifest_2stage.json" "127.0.0.1:9021" "/tmp/bench_2stage.json"

echo "Running 3-stage benchmark..."
run_bench 3 "$EXAMPLE_DIR/manifests/manifest_3stage.json" "127.0.0.1:9031" "/tmp/bench_3stage.json"

# Parse results
parse_latencies() {
    local file="$1"
    python3 -c "
import json, sys
d = json.load(open('$file'))
lats = d['latencies_ms']
lats_sorted = sorted(lats)
n = len(lats)
prompt = lats[0]
gen = lats[1:] if n > 1 else []
gen_avg = sum(gen) / len(gen) if gen else 0
p50 = lats_sorted[n // 2]
p95_idx = min(int(n * 0.95), n - 1)
p95 = lats_sorted[p95_idx]
p99_idx = min(int(n * 0.99), n - 1)
p99 = lats_sorted[p99_idx]
print(f'{prompt:.1f}|{gen_avg:.1f}|{p50:.1f}|{p95:.1f}|{p99:.1f}')
"
}

STATS_2=$(parse_latencies /tmp/bench_2stage.json)
STATS_3=$(parse_latencies /tmp/bench_3stage.json)

IFS='|' read -r P2_PROMPT P2_AVG P2_P50 P2_P95 P2_P99 <<< "$STATS_2"
IFS='|' read -r P3_PROMPT P3_AVG P3_P50 P3_P95 P3_P99 <<< "$STATS_3"

echo ""
echo "## GPT-2 Pipeline Benchmark"
echo ""
echo "| Metric | 2-stage | 3-stage |"
echo "|--------|---------|---------|"
echo "| Prompt (TTFT) | ${P2_PROMPT}ms | ${P3_PROMPT}ms |"
echo "| Gen avg | ${P2_AVG}ms/tok | ${P3_AVG}ms/tok |"
echo "| p50 | ${P2_P50}ms | ${P3_P50}ms |"
echo "| p95 | ${P2_P95}ms | ${P3_P95}ms |"
echo "| p99 | ${P2_P99}ms | ${P3_P99}ms |"
echo "| Tokens | ${MAX_TOKENS} | ${MAX_TOKENS} |"
echo ""
echo "**Config:** GPT-2 small (124M), KV-cache enabled, greedy decoding, TCP mock transport"
echo "**CPU:** ${CPU_MODEL}"
echo "**Commit:** ${COMMIT}"

# Clean up temp files
rm -f /tmp/bench_2stage.json /tmp/bench_3stage.json
