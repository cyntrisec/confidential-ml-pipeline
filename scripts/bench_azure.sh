#!/bin/bash
# Azure SEV-SNP pipeline latency benchmark.
# Runs 1-stage and 2-stage GPT-2 pipeline N times each on local TCP,
# collects per-token latencies, and outputs JSON + markdown summary.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
EXAMPLE_DIR="$PIPELINE_DIR/examples/gpt2-pipeline"

MODEL_DIR="$EXAMPLE_DIR/model"
TOKENIZER="$MODEL_DIR/tokenizer.json"
STAGE_BIN="$EXAMPLE_DIR/target/release/stage-worker"
ORCH_BIN="$EXAMPLE_DIR/target/release/pipeline-orch"
MAX_TOKENS=20
PROMPT="The capital of France is"
N_RUNS="${1:-5}"
RESULTS_DIR="${2:-$PIPELINE_DIR/benchmark_results/azure_dc4ads_v5}"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "ERROR: model not downloaded (run examples/gpt2-pipeline/scripts/download_model.sh first)"
    exit 1
fi

echo "Building (release)..."
cargo build --release --manifest-path "$EXAMPLE_DIR/Cargo.toml" 2>&1 | tail -1

CPU_MODEL=$(grep -m1 'model name' /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs || echo "unknown")
KERNEL=$(uname -r)
HOSTNAME=$(hostname)

echo "Platform: $CPU_MODEL"
echo "Kernel: $KERNEL"
echo "Runs per config: $N_RUNS"
echo "Max tokens: $MAX_TOKENS"
echo ""

PIDS=()
cleanup() {
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null
    PIDS=()
}
trap cleanup EXIT

run_once() {
    local stages="$1"
    local manifest="$2"
    local dout_listen="$3"
    local latency_file="$4"

    cleanup

    for i in $(seq 0 $((stages - 1))); do
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

# Run benchmarks
for config in 1stage 2stage; do
    if [ "$config" = "1stage" ]; then
        STAGES=1
        MANIFEST="$EXAMPLE_DIR/manifests/manifest_1stage.json"
        DOUT="127.0.0.1:9011"
    else
        STAGES=2
        MANIFEST="$EXAMPLE_DIR/manifests/manifest_2stage.json"
        DOUT="127.0.0.1:9021"
    fi

    for run in $(seq 1 "$N_RUNS"); do
        echo "Running $config run $run/$N_RUNS..."
        run_once "$STAGES" "$MANIFEST" "$DOUT" "$RESULTS_DIR/${config}_run${run}.json"
        sleep 1
    done
done

echo ""
echo "All runs complete. Generating summary..."

# Generate statistical summary
python3 << 'PYEOF'
import json, math, os, sys

results_dir = os.environ.get("RESULTS_DIR", sys.argv[1] if len(sys.argv) > 1 else ".")
n_runs = int(os.environ.get("N_RUNS", "5"))

def percentile(data, p):
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)

def stats_for_config(config):
    ttfts = []
    gen_avgs = []
    gen_p50s = []
    gen_p95s = []
    gen_p99s = []
    all_gen = []

    for run in range(1, n_runs + 1):
        path = os.path.join(results_dir, f"{config}_run{run}.json")
        if not os.path.exists(path):
            continue
        d = json.load(open(path))
        lats = d["latencies_ms"]
        ttfts.append(lats[0])
        gen = sorted(lats[1:]) if len(lats) > 1 else []
        all_gen.extend(gen)
        if gen:
            n = len(gen)
            gen_avgs.append(sum(gen) / n)
            gen_p50s.append(gen[n // 2])
            gen_p95s.append(gen[min(math.ceil(n * 0.95) - 1, n - 1)])
            gen_p99s.append(gen[min(math.ceil(n * 0.99) - 1, n - 1)])

    def mean_std(vals):
        if not vals:
            return 0, 0
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return m, math.sqrt(v)

    ttft_mean, ttft_std = mean_std(ttfts)
    avg_mean, avg_std = mean_std(gen_avgs)
    p50_mean, p50_std = mean_std(gen_p50s)
    p95_mean, p95_std = mean_std(gen_p95s)
    p99_mean, p99_std = mean_std(gen_p99s)

    tps_list = [1000.0 / g if g > 0 else 0 for g in gen_avgs]
    tps_mean, tps_std = mean_std(tps_list)

    return {
        "runs": len(ttfts),
        "ttft_ms": {"mean": round(ttft_mean, 1), "std": round(ttft_std, 1)},
        "gen_avg_ms": {"mean": round(avg_mean, 1), "std": round(avg_std, 1)},
        "gen_p50_ms": {"mean": round(p50_mean, 1), "std": round(p50_std, 1)},
        "gen_p95_ms": {"mean": round(p95_mean, 1), "std": round(p95_std, 1)},
        "gen_p99_ms": {"mean": round(p99_mean, 1), "std": round(p99_std, 1)},
        "tokens_per_sec": {"mean": round(tps_mean, 1), "std": round(tps_std, 1)},
    }

summary = {}
for config in ["1stage", "2stage"]:
    summary[config] = stats_for_config(config)

# Compute overhead
if summary["1stage"]["gen_avg_ms"]["mean"] > 0:
    overhead_ms = summary["2stage"]["gen_avg_ms"]["mean"] - summary["1stage"]["gen_avg_ms"]["mean"]
    overhead_pct = overhead_ms / summary["1stage"]["gen_avg_ms"]["mean"] * 100
    summary["overhead_2stage_vs_1stage"] = {
        "ms": round(overhead_ms, 1),
        "pct": round(overhead_pct, 1),
    }

out_path = os.path.join(results_dir, "statistical_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {out_path}")

# Print markdown table
s1 = summary["1stage"]
s2 = summary["2stage"]
print()
print("## GPT-2 Pipeline Latency — Azure DC4ads_v5 (SEV-SNP)")
print()
print(f"| Metric | 1-stage (12 layers) | 2-stage (6+6) |")
print(f"|--------|---------------------|---------------|")
print(f"| TTFT | {s1['ttft_ms']['mean']} ± {s1['ttft_ms']['std']}ms | {s2['ttft_ms']['mean']} ± {s2['ttft_ms']['std']}ms |")
print(f"| Gen avg | {s1['gen_avg_ms']['mean']} ± {s1['gen_avg_ms']['std']}ms/tok | {s2['gen_avg_ms']['mean']} ± {s2['gen_avg_ms']['std']}ms/tok |")
print(f"| Gen p50 | {s1['gen_p50_ms']['mean']} ± {s1['gen_p50_ms']['std']}ms/tok | {s2['gen_p50_ms']['mean']} ± {s2['gen_p50_ms']['std']}ms/tok |")
print(f"| Gen p95 | {s1['gen_p95_ms']['mean']} ± {s1['gen_p95_ms']['std']}ms/tok | {s2['gen_p95_ms']['mean']} ± {s2['gen_p95_ms']['std']}ms/tok |")
print(f"| Gen p99 | {s1['gen_p99_ms']['mean']} ± {s1['gen_p99_ms']['std']}ms/tok | {s2['gen_p99_ms']['mean']} ± {s2['gen_p99_ms']['std']}ms/tok |")
print(f"| Tokens/sec | {s1['tokens_per_sec']['mean']} ± {s1['tokens_per_sec']['std']} | {s2['tokens_per_sec']['mean']} ± {s2['tokens_per_sec']['std']} |")
if "overhead_2stage_vs_1stage" in summary:
    o = summary["overhead_2stage_vs_1stage"]
    print(f"| **2-stage overhead** | — | **+{o['ms']}ms ({o['pct']}%)** |")
print()
print(f"**Config:** GPT-2 small (124M), KV-cache, greedy, TCP mock, N={n_runs} runs")
PYEOF

echo "Done."
