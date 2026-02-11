#!/bin/bash
# GCP multi-VM pipeline benchmark.
# Runs 1/2/3-stage GPT-2 pipeline N times each across GCP VMs,
# collects per-token latencies, computes stats, and saves all artifacts.
# Usage: bench_gcp.sh [N_RUNS] [RESULTS_DIR] [--tdx] [--feature FEATURE]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXAMPLE_DIR="$(dirname "$SCRIPT_DIR")"
PIPELINE_DIR="$(dirname "$(dirname "$EXAMPLE_DIR")")"

N_RUNS="${1:-5}"
RESULTS_DIR="${2:-$PIPELINE_DIR/benchmark_results/gcp_c3_standard}"
shift 2 2>/dev/null || true

TDX_FLAG=""
FEATURE_FLAG=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tdx) TDX_FLAG="--tdx"; EXTRA_ARGS+=(--tdx); shift ;;
        --feature) FEATURE_FLAG="$2"; EXTRA_ARGS+=(--feature "$2"); shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

ZONE="us-central1-a"
PROMPT="The capital of France is"
MAX_TOKENS=20

mkdir -p "$RESULTS_DIR"

echo "=== GCP Pipeline Benchmark ==="
echo "Runs per config: $N_RUNS"
echo "Results dir: $RESULTS_DIR"
echo "TDX: ${TDX_FLAG:-no}"
echo "Feature: ${FEATURE_FLAG:-tcp-mock (default)}"
echo ""

# --- Save platform metadata from stage0 ---
save_metadata() {
    echo "Collecting platform metadata..."
    local meta_file="$RESULTS_DIR/platform_metadata.json"

    local cpu_model kernel hostname_vm zone machine_type tdx_active
    cpu_model=$(gcloud compute ssh cmt-stage0 --zone="$ZONE" \
        --command="grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs" 2>/dev/null || echo "unknown")
    kernel=$(gcloud compute ssh cmt-stage0 --zone="$ZONE" \
        --command="uname -r" 2>/dev/null || echo "unknown")
    hostname_vm=$(gcloud compute ssh cmt-stage0 --zone="$ZONE" \
        --command="hostname" 2>/dev/null || echo "unknown")
    machine_type=$(gcloud compute instances describe cmt-stage0 --zone="$ZONE" \
        --format='get(machineType)' 2>/dev/null | rev | cut -d/ -f1 | rev || echo "unknown")

    if [ -n "$TDX_FLAG" ]; then
        tdx_active=$(gcloud compute ssh cmt-stage0 --zone="$ZONE" \
            --command="dmesg 2>/dev/null | grep -c -i tdx || echo 0" 2>/dev/null || echo "unknown")
    else
        tdx_active="N/A"
    fi

    local commit
    commit=$(gcloud compute ssh cmt-stage0 --zone="$ZONE" \
        --command="cd ~/workspace/confidential-ml-pipeline && git rev-parse --short HEAD" 2>/dev/null || \
        git -C "$PIPELINE_DIR" rev-parse --short HEAD 2>/dev/null || echo "unknown")

    cat > "$meta_file" <<EOF
{
  "cpu_model": "$cpu_model",
  "kernel": "$kernel",
  "hostname": "$hostname_vm",
  "machine_type": "$machine_type",
  "zone": "$ZONE",
  "tdx_flag": "${TDX_FLAG:-none}",
  "tdx_dmesg_lines": "$tdx_active",
  "feature": "${FEATURE_FLAG:-tcp-mock}",
  "n_runs": $N_RUNS,
  "max_tokens": $MAX_TOKENS,
  "prompt": "$PROMPT",
  "commit": "$commit",
  "timestamp": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
EOF
    echo "  Saved platform metadata to $meta_file"
}

# --- Save cross-VM RTT ---
save_rtt() {
    echo "Measuring cross-VM RTT..."
    local rtt_file="$RESULTS_DIR/cross_vm_rtt.txt"
    > "$rtt_file"  # Truncate before writing
    for i in 0 1 2; do
        local vm="cmt-stage${i}"
        if ! gcloud compute instances describe "$vm" --zone="$ZONE" &>/dev/null; then
            continue
        fi
        for j in 0 1 2; do
            if [ "$i" -eq "$j" ]; then continue; fi
            local target_vm="cmt-stage${j}"
            if ! gcloud compute instances describe "$target_vm" --zone="$ZONE" &>/dev/null; then
                continue
            fi
            local target_ip
            target_ip=$(gcloud compute instances describe "$target_vm" --zone="$ZONE" \
                --format='get(networkInterfaces[0].networkIP)')
            echo "  $vm -> $target_vm ($target_ip):" | tee -a "$rtt_file"
            gcloud compute ssh "$vm" --zone="$ZONE" \
                --command="ping -c 5 -q $target_ip" 2>/dev/null | tee -a "$rtt_file" || true
        done
    done
    echo ""
}

# --- Save stage worker logs after each run ---
save_stage_logs() {
    local config="$1"
    local run="$2"
    local num_stages="$3"
    local log_dir="$RESULTS_DIR/logs"
    mkdir -p "$log_dir"

    for i in $(seq 0 $((num_stages - 1))); do
        gcloud compute scp "cmt-stage${i}:~/stage${i}.log" \
            "$log_dir/${config}_run${run}_stage${i}.log" --zone="$ZONE" 2>/dev/null || true
    done
}

# --- Pre-create all VMs needed for the full benchmark ---
echo "Pre-creating all VMs for benchmark..."
MAX_STAGES=3  # We run 1/2/3-stage configs
VM_CREATE_ARGS=(--zone="$ZONE" --machine-type=c3-standard-4
    --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud
    --boot-disk-size=30GB)

if [ -n "$TDX_FLAG" ]; then
    VM_CREATE_ARGS+=(--confidential-compute-type=TDX
        --min-cpu-platform="Intel Sapphire Rapids"
        --maintenance-policy=TERMINATE)
fi

ORCH_CREATE_ARGS=(--zone="$ZONE"
    --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud)

if [ -n "$TDX_FLAG" ]; then
    ORCH_CREATE_ARGS+=(--machine-type=c3-standard-4 --boot-disk-size=30GB
        --confidential-compute-type=TDX
        --min-cpu-platform="Intel Sapphire Rapids"
        --maintenance-policy=TERMINATE)
else
    ORCH_CREATE_ARGS+=(--machine-type=e2-standard-2 --boot-disk-size=20GB)
fi

for i in $(seq 0 $((MAX_STAGES - 1))); do
    if ! gcloud compute instances describe "cmt-stage${i}" --zone="$ZONE" &>/dev/null; then
        echo "  Creating cmt-stage${i}..."
        if ! gcloud compute instances create "cmt-stage${i}" "${VM_CREATE_ARGS[@]}"; then
            echo "  WARNING: Failed to create cmt-stage${i} (likely quota). Skipping 3-stage config."
        fi
    else
        echo "  cmt-stage${i} already exists"
    fi
done
if ! gcloud compute instances describe "cmt-orch" --zone="$ZONE" &>/dev/null; then
    echo "  Creating cmt-orch..."
    gcloud compute instances create "cmt-orch" "${ORCH_CREATE_ARGS[@]}"
else
    echo "  cmt-orch already exists"
fi

# Firewall rule
gcloud compute firewall-rules create allow-pipeline-ports \
    --network=default --allow=tcp:9000-9031 \
    --source-ranges=10.128.0.0/20 2>/dev/null || true

echo ""

# Collect metadata and RTT now that VMs exist
save_metadata
save_rtt

# --- Run benchmarks ---

for config in 1stage 2stage 3stage; do
    case "$config" in
        1stage) STAGES=1 ;;
        2stage) STAGES=2 ;;
        3stage) STAGES=3 ;;
    esac

    # Check if enough VMs exist for this config
    all_exist=true
    for i in $(seq 0 $((STAGES - 1))); do
        if ! gcloud compute instances describe "cmt-stage${i}" --zone="$ZONE" &>/dev/null; then
            all_exist=false
            break
        fi
    done
    if ! $all_exist; then
        echo "Skipping $config — not enough VMs (need $STAGES stage VMs)"
        continue
    fi

    for run in $(seq 1 "$N_RUNS"); do
        echo "Running $config run $run/$N_RUNS..."
        LATENCY_FILE="$RESULTS_DIR/${config}_run${run}.json"

        "$SCRIPT_DIR/run_gcp.sh" "$STAGES" "$PROMPT" "$MAX_TOKENS" \
            ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} --latency-out "$LATENCY_FILE" || {
            echo "  WARNING: $config run $run failed, skipping"
            continue
        }

        # Save stage logs for this run
        save_stage_logs "$config" "$run" "$STAGES"

        sleep 2
    done
done

echo ""
echo "All runs complete. Generating summary..."

# --- Generate statistical summary ---
export RESULTS_DIR N_RUNS
python3 << 'PYEOF'
import json, math, os, sys

results_dir = os.environ.get("RESULTS_DIR", ".")
n_runs = int(os.environ.get("N_RUNS", "5"))

def percentile(data, p):
    k = (len(data) - 1) * p / 100.0
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return data[int(k)]
    return data[int(f)] * (c - k) + data[int(c)] * (k - f)

def t_critical(df):
    """Approximate t-critical value for 95% CI (two-tailed)."""
    # Lookup table for common df values
    table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
             6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
             15: 2.131, 20: 2.086, 30: 2.042}
    if df in table:
        return table[df]
    if df > 30:
        return 1.96  # Normal approximation
    # Linear interpolation for missing values
    keys = sorted(table.keys())
    for i in range(len(keys) - 1):
        if keys[i] < df < keys[i+1]:
            frac = (df - keys[i]) / (keys[i+1] - keys[i])
            return table[keys[i]] * (1 - frac) + table[keys[i+1]] * frac
    return 2.0

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

    def mean_std_ci(vals):
        if not vals:
            return 0, 0, 0
        m = sum(vals) / len(vals)
        if len(vals) < 2:
            return m, 0, 0
        v = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        s = math.sqrt(v)
        df = len(vals) - 1
        t = t_critical(df)
        ci = t * s / math.sqrt(len(vals))
        return m, s, ci

    ttft_mean, ttft_std, ttft_ci = mean_std_ci(ttfts)
    avg_mean, avg_std, avg_ci = mean_std_ci(gen_avgs)
    p50_mean, p50_std, p50_ci = mean_std_ci(gen_p50s)
    p95_mean, p95_std, p95_ci = mean_std_ci(gen_p95s)
    p99_mean, p99_std, p99_ci = mean_std_ci(gen_p99s)

    tps_list = [1000.0 / g if g > 0 else 0 for g in gen_avgs]
    tps_mean, tps_std, tps_ci = mean_std_ci(tps_list)

    return {
        "runs": len(ttfts),
        "ttft_ms": {"mean": round(ttft_mean, 1), "std": round(ttft_std, 1), "ci95": round(ttft_ci, 1)},
        "gen_avg_ms": {"mean": round(avg_mean, 1), "std": round(avg_std, 1), "ci95": round(avg_ci, 1)},
        "gen_p50_ms": {"mean": round(p50_mean, 1), "std": round(p50_std, 1), "ci95": round(p50_ci, 1)},
        "gen_p95_ms": {"mean": round(p95_mean, 1), "std": round(p95_std, 1), "ci95": round(p95_ci, 1)},
        "gen_p99_ms": {"mean": round(p99_mean, 1), "std": round(p99_std, 1), "ci95": round(p99_ci, 1)},
        "tokens_per_sec": {"mean": round(tps_mean, 1), "std": round(tps_std, 1), "ci95": round(tps_ci, 1)},
    }

summary = {}
for config in ["1stage", "2stage", "3stage"]:
    s = stats_for_config(config)
    if s["runs"] > 0:
        summary[config] = s

# Compute overheads vs 1-stage baseline
if "1stage" in summary and summary["1stage"]["gen_avg_ms"]["mean"] > 0:
    base = summary["1stage"]["gen_avg_ms"]["mean"]
    for config in ["2stage", "3stage"]:
        if config in summary and summary[config]["gen_avg_ms"]["mean"] > 0:
            overhead_ms = summary[config]["gen_avg_ms"]["mean"] - base
            overhead_pct = overhead_ms / base * 100
            summary[f"overhead_{config}_vs_1stage"] = {
                "ms": round(overhead_ms, 1),
                "pct": round(overhead_pct, 1),
            }

# Load platform metadata if available
meta_path = os.path.join(results_dir, "platform_metadata.json")
if os.path.exists(meta_path):
    summary["platform"] = json.load(open(meta_path))

out_path = os.path.join(results_dir, "statistical_summary.json")
with open(out_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"Summary written to {out_path}")

# Print markdown table
print()
configs = [c for c in ["1stage", "2stage", "3stage"] if c in summary]
if not configs:
    print("No results collected.")
    sys.exit(0)

header = "| Metric |"
sep = "|--------|"
for c in configs:
    stages = c.replace("stage", "")
    header += f" {stages}-stage |"
    sep += "---------|"

print("## GPT-2 Pipeline Latency — GCP Cross-VM")
print()
print(header)
print(sep)

for metric, key in [("TTFT", "ttft_ms"), ("Gen avg", "gen_avg_ms"),
                     ("Gen p50", "gen_p50_ms"), ("Gen p95", "gen_p95_ms"),
                     ("Gen p99", "gen_p99_ms"), ("Tokens/sec", "tokens_per_sec")]:
    row = f"| {metric} |"
    for c in configs:
        s = summary[c]
        unit = "ms/tok" if "gen_" in key else ("ms" if key == "ttft_ms" else "tok/s")
        if key == "ttft_ms":
            unit = "ms"
        row += f" {s[key]['mean']} +/- {s[key]['ci95']} {unit} |"
    print(row)

# Overhead rows
for config in ["2stage", "3stage"]:
    key = f"overhead_{config}_vs_1stage"
    if key in summary:
        o = summary[key]
        stages = config.replace("stage", "")
        row = f"| **{stages}-stage overhead** |"
        sign = "+" if o['ms'] >= 0 else ""
        for c in configs:
            if c == config:
                row += f" **{sign}{o['ms']}ms ({sign}{o['pct']}%)** |"
            else:
                row += " — |"
        print(row)

print()
platform = summary.get("platform", {})
cpu = platform.get("cpu_model", "unknown")
commit = platform.get("commit", "unknown")
tdx = platform.get("tdx_flag", "none")
feat = platform.get("feature", "tcp-mock")
print(f"**Config:** GPT-2 small (124M), KV-cache, greedy, {feat}, N={n_runs} runs")
print(f"**CPU:** {cpu}")
print(f"**TDX:** {tdx}")
print(f"**Commit:** {commit}")
PYEOF

echo ""
echo "All artifacts saved to $RESULTS_DIR/"
echo "  - Raw latency files: ${RESULTS_DIR}/*_run*.json"
echo "  - Stage logs: ${RESULTS_DIR}/logs/"
echo "  - Platform metadata: ${RESULTS_DIR}/platform_metadata.json"
echo "  - Cross-VM RTT: ${RESULTS_DIR}/cross_vm_rtt.txt"
echo "  - Statistical summary: ${RESULTS_DIR}/statistical_summary.json"
echo ""
echo "Done."
