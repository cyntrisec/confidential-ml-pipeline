#!/bin/bash
# GCP multi-VM pipeline runner.
# Launches stage workers on separate GCP VMs and orchestrator on a dedicated VM.
# Usage: run_gcp.sh <num_stages> "<prompt>" <max_tokens> [--tdx] [--feature FEATURE] [--latency-out FILE] [--taskset CORES]
set -euo pipefail

ZONE="us-central1-a"
IMAGE_FAMILY="ubuntu-2404-lts-amd64"
IMAGE_PROJECT="ubuntu-os-cloud"
STAGE_MACHINE="c3-standard-4"
ORCH_MACHINE="e2-standard-2"
BOOT_DISK="30GB"
ORCH_BOOT_DISK="20GB"
SETUP_SENTINEL=".cmt-setup-done"
WORKSPACE="workspace"

# --- Parse arguments ---
NUM_STAGES="${1:?Usage: run_gcp.sh <num_stages> \"<prompt>\" <max_tokens> [--tdx] [--feature FEATURE] [--latency-out FILE]}"
PROMPT="${2:?Missing prompt}"
MAX_TOKENS="${3:?Missing max_tokens}"
shift 3

TDX=false
FEATURE="tcp-mock"
LATENCY_OUT=""
TASKSET_CORES=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tdx) TDX=true; shift ;;
        --feature) FEATURE="$2"; shift 2 ;;
        --latency-out) LATENCY_OUT="$2"; shift 2 ;;
        --taskset) TASKSET_CORES="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if $TDX; then
    ORCH_MACHINE="c3-standard-4"
    ORCH_BOOT_DISK="30GB"
fi

echo "=== GCP Pipeline Runner ==="
echo "Stages: $NUM_STAGES | Feature: $FEATURE | TDX: $TDX"
echo "Prompt: \"$PROMPT\" | Max tokens: $MAX_TOKENS"
echo ""

# --- Helper: create VM if it doesn't exist ---
create_vm() {
    local name="$1"
    local machine="$2"
    local disk="$3"

    if gcloud compute instances describe "$name" --zone="$ZONE" &>/dev/null; then
        echo "  $name already exists, skipping creation"
        return 0
    fi

    local cmd=(gcloud compute instances create "$name"
        --zone="$ZONE"
        --machine-type="$machine"
        --image-family="$IMAGE_FAMILY"
        --image-project="$IMAGE_PROJECT"
        --boot-disk-size="$disk"
    )

    if $TDX; then
        cmd+=(
            --confidential-compute-type=TDX
            --min-cpu-platform="Intel Sapphire Rapids"
            --maintenance-policy=TERMINATE
        )
    fi

    "${cmd[@]}"
}

# --- Helper: SSH with retries (VM may still be booting) ---
gssh() {
    local vm="$1"
    shift
    gcloud compute ssh "$vm" --zone="$ZONE" --command="$*" 2>/dev/null
}

gssh_retry() {
    local vm="$1"
    shift
    for attempt in 1 2 3 4 5; do
        if gcloud compute ssh "$vm" --zone="$ZONE" --command="$*" 2>/dev/null; then
            return 0
        fi
        echo "  SSH to $vm failed (attempt $attempt/5), retrying in 10s..."
        sleep 10
    done
    echo "ERROR: Failed to SSH to $vm after 5 attempts"
    return 1
}

# --- Helper: get internal IP ---
get_ip() {
    local name="$1"
    gcloud compute instances describe "$name" --zone="$ZONE" \
        --format='get(networkInterfaces[0].networkIP)'
}

# --- 1. Create VMs ---
echo "Creating VMs..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    create_vm "cmt-stage${i}" "$STAGE_MACHINE" "$BOOT_DISK"
done
create_vm "cmt-orch" "$ORCH_MACHINE" "$ORCH_BOOT_DISK"
echo ""

# Wait for VMs to be reachable
echo "Waiting for VMs to boot..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    gssh_retry "cmt-stage${i}" "true"
done
gssh_retry "cmt-orch" "true"
echo "All VMs reachable."
echo ""

# --- 2. Firewall rule (idempotent) ---
gcloud compute firewall-rules create allow-pipeline-ports \
    --network=default --allow=tcp:9000-9031 \
    --source-ranges=10.128.0.0/20 2>/dev/null || true

# --- 3. Collect internal IPs ---
declare -a STAGE_IPS
for i in $(seq 0 $((NUM_STAGES - 1))); do
    STAGE_IPS[$i]=$(get_ip "cmt-stage${i}")
    echo "Stage $i IP: ${STAGE_IPS[$i]}"
done
ORCH_IP=$(get_ip "cmt-orch")
echo "Orch IP: $ORCH_IP"
echo ""

# --- 4. First-run setup ---
setup_vm() {
    local vm="$1"
    if gssh "$vm" "test -f ~/$SETUP_SENTINEL"; then
        echo "  $vm already set up, skipping"
        return 0
    fi
    echo "  Setting up $vm..."
    # Run setup in background to avoid SSH timeout on slow apt-get
    gssh "$vm" "nohup bash -c 'sudo apt-get update -y && \
        sudo DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential pkg-config libssl-dev git curl && \
        curl --proto =https --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y && \
        touch ~/$SETUP_SENTINEL' > ~/setup.log 2>&1 &"

    # Poll for completion (timeout after 10 minutes)
    local max_attempts=60
    local attempt=0
    while true; do
        if gssh "$vm" "test -f ~/$SETUP_SENTINEL" 2>/dev/null; then
            echo "  $vm setup complete"
            break
        fi
        attempt=$((attempt + 1))
        if [ "$attempt" -ge "$max_attempts" ]; then
            echo "ERROR: $vm setup timed out after $((max_attempts * 10))s"
            echo "  Check ~/setup.log on the VM for details"
            exit 1
        fi
        sleep 10
        echo "  $vm still setting up... (${attempt}/${max_attempts})"
    done
}

echo "Setting up VMs (first-run only)..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    setup_vm "cmt-stage${i}"
done
setup_vm "cmt-orch"
echo ""

# Clone, build, and download model on stage0
NEED_BUILD=false
if ! gssh "cmt-stage0" "test -f ~/$WORKSPACE/confidential-ml-pipeline/examples/gpt2-pipeline/target/release/stage-worker"; then
    NEED_BUILD=true
fi

if $NEED_BUILD; then
    echo "Cloning repos on stage0..."
    gssh "cmt-stage0" "mkdir -p ~/$WORKSPACE && cd ~/$WORKSPACE && \
        (test -d confidential-ml-transport || git clone https://github.com/cyntrisec/confidential-ml-transport.git) && \
        (test -d confidential-ml-pipeline || git clone https://github.com/cyntrisec/confidential-ml-pipeline.git)"

    echo "Downloading model on stage0..."
    gssh "cmt-stage0" "source ~/.cargo/env && cd ~/$WORKSPACE/confidential-ml-pipeline && \
        bash examples/gpt2-pipeline/scripts/download_model.sh"

    echo "Building binaries on stage0 (feature: $FEATURE)..."
    echo "  (This may take 5-10 minutes on first build...)"
    # Long cargo builds can exceed SSH timeouts. Use nohup + poll.
    gssh "cmt-stage0" "nohup bash -c 'source ~/.cargo/env && cd ~/$WORKSPACE/confidential-ml-pipeline && \
        cargo build --release --manifest-path examples/gpt2-pipeline/Cargo.toml \
        --no-default-features --features $FEATURE > ~/build.log 2>&1 && touch ~/build-done || touch ~/build-failed' &"

    # Poll for build completion
    echo "  Waiting for build to complete..."
    while true; do
        if gssh "cmt-stage0" "test -f ~/build-done" 2>/dev/null; then
            echo "  Build succeeded!"
            break
        fi
        if gssh "cmt-stage0" "test -f ~/build-failed" 2>/dev/null; then
            echo "  Build FAILED. Last 20 lines of build log:"
            gssh "cmt-stage0" "tail -20 ~/build.log" || true
            exit 1
        fi
        sleep 15
        echo "  Still building..."
    done
fi

# Paths on stage0
REMOTE_BIN_DIR="~/$WORKSPACE/confidential-ml-pipeline/examples/gpt2-pipeline/target/release"
REMOTE_MODEL_DIR="~/$WORKSPACE/confidential-ml-pipeline/examples/gpt2-pipeline/model"

# Copy binaries + model to other stage VMs
for i in $(seq 1 $((NUM_STAGES - 1))); do
    echo "Copying binaries + model to cmt-stage${i}..."
    gssh "cmt-stage0" "gcloud compute scp \
        $REMOTE_BIN_DIR/stage-worker \
        $REMOTE_MODEL_DIR/model.safetensors \
        $REMOTE_MODEL_DIR/config.json \
        $REMOTE_MODEL_DIR/tokenizer.json \
        cmt-stage${i}:~ --zone=$ZONE" 2>/dev/null || {
        # Fallback: use local machine as relay
        echo "  Direct VM-to-VM scp failed, using relay..."
        gcloud compute scp "cmt-stage0:$REMOTE_BIN_DIR/stage-worker" /tmp/stage-worker --zone="$ZONE"
        gcloud compute scp "cmt-stage0:$REMOTE_MODEL_DIR/model.safetensors" /tmp/model.safetensors --zone="$ZONE"
        gcloud compute scp "cmt-stage0:$REMOTE_MODEL_DIR/config.json" /tmp/config.json --zone="$ZONE"
        gcloud compute scp "cmt-stage0:$REMOTE_MODEL_DIR/tokenizer.json" /tmp/tokenizer.json --zone="$ZONE"
        gcloud compute scp /tmp/stage-worker /tmp/model.safetensors /tmp/config.json /tmp/tokenizer.json \
            "cmt-stage${i}:~" --zone="$ZONE"
    }
    gssh "cmt-stage${i}" "chmod +x ~/stage-worker && mkdir -p ~/model && \
        mv -f ~/model.safetensors ~/config.json ~/tokenizer.json ~/model/ 2>/dev/null || true"
done

# Copy binaries + tokenizer to orch VM
echo "Copying binaries + tokenizer to cmt-orch..."
gssh "cmt-stage0" "gcloud compute scp \
    $REMOTE_BIN_DIR/pipeline-orch \
    $REMOTE_MODEL_DIR/tokenizer.json \
    cmt-orch:~ --zone=$ZONE" 2>/dev/null || {
    echo "  Direct VM-to-VM scp failed, using relay..."
    gcloud compute scp "cmt-stage0:$REMOTE_BIN_DIR/pipeline-orch" /tmp/pipeline-orch --zone="$ZONE"
    gcloud compute scp "cmt-stage0:$REMOTE_MODEL_DIR/tokenizer.json" /tmp/tokenizer-orch.json --zone="$ZONE"
    gcloud compute scp /tmp/pipeline-orch /tmp/tokenizer-orch.json "cmt-orch:~" --zone="$ZONE"
    gssh "cmt-orch" "mv -f ~/tokenizer-orch.json ~/tokenizer.json 2>/dev/null || true"
}
gssh "cmt-orch" "chmod +x ~/pipeline-orch"
echo ""

# --- 5. Generate manifest ---
echo "Generating manifest..."

# Compute layer splits
TOTAL_LAYERS=12
LAYERS_PER_STAGE=$((TOTAL_LAYERS / NUM_STAGES))
REMAINDER=$((TOTAL_LAYERS % NUM_STAGES))

generate_manifest() {
    local manifest='{\n  "model_name": "gpt2",\n  "model_version": "1.0",\n  "total_layers": 12,\n  "stages": ['
    local layer_start=0

    for i in $(seq 0 $((NUM_STAGES - 1))); do
        local extra=0
        if [ "$i" -lt "$REMAINDER" ]; then
            extra=1
        fi
        local layer_end=$((layer_start + LAYERS_PER_STAGE + extra))

        local ctrl_port=$((9000 + i * 10))
        local din_port=$((9001 + i * 10))
        local dout_port=$((9011 + i * 10))

        local ip="${STAGE_IPS[$i]}"

        # data_out addr: next stage's data_in, or orch listener for last stage
        local dout_addr
        if [ "$i" -eq $((NUM_STAGES - 1)) ]; then
            dout_addr="$ORCH_IP:$dout_port"
        else
            local next_ip="${STAGE_IPS[$((i + 1))]}"
            local next_din_port=$((9001 + (i + 1) * 10))
            dout_addr="$next_ip:$next_din_port"
        fi

        if [ "$i" -gt 0 ]; then
            manifest+=','
        fi

        manifest+='\n    {'
        manifest+="\n      \"stage_idx\": $i,"
        manifest+="\n      \"layer_start\": $layer_start,"
        manifest+="\n      \"layer_end\": $layer_end,"
        manifest+='\n      "weight_hashes": [],'
        manifest+='\n      "expected_measurements": {},'
        manifest+='\n      "endpoint": {'
        manifest+="\n        \"control\": { \"type\": \"tcp\", \"addr\": \"$ip:$ctrl_port\" },"
        manifest+="\n        \"data_in\": { \"type\": \"tcp\", \"addr\": \"$ip:$din_port\" },"
        manifest+="\n        \"data_out\": { \"type\": \"tcp\", \"addr\": \"$dout_addr\" }"
        manifest+='\n      }'
        manifest+='\n    }'

        layer_start=$layer_end
    done

    manifest+='\n  ],'
    manifest+='\n  "activation_spec": {'
    manifest+='\n    "dtype": "F32",'
    manifest+='\n    "hidden_dim": 768,'
    manifest+='\n    "max_seq_len": 1024'
    manifest+='\n  }'
    manifest+='\n}'

    echo -e "$manifest"
}

MANIFEST_CONTENT=$(generate_manifest)
echo "$MANIFEST_CONTENT"
echo ""

# Upload manifest to all VMs
for i in $(seq 0 $((NUM_STAGES - 1))); do
    echo "$MANIFEST_CONTENT" | gcloud compute ssh "cmt-stage${i}" --zone="$ZONE" \
        --command="cat > ~/manifest.json" 2>/dev/null
done
echo "$MANIFEST_CONTENT" | gcloud compute ssh "cmt-orch" --zone="$ZONE" \
    --command="cat > ~/manifest.json" 2>/dev/null

# --- 6. Kill any leftover stage workers ---
echo "Killing any leftover stage workers..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    gssh "cmt-stage${i}" "pkill -f stage-worker || true"
done
sleep 1

# --- 7. Measure cross-VM RTT ---
echo "Measuring cross-VM RTT..."
if [ "$NUM_STAGES" -gt 1 ]; then
    gssh "cmt-stage0" "ping -c 3 -q ${STAGE_IPS[1]}" || true
fi
echo ""

# --- 8. Start stage workers ---
echo "Starting stage workers..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    local_ip="${STAGE_IPS[$i]}"
    dout_port=$((9011 + i * 10))

    # Determine data_out_target
    if [ "$i" -eq $((NUM_STAGES - 1)) ]; then
        DOUT_TARGET="$ORCH_IP:$dout_port"
    else
        next_ip="${STAGE_IPS[$((i + 1))]}"
        next_din=$((9001 + (i + 1) * 10))
        DOUT_TARGET="$next_ip:$next_din"
    fi

    # Determine model dir and binary path
    if [ "$i" -eq 0 ]; then
        STAGE_MODEL="$WORKSPACE/confidential-ml-pipeline/examples/gpt2-pipeline/model"
        STAGE_BIN="$WORKSPACE/confidential-ml-pipeline/examples/gpt2-pipeline/target/release/stage-worker"
    else
        STAGE_MODEL="model"
        STAGE_BIN="stage-worker"
    fi

    # Build taskset prefix if --taskset was specified
    local taskset_prefix=""
    if [ -n "$TASKSET_CORES" ]; then
        taskset_prefix="taskset -c $TASKSET_CORES"
        echo "  Stage $i: ctrl=${local_ip}:$((9000 + i*10)), data_out_target=$DOUT_TARGET (pinned to cores $TASKSET_CORES)"
    else
        echo "  Stage $i: ctrl=${local_ip}:$((9000 + i*10)), data_out_target=$DOUT_TARGET"
    fi
    gssh "cmt-stage${i}" "nohup $taskset_prefix ~/$STAGE_BIN \
        --manifest ~/manifest.json \
        --stage-idx $i \
        --data-out-target $DOUT_TARGET \
        --model-dir ~/$STAGE_MODEL \
        > ~/stage${i}.log 2>&1 &"
done

echo "Waiting for model loading..."
sleep 5
echo ""

# --- 9. Run orchestrator ---
LAST_DOUT_PORT=$((9011 + (NUM_STAGES - 1) * 10))
DOUT_LISTEN="$ORCH_IP:$LAST_DOUT_PORT"

echo "Starting orchestrator (data_out_listen=$DOUT_LISTEN)..."
echo "Prompt: \"$PROMPT\""
echo "---"

gssh "cmt-orch" "RUST_LOG=error ~/pipeline-orch \
    --manifest ~/manifest.json \
    --tokenizer ~/tokenizer.json \
    --text '$PROMPT' \
    --max-tokens $MAX_TOKENS \
    --data-out-listen $DOUT_LISTEN \
    --latency-out ~/latency.json"

echo "---"

# --- 10. Download latency results ---
if [ -n "$LATENCY_OUT" ]; then
    echo "Downloading latency data..."
    gcloud compute scp "cmt-orch:~/latency.json" "$LATENCY_OUT" --zone="$ZONE"
    echo "Saved to $LATENCY_OUT"
else
    echo "Downloading latency data..."
    gcloud compute scp "cmt-orch:~/latency.json" /tmp/gcp_latency.json --zone="$ZONE"
    echo "Saved to /tmp/gcp_latency.json"
fi

# --- 11. Kill stage workers ---
echo "Stopping stage workers..."
for i in $(seq 0 $((NUM_STAGES - 1))); do
    gssh "cmt-stage${i}" "pkill -f stage-worker || true"
done

echo "Done."
