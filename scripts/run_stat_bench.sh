#!/bin/bash
# Statistical benchmark runner: 5 runs × 3 configs = 15 total runs
# Fixed: "The capital of France is" (5 prompt tokens), 20 generated tokens
# Each run: start instance → upload runner → launch enclaves → benchmark → force stop
set -euo pipefail

INSTANCE_ID="$1"
KEY_PATH="$2"
RESULTS_DIR="$3"
REGION="us-east-1"
PROMPT="The capital of France is"
MAX_TOKENS=20
RUNS=5
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

mkdir -p "$RESULTS_DIR"

get_ip() {
  aws ec2 describe-instances --region "$REGION" --instance-ids "$INSTANCE_ID" \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
}

wait_ssh() {
  local ip="$1"
  for i in $(seq 1 20); do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
         -i "$KEY_PATH" ec2-user@"$ip" "true" 2>/dev/null; then
      return 0
    fi
    sleep 5
  done
  echo "ERROR: SSH timeout after 100s" >&2
  return 1
}

# Main loop
for config in 1stage 2stage 3stage; do
  for run in $(seq 1 $RUNS); do
    echo ""
    echo "=============================="
    echo "  $config run $run / $RUNS"
    echo "=============================="

    # Start instance
    echo "Starting instance..."
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
    IP=$(get_ip)
    echo "Instance running at $IP"

    # Wait for SSH
    wait_ssh "$IP"
    echo "SSH ready"

    # Upload runner script
    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
      "${SCRIPT_DIR}/enclave_run.sh" ec2-user@"$IP":~/enclave_run.sh

    # Run benchmark
    echo "--- Config: ${config}, Run: ${run}, IP: ${IP} ---"
    ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ec2-user@"$IP" \
      "bash ~/enclave_run.sh $config '$PROMPT' $MAX_TOKENS"

    # Download latency file
    local_file="${RESULTS_DIR}/${config}_run${run}.json"
    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
      ec2-user@"$IP":~/latency_run.json "$local_file"
    echo "Saved: $local_file"

    # Force stop
    echo "Stopping instance..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --force >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "Instance stopped"
  done
done

echo ""
echo "=== ALL 15 RUNS COMPLETE ==="
echo "Results in: $RESULTS_DIR"
