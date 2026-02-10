#!/bin/bash
# Resume statistical benchmark from 2stage
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

for config in 2stage 3stage; do
  for run in $(seq 1 $RUNS); do
    echo ""
    echo "=============================="
    echo "  $config run $run / $RUNS"
    echo "=============================="

    echo "Starting instance..."
    aws ec2 start-instances --instance-ids "$INSTANCE_ID" --region "$REGION" >/dev/null
    aws ec2 wait instance-running --instance-ids "$INSTANCE_ID" --region "$REGION"
    IP=$(get_ip)
    echo "Instance running at $IP"

    wait_ssh "$IP"
    echo "SSH ready"

    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
      "${SCRIPT_DIR}/enclave_run.sh" ec2-user@"$IP":~/enclave_run.sh

    echo "--- Config: ${config}, Run: ${run}, IP: ${IP} ---"
    ssh -o StrictHostKeyChecking=no -i "$KEY_PATH" ec2-user@"$IP" \
      "bash ~/enclave_run.sh $config '$PROMPT' $MAX_TOKENS"

    local_file="${RESULTS_DIR}/${config}_run${run}.json"
    scp -o StrictHostKeyChecking=no -i "$KEY_PATH" \
      ec2-user@"$IP":~/latency_run.json "$local_file"
    echo "Saved: $local_file"

    echo "Stopping instance..."
    aws ec2 stop-instances --instance-ids "$INSTANCE_ID" --region "$REGION" --force >/dev/null
    aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID" --region "$REGION"
    echo "Instance stopped"
  done
done

echo ""
echo "=== ALL REMAINING RUNS COMPLETE ==="
