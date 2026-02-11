#!/bin/bash
# Clean up GCP VMs and firewall rules created by run_gcp.sh / bench_gcp.sh.
# Usage: cleanup_gcp.sh [--zone ZONE]
set -euo pipefail

ZONE="us-central1-a"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --zone) ZONE="$2"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

echo "=== GCP Pipeline Cleanup ==="
echo "Zone: $ZONE"
echo ""

# Delete VMs
VMS_TO_DELETE=()
for vm in cmt-stage0 cmt-stage1 cmt-stage2 cmt-orch; do
    if gcloud compute instances describe "$vm" --zone="$ZONE" &>/dev/null; then
        VMS_TO_DELETE+=("$vm")
    fi
done

if [ ${#VMS_TO_DELETE[@]} -gt 0 ]; then
    echo "Deleting VMs: ${VMS_TO_DELETE[*]}"
    gcloud compute instances delete "${VMS_TO_DELETE[@]}" --zone="$ZONE" --quiet
else
    echo "No VMs to delete."
fi

# Delete firewall rule
if gcloud compute firewall-rules describe allow-pipeline-ports &>/dev/null; then
    echo "Deleting firewall rule: allow-pipeline-ports"
    gcloud compute firewall-rules delete allow-pipeline-ports --quiet
else
    echo "No firewall rule to delete."
fi

echo ""
echo "Cleanup complete."
