#!/bin/bash
# PID1 wrapper for Nitro Enclave
# Stage index and data-out target passed via env vars
exec /app/stage-worker \
    --manifest /app/manifest.json \
    --stage-idx "${STAGE_IDX}" \
    --data-out-cid "${DATA_OUT_CID}" \
    --data-out-port "${DATA_OUT_PORT}" \
    --model-dir /model
