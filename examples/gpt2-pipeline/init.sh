#!/bin/bash
# PID1 wrapper for Nitro Enclave
# STAGE_IDX is baked in via Dockerfile ENV.
# data_out target is read from the manifest by the stage-worker binary.
exec /app/stage-worker \
    --manifest /app/manifest.json \
    --stage-idx "${STAGE_IDX}" \
    --model-dir /model
