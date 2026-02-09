# Stage 1: Build the enclave binary
FROM rust:1.93-bookworm AS builder
RUN apt-get update && apt-get install -y libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*
WORKDIR /build

# Copy workspace
COPY confidential-ml-transport/ confidential-ml-transport/
COPY confidential-ml-pipeline/ confidential-ml-pipeline/

# Build stage-worker with vsock-mock features (mock attestation over real VSock)
RUN cargo build --release --bin stage-worker \
    --manifest-path confidential-ml-pipeline/examples/gpt2-pipeline/Cargo.toml \
    --no-default-features --features vsock-mock

# Stage 2: Runtime image (minimal)
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y ca-certificates libssl3 && rm -rf /var/lib/apt/lists/*

# Copy binary
COPY --from=builder /build/confidential-ml-pipeline/examples/gpt2-pipeline/target/release/stage-worker /app/stage-worker

# Copy model files (pre-downloaded by build_nitro.sh)
COPY confidential-ml-pipeline/examples/gpt2-pipeline/model/ /model/

# Copy manifest for single-stage pipeline (ports are fixed, CIDs don't matter inside enclave)
COPY confidential-ml-pipeline/examples/gpt2-pipeline/manifests/manifest_1stage_vsock.json /app/manifest.json

# Copy init script
COPY confidential-ml-pipeline/examples/gpt2-pipeline/init.sh /app/init.sh
RUN chmod +x /app/init.sh

# Default env vars for single-stage pipeline
ENV STAGE_IDX=0
ENV DATA_OUT_CID=3
ENV DATA_OUT_PORT=5002

ENTRYPOINT ["/app/init.sh"]
