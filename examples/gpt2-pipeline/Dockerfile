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

# Copy manifest (configurable: 1-stage or 2-stage)
ARG MANIFEST=manifest_2stage_vsock.json
COPY confidential-ml-pipeline/examples/gpt2-pipeline/manifests/${MANIFEST} /app/manifest.json

# Copy init script
COPY confidential-ml-pipeline/examples/gpt2-pipeline/init.sh /app/init.sh
RUN chmod +x /app/init.sh

# Stage index (each EIF gets its own value)
ARG STAGE_IDX=0
ENV STAGE_IDX=${STAGE_IDX}

ENTRYPOINT ["/app/init.sh"]
