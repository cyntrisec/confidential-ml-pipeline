# GPT-2 Pipeline Example

End-to-end confidential ML inference example that shards GPT-2 small (124M params, 12 layers) across multiple pipeline stages, with activation tensors flowing through encrypted `SecureChannel` connections.

## Architecture

```
Orchestrator                    Stage 0                     Stage 1
tokenize("Hello")              (embed + layers 0-5)        (layers 6-11 + ln_f + lm_head)
    │                               │                           │
    ├─ input_ids [1,T] U32 ────►    │                           │
    │                          embed + forward(0-5)             │
    │                               │                           │
    │                    hidden [1,T,768] F32 ──►relay──►       │
    │                                                      forward(6-11) + ln_f + lm_head
    │                                                           │
    │◄─────────────────────── logits [1,50257] F32 ─────────────┤
argmax → next token
```

Each stage runs as a separate process. All inter-stage communication is encrypted via `SecureChannel`.

## Quick Start

### 1. Download Model (~500MB)

```bash
bash scripts/download_model.sh
```

### 2. Run 2-Stage Pipeline (TCP, localhost)

```bash
bash scripts/run_local.sh "The capital of France is"
```

### 3. Run 3-Stage Pipeline

```bash
bash scripts/run_local_3stage.sh "Once upon a time" 30
```

## Manual Run

```bash
# Build
cargo build --release --manifest-path Cargo.toml

# Terminal 1: Stage 0
RUST_LOG=info target/release/stage-worker \
  --manifest manifests/manifest_2stage.json \
  --stage-idx 0 \
  --data-out-target 127.0.0.1:9011 \
  --model-dir model

# Terminal 2: Stage 1
RUST_LOG=info target/release/stage-worker \
  --manifest manifests/manifest_2stage.json \
  --stage-idx 1 \
  --data-out-target 127.0.0.1:9021 \
  --model-dir model

# Terminal 3: Orchestrator
RUST_LOG=info target/release/pipeline-orch \
  --manifest manifests/manifest_2stage.json \
  --data-out-listen 127.0.0.1:9021 \
  --tokenizer model/tokenizer.json \
  --text "The capital of France is" \
  --max-tokens 20
```

## Model Details

| Metric | Value |
|--------|-------|
| Model | GPT-2 Small (openai-community/gpt2) |
| Parameters | 124M |
| Layers | 12 |
| Hidden dim | 768 |
| Attention heads | 12 |
| Vocab size | 50257 |
| Weights format | SafeTensors (~500MB) |

## Benchmarks

Greedy decoding, 20 tokens, KV-cache enabled, TCP mock transport, Intel Core i7-8565U @ 1.80GHz:

| Metric | 2-stage | 3-stage |
|--------|---------|---------|
| Prompt (TTFT) | 71.9ms | 64.9ms |
| Generation avg | 45.9ms/tok | 47.3ms/tok |
| Generation p50 | 43.3ms | 46.6ms |
| Generation p95 | 59.3ms | 60.5ms |

Run `bash scripts/bench.sh` to reproduce.

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_model.sh` | Download GPT-2 from HuggingFace (~500MB) |
| `scripts/run_local.sh` | 2-stage TCP test on localhost |
| `scripts/run_local_3stage.sh` | 3-stage TCP test on localhost |
| `scripts/bench.sh` | Benchmark 2-stage and 3-stage, output markdown table |
| `scripts/check_quality.sh` | Quality checks (geography, continuation, determinism) |
| `scripts/test_failure.sh` | Kill a stage mid-generation, verify graceful recovery |

## Nitro Enclave Deployment

### Benchmarks (m6i.2xlarge, 2-vCPU enclaves)

Greedy decoding, KV-cache enabled, encrypted VSock transport (ChaCha20-Poly1305):

| Metric | 1-stage (12 layers) | 2-stage (6+6 layers) | 3-stage (4+4+4 layers) |
|--------|---------------------|----------------------|------------------------|
| TTFT | 91.7ms | 96.6ms | 99.6ms |
| Generation p50 | 42.0ms/tok | 45.6ms/tok | 45.2ms/tok |
| Generation p95 | 42.9ms/tok | 46.2ms/tok | 46.2ms/tok |
| Tokens/sec | 23.8 | 22.0 | 22.1 |
| Relay overhead | — | +3.5ms/tok (+8%) | +3.2ms/tok (+8%) |

### Reproduce (commit `61cb135`)

**Prerequisites:** AWS account, `aws` CLI configured, `nitro-cli` on an enclave-enabled instance.

```bash
# 1. Launch m6i.2xlarge with Nitro Enclave support
aws ec2 run-instances --region us-east-1 \
  --image-id ami-0c1fe732b5494dc14 \
  --instance-type m6i.2xlarge \
  --key-name YOUR_KEY \
  --enclave-options 'Enabled=true' \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":50,"VolumeType":"gp3"}}]'

# 2. SSH in and install dependencies
sudo dnf install -y docker aws-nitro-enclaves-cli aws-nitro-enclaves-cli-devel git gcc gcc-c++ openssl-devel jq
sudo sed -i 's/^memory_mib:.*/memory_mib: 3072/' /etc/nitro_enclaves/allocator.yaml
sudo sed -i 's/^cpu_count:.*/cpu_count: 4/' /etc/nitro_enclaves/allocator.yaml
sudo systemctl enable --now docker nitro-enclaves-allocator.service
sudo usermod -aG docker,ne ec2-user
# Log out and back in for group changes

# 3. Install Rust and clone repos
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env
git clone https://github.com/cyntrisec/confidential-ml-transport.git ~/cmt
git clone https://github.com/cyntrisec/confidential-ml-pipeline.git ~/cmp
cd ~/cmp && git checkout 61cb135

# 4. Download model
mkdir -p ~/cmp/examples/gpt2-pipeline/model
cd ~/cmp/examples/gpt2-pipeline/model
for f in config.json tokenizer.json model.safetensors; do
  curl -sL "https://huggingface.co/openai-community/gpt2/resolve/main/$f" -o "$f"
done

# 5. Build orchestrator (runs on host)
cd ~/cmp/examples/gpt2-pipeline
cargo build --release --bin pipeline-orch --no-default-features --features vsock-mock

# 6. Build Docker image + EIF (builds stage-worker inside container)
mkdir -p ~/workspace
cp -r ~/cmt ~/workspace/confidential-ml-transport
cp -r ~/cmp ~/workspace/confidential-ml-pipeline
cd ~/workspace
docker build -f confidential-ml-pipeline/examples/gpt2-pipeline/Dockerfile \
  -t gpt2-pipeline-enclave:latest .
nitro-cli build-enclave --docker-uri gpt2-pipeline-enclave:latest \
  --output-file ~/gpt2-pipeline.eif

# 7. Launch enclave
RESULT=$(nitro-cli run-enclave --eif-path ~/gpt2-pipeline.eif \
  --memory 2560 --cpu-count 2 --enclave-name gpt2-stage0 --debug-mode)
CID=$(echo "$RESULT" | grep -oP '"EnclaveCID":\s*\K[0-9]+')
sleep 12  # wait for boot + model load

# 8. Generate manifest and run benchmark
cat > ~/manifest.json <<EOF
{
  "model_name": "gpt2", "model_version": "1.0", "total_layers": 12,
  "stages": [{
    "stage_idx": 0, "layer_start": 0, "layer_end": 12,
    "weight_hashes": [], "expected_measurements": {},
    "endpoint": {
      "control": {"type": "vsock", "cid": $CID, "port": 5000},
      "data_in": {"type": "vsock", "cid": $CID, "port": 5001},
      "data_out": {"type": "vsock", "cid": 3, "port": 5002}
    }
  }],
  "activation_spec": {"dtype": "F32", "hidden_dim": 768, "max_seq_len": 1024}
}
EOF

RUST_LOG=info target/release/pipeline-orch \
  --manifest ~/manifest.json \
  --data-out-port 5002 \
  --tokenizer model/tokenizer.json \
  --text "The capital of France is" \
  --max-tokens 20 \
  --latency-out ~/latency.json
```

### Enclave Configuration

| Parameter | Value |
|-----------|-------|
| Instance type | m6i.2xlarge (8 vCPUs, 32 GiB) |
| CPU | Intel Xeon Platinum 8375C @ 2.90GHz |
| Enclave vCPUs | 2 |
| Enclave memory | 3072 MiB |
| EIF size | 614 MB |
| Host OS | Amazon Linux 2023 (kernel 6.1) |
| Enclave kernel | 4.14.256 (Nitro) |

### Scripts

| Script | Purpose |
|--------|---------|
| `scripts/build_nitro.sh` | Download model, build Docker image, build EIF |
| `scripts/run_nitro.sh` | Launch 2 enclaves, generate manifest, run orchestrator |

## Notes

- **KV-cache**: After the initial prompt, only the new token is sent per step. A cache-clear sentinel (U32 tensor with shape `[0]`) resets KV state between independent requests.
- **Greedy decoding**: Uses argmax (no sampling/temperature).
- **Conv1D weights**: GPT-2 stores linear weights as `[in, out]`. The loader transposes them for candle-nn's `[out, in]` convention.
- **Tied lm_head**: GPT-2 ties the output projection to the token embedding weights (`wte.weight`).
