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

## Notes

- **KV-cache**: After the initial prompt, only the new token is sent per step. A cache-clear sentinel (U32 tensor with shape `[0]`) resets KV state between independent requests.
- **Greedy decoding**: Uses argmax (no sampling/temperature).
- **Conv1D weights**: GPT-2 stores linear weights as `[in, out]`. The loader transposes them for candle-nn's `[out, in]` convention.
- **Tied lm_head**: GPT-2 ties the output projection to the token embedding weights (`wte.weight`).
