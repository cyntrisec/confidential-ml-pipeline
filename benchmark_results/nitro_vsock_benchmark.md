# GPT-2 Nitro Enclave Pipeline Benchmark

**Date:** 2026-02-09
**Commit:** `c51cbc9`

## Environment

| Parameter | Value |
|-----------|-------|
| Instance type | `m6i.2xlarge` (8 vCPUs, 32 GiB) |
| CPU | Intel Xeon Platinum 8375C @ 2.90GHz (Ice Lake) |
| Region | `us-east-1c` |
| Host OS | Amazon Linux 2023 (kernel 6.1.161) |
| Enclave kernel | 4.14.256-209.484.amzn2 (Nitro Enclave) |
| Enclave vCPUs | 2 |
| Enclave memory | 3072 MiB |
| EIF size | 614 MB |
| Debug mode | Yes |

## Model

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 (`openai-community/gpt2`) |
| Parameters | 124M |
| Layers | 12 (all in single enclave) |
| Hidden dim | 768 |
| Vocab size | 50,257 |
| Format | SafeTensors (523 MB) |
| KV-cache | Enabled |

## Transport

| Parameter | Value |
|-----------|-------|
| Protocol | VSock (host <-> enclave) |
| Encryption | ChaCha20-Poly1305 (AEAD per frame) |
| Key exchange | X25519 ECDH + HPKE derivation |
| Attestation | Mock (VSock transport verified) |
| Framing | Binary tensor (`confidential-ml-transport` v0.2.0) |

## Results

### Run 1: "The capital of France is" (20 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **91.7ms** |
| Generation avg | **42.1ms/token** |
| Generation p50 | **42.0ms** |
| Generation p95 | **42.9ms** |
| Tokens/sec | **23.8 tok/s** |

**Output:** `The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French`

### Run 2: "Machine learning is transforming" (50 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **84.4ms** |
| Generation avg | **42.2ms/token** |
| Generation p50 | **42.1ms** |
| Generation p95 | **42.8ms** |
| Tokens/sec | **23.7 tok/s** |

**Output:** `Machine learning is transforming the way we think about the world. The world is changing, and we need to change it. We need to change the way we think about the world. We need to change the way we think about the world.`

### Pipeline Init Breakdown

| Phase | Latency |
|-------|---------|
| VSock control connect | <1ms |
| Handshake (X25519 + mock attestation) | ~1ms |
| Model load (inside enclave) | ~818ms |
| Data channel setup | ~4ms |
| **Total init** | **~824ms** |

## Key Observations

1. **Token generation is remarkably consistent** — p95/p50 ratio is 1.02, indicating near-zero variance. The 42ms/token enclave latency is dominated by the forward pass, not transport overhead.

2. **TTFT includes model-load-on-first-inference** — the 84-92ms TTFT contains the first forward pass with KV-cache population. Subsequent tokens are steady at ~42ms.

3. **VSock overhead is negligible** — handshake completes in <2ms, and per-frame encryption (ChaCha20-Poly1305) adds <0.1ms per token based on prior transport benchmarks.

4. **Pipeline initialization is dominated by model loading** — the ~818ms model load inside the enclave (reading 523MB SafeTensors from the EIF ramdisk) is the bottleneck, not the transport layer.

## PCR Measurements

```
PCR0: 3b343ba7ae9664b137535525f174e62a29ff2dcab4edc0474e875d389cb87902950f5ef6f6a2e7486948d9f055ca08cd
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: ee4cfec6b1a13112c34bad1af4538ec423067e01e6c341a6eef005d3d544e3cacbfb0cc4e7d4f4b614e22b5e5e01ee94
```
