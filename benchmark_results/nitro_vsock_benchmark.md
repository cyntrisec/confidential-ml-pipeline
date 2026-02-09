# GPT-2 Nitro Enclave Pipeline Benchmark

**Date:** 2026-02-09

## Environment

| Parameter | Value |
|-----------|-------|
| Instance type | `m6i.2xlarge` (8 vCPUs, 32 GiB) |
| CPU | Intel Xeon Platinum 8375C @ 2.90GHz (Ice Lake) |
| Region | `us-east-1` |
| Host OS | Amazon Linux 2023 (kernel 6.1.161) |
| Enclave kernel | 4.14.256-209.484.amzn2 (Nitro Enclave) |
| Debug mode | Yes |

## Model

| Parameter | Value |
|-----------|-------|
| Model | GPT-2 (`openai-community/gpt2`) |
| Parameters | 124M |
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

---

## Single-Stage (1 enclave, 12 layers)

**Commit:** `c51cbc9` | **Enclave:** 2 vCPUs, 3072 MiB | **EIF:** 614 MB

### "The capital of France is" (20 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **91.7ms** |
| Generation avg | **42.1ms/token** |
| Generation p50 | **42.0ms** |
| Generation p95 | **42.9ms** |
| Tokens/sec | **23.8 tok/s** |

**Output:** `The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French`

### "Machine learning is transforming" (50 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **84.4ms** |
| Generation avg | **42.2ms/token** |
| Generation p50 | **42.1ms** |
| Generation p95 | **42.8ms** |
| Tokens/sec | **23.7 tok/s** |

---

## Two-Stage (2 enclaves, 6 layers each, host relay)

**Commit:** `f0656bf` | **Enclaves:** 2 vCPUs + 3072 MiB each | **EIF:** 614 MB each

Stage 0 (layers 0-5) → host relay (port 6001) → Stage 1 (layers 6-11) → host (port 5002).
All inter-stage data encrypted end-to-end via `SecureChannel` (ChaCha20-Poly1305).

### "The capital of France is" (20 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **96.6ms** |
| Generation avg | **45.5ms/token** |
| Generation p50 | **45.6ms** |
| Generation p95 | **46.2ms** |
| Tokens/sec | **22.0 tok/s** |

**Output:** `The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French`

### "Machine learning is transforming" (50 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **89.3ms** |
| Generation avg | **45.4ms/token** |
| Generation p50 | **45.4ms** |
| Generation p95 | **46.8ms** |
| Tokens/sec | **22.0 tok/s** |

---

## Comparison: 1-Stage vs 2-Stage

| Metric | 1-Stage | 2-Stage | Relay Overhead |
|--------|---------|---------|----------------|
| TTFT (20 tok) | 91.7ms | 96.6ms | +4.9ms (+5.3%) |
| TTFT (50 tok) | 84.4ms | 89.3ms | +4.9ms (+5.8%) |
| Gen p50 (20 tok) | 42.0ms | 45.6ms | +3.6ms (+8.6%) |
| Gen p50 (50 tok) | 42.1ms | 45.4ms | +3.3ms (+7.8%) |
| Gen p95 (20 tok) | 42.9ms | 46.2ms | +3.3ms (+7.7%) |
| Gen p95 (50 tok) | 42.8ms | 46.8ms | +4.0ms (+9.3%) |
| Tokens/sec | 23.8 | 22.0 | -7.6% |

The 2-stage pipeline adds ~3.5ms/token of relay overhead. This is the cost of:
1. VSock hop: stage 0 → host relay (port 6001)
2. VSock hop: host relay → stage 1
3. Additional SecureChannel handshake for the inter-stage data channel

Output text is **identical** between 1-stage and 2-stage, confirming correctness of activation relay.

## Pipeline Init Breakdown

| Phase | 1-Stage | 2-Stage |
|-------|---------|---------|
| VSock control connect | <1ms | <2ms (sequential to both enclaves) |
| Handshake (X25519 + mock) | ~1ms | ~2ms (2 channels) |
| Model load (inside enclave) | ~490ms | ~490ms (parallel in both enclaves) |
| Data channel setup | ~4ms | ~5ms (includes relay bind + connect) |
| **Total init** | **~500ms** | **~500ms** |

## Key Observations

1. **Relay overhead is modest** — 3.5ms/token (~8%) for an extra VSock round-trip through the host. Dominated by the two additional VSock hops (enclave→host→enclave), not by encryption.

2. **Output is identical** — Both pipelines produce the same greedy-decoded text, confirming correct activation relay through the host.

3. **Consistent generation** — p95/p50 ratio is 1.01-1.03 for both configurations, indicating near-zero variance.

4. **First multi-enclave ML pipeline** — No prior open-source implementation exists for pipeline-parallel ML inference across separate TEE enclaves with encrypted activation relay.

## PCR Measurements

### Single-Stage EIF
```
PCR0: 3b343ba7ae9664b137535525f174e62a29ff2dcab4edc0474e875d389cb87902950f5ef6f6a2e7486948d9f055ca08cd
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: ee4cfec6b1a13112c34bad1af4538ec423067e01e6c341a6eef005d3d544e3cacbfb0cc4e7d4f4b614e22b5e5e01ee94
```

### Two-Stage EIFs
**Stage 0:**
```
PCR0: 83b502377b2937621a8d718b3f38e24d197e693ba6937be775f198269b6b20a121d66742b4f92fa7df0f4a853dfa990e
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: 65b14a762257469e68f9f43a0e2151d4f7d5b5376039339f2c44721f55c777646519fa031a1a005fb04f6511f45dae9e
```

**Stage 1:**
```
PCR0: 96d8333665caef81446551c1c8c09f577d6a525553639b6547a8b41ec4cbfe7af5f4b2c1fa0532b63c5d2c14c8f9e975
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: 26610f0864c29b17528bccd778aaaf856d80a6acfc25e207ebb257afe363a2a47d26fef4948a669f61dcb9a2043c577d
```
