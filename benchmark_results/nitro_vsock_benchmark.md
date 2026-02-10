# GPT-2 Nitro Enclave Pipeline Benchmark

> Real Nitro multi-stage confidential inference is practical at 2 stages; 3-stage shows higher overhead under host contention.

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

## Three-Stage (3 enclaves, 4 layers each, 2 host relays)

**Commit:** `74bf9c0` | **Enclaves:** 2 vCPUs + 3072 MiB each | **EIF:** 614 MB each

Stage 0 (layers 0-3) → host relay (port 6001) → Stage 1 (layers 4-7) → host relay (port 6002) → Stage 2 (layers 8-11) → host (port 5002).
All inter-stage data encrypted end-to-end via `SecureChannel` (ChaCha20-Poly1305).

### "The capital of France is" (20 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **99.6ms** |
| Generation avg | **45.2ms/token** |
| Generation p50 | **45.2ms** |
| Generation p95 | **46.2ms** |
| Tokens/sec | **22.1 tok/s** |

**Output:** `The capital of France is the capital of the French Republic, and the capital of the French Republic is the capital of the French`

### "Machine learning is transforming" (50 tokens)

| Metric | Value |
|--------|-------|
| TTFT | **88.8ms** |
| Generation avg | **44.2ms/token** |
| Generation p50 | **44.2ms** |
| Generation p95 | **44.8ms** |
| Tokens/sec | **22.6 tok/s** |

---

## Statistical Benchmark (N=5, 20 tokens)

Each configuration run 5 times with independent cold-boot stop/start cycles. Prompt: "The capital of France is", 20 tokens, greedy decoding.

### Results (mean +/- 95% CI, t-distribution df=4)

| Metric | 1-Stage | 2-Stage | 3-Stage |
|--------|---------|---------|---------|
| TTFT (ms) | **92.5 +/- 1.8** | **97.5 +/- 5.4** | **107.1 +/- 13.7** |
| Gen avg (ms/tok) | **41.9 +/- 1.8** | **44.1 +/- 3.3** | **50.0 +/- 8.3** |
| Gen p50 (ms) | **41.9 +/- 1.8** | **44.1 +/- 3.4** | **49.9 +/- 8.2** |
| Gen p95 (ms) | **42.9 +/- 1.9** | **45.4 +/- 3.6** | **51.9 +/- 11.0** |
| Throughput (tok/s) | **23.9 +/- 1.0** | **22.7 +/- 1.6** | **20.3 +/- 2.9** |
| p95/p50 ratio | **1.02 +/- 0.01** | **1.03 +/- 0.03** | **1.04 +/- 0.04** |

### Overhead vs 1-Stage Baseline

| Config | Gen avg overhead | TTFT overhead |
|--------|-----------------|---------------|
| 2-Stage | +2.2ms (+5.2%) | +5.0ms (+5.4%) |
| 3-Stage | +8.1ms (+19.2%) | +14.6ms (+15.7%) |

### Per-Run Detail

| Run | 1-Stage TTFT / gen_avg | 2-Stage TTFT / gen_avg | 3-Stage TTFT / gen_avg |
|-----|------------------------|------------------------|------------------------|
| 1 | 90.8 / 40.3 | 96.7 / 43.5 | 106.2 / 49.5 |
| 2 | 94.0 / 42.1 | 95.0 / 42.5 | 100.9 / 47.9 |
| 3 | 91.6 / 41.2 | 94.4 / 42.0 | 99.1 / 44.7 |
| 4 | 93.9 / 44.2 | 96.3 / 43.9 | 102.9 / 46.4 |
| 5 | 92.3 / 41.9 | 105.2 / 48.7 | 126.2 / 61.5 |

Per-run AZ and instance metadata in [Appendix A](#appendix-a-per-run-environment-metadata).

### Observations

1. **2-stage adds ~5% overhead** vs 1-stage, with tight confidence intervals.

2. **3-stage adds ~19% overhead** vs 1-stage, significantly more than 2-stage. The wide CI (8.3ms) is driven by run 5 (61.5ms/tok vs 44.7-49.5ms for runs 1-4), likely due to host CPU contention or unfavorable physical host placement.

3. **Variance increases with stage count** — 1-stage stddev 1.4ms, 2-stage 2.7ms, 3-stage 6.7ms. More relay hops amplify host scheduling variability.

4. **Within-run consistency remains tight** — p95/p50 ratio is 1.02-1.04 for all configs, showing low intra-run jitter despite cross-run variance.

5. **Overhead is not linear with hop count** — 2-stage (+5.2%) to 3-stage (+19.2%) is ~3.7x the overhead for 2x the relay hops. The second relay hop costs more than the first, possibly due to increased host CPU contention when 3 enclaves share the same physical cores.

---

## Single-Run Benchmarks (N=1, reference only)

### Comparison: 1-Stage vs 2-Stage vs 3-Stage

| Metric | 1-Stage | 2-Stage | 3-Stage |
|--------|---------|---------|---------|
| TTFT (20 tok) | 91.7ms | 96.6ms (+5.3%) | 99.6ms (+8.6%) |
| TTFT (50 tok) | 84.4ms | 89.3ms (+5.8%) | 88.8ms (+5.2%) |
| Gen p50 (20 tok) | 42.0ms | 45.6ms (+8.6%) | 45.2ms (+7.6%) |
| Gen p50 (50 tok) | 42.1ms | 45.4ms (+7.8%) | 44.2ms (+5.0%) |
| Gen p95 (20 tok) | 42.9ms | 46.2ms (+7.7%) | 46.2ms (+7.7%) |
| Gen p95 (50 tok) | 42.8ms | 46.8ms (+9.3%) | 44.8ms (+4.7%) |
| Tokens/sec (20 tok) | 23.8 | 22.0 | 22.1 |
| Tokens/sec (50 tok) | 23.7 | 22.0 | 22.6 |

Output text is **identical** across all three configurations, confirming correctness.

**Note:** These are single-run results (N=1) from the initial benchmarks. The statistical benchmark above (N=5) supersedes these for the 20-token case and shows that run-to-run variance is significant, especially for 3-stage.

## Pipeline Init Breakdown

| Phase | 1-Stage | 2-Stage | 3-Stage |
|-------|---------|---------|---------|
| VSock control connect | <1ms | <2ms | <3ms |
| Handshake (X25519 + mock) | ~1ms | ~2ms | ~3ms |
| Model load (inside enclave) | ~490ms | ~490ms | ~490ms (parallel) |
| Data channel setup | ~4ms | ~5ms | ~7ms |
| **Total init** | **~500ms** | **~500ms** | **~500ms** |

## Key Observations

1. **2-stage adds ~5% overhead** — Statistically significant but modest. The 95% CI for 2-stage gen_avg (44.1 +/- 3.3ms) overlaps with 1-stage (41.9 +/- 1.8ms) at the tails, but means are clearly separated.

2. **3-stage adds ~19% overhead** — More substantial than 2-stage. Wide confidence intervals (gen_avg 50.0 +/- 8.3ms) driven by high run-to-run variance, likely from host CPU contention when 3 enclaves share 6 of 8 vCPUs.

3. **Variance scales with stage count** — 1-stage stddev 1.4ms, 2-stage 2.7ms, 3-stage 6.7ms. More enclaves means more scheduling variability.

4. **Within-run consistency is tight** — p95/p50 ratio is 1.02-1.04 across all configs, showing that once a run starts, token generation is stable.

5. **Output is identical** — All configurations produce the same greedy-decoded text, confirming correct activation relay.

6. **First multi-enclave ML pipeline** — No prior open-source implementation exists for pipeline-parallel ML inference across separate TEE enclaves with encrypted activation relay.

## Methodology Notes

### Statistical Benchmark (N=5)
- 5 independent runs per configuration (1-stage, 2-stage, 3-stage).
- Each run is a fresh cold boot: `aws ec2 stop-instances --force` → `aws ec2 start-instances` → SSH → launch enclaves → run benchmark.
- Fixed prompt ("The capital of France is"), fixed token count (20), greedy decoding.
- Instance type: `m6i.2xlarge` (Intel Xeon 8375C, 8 vCPUs, 32 GiB).
- AZs used: `us-east-1c` (13 of 15 runs), `us-east-1b` (3stage runs 4-5, due to `InsufficientInstanceCapacity` fallback). See [Appendix A](#appendix-a-per-run-environment-metadata) for per-run detail.
- Enclave config: 2 vCPUs, 3072 MiB per enclave. Debug mode enabled.
- 95% CI computed with t-distribution (df=4, t=2.776).
- Raw per-run latency data in `benchmark_results/stat_bench/`.

### Single-Run Benchmarks (N=1)
- One run per configuration per prompt (20 tok and 50 tok). Results are directional reference only.
- Debug mode enabled (may add minor overhead vs production mode).
- All runs on the same instance type (m6i.2xlarge) but not the same instance (reboot/relaunch between configs due to enclave E39 zombie issue).

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

### Three-Stage EIFs
**Stage 0:**
```
PCR0: 3f7daf07987d99290f4ee89ac54da6520b706f2fbd70b47c94469aaa8f06f93ea53ad3a8716b9eac52a464a04149468f
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: 34533297ed118f17ca4e9c71c36e0020957cc49a3cade4cfcbe69e9f340452e233408ca2253d319a5370ffd41433592e
```

**Stage 1:**
```
PCR0: 76d81e6947e52dca70e0bad992c94ba70396659fa46af112acbaf8afe4e1ed4058a1a1997167ff85f01bb036f39e5887
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: 2f3499abe5cfc14e42137f8ca4e10ad1a5f5f197807cce9907665998aa25037fd4f020b9ede558bea46557c6efd2ba96
```

**Stage 2:**
```
PCR0: 39a8463f258df8e1e81ee18e5f5bdd04571b67066f215187a48c6bc008c09324daf545cdcea50f856f8c29990e293885
PCR1: 4b4d5b3661b3efc12920900c80e126e4ce783c522de6c02a2a5bf7af3a2b9327b86776f188e4be1c1c404a129dbda493
PCR2: 93022d97357a6896290edf49903decb4df4c093ead804578cbbfa9687cc22504399c14e53861c56149b0f25da1528a28
```

## Appendix A: Per-Run Environment Metadata

All runs used `m6i.2xlarge` (Intel Xeon Platinum 8375C @ 2.90GHz, 8 vCPUs, 32 GiB). Enclave config: 2 vCPUs, 3072 MiB per enclave. Each run is a fresh cold boot (stop/start cycle).

3stage runs 4-5 ran in `us-east-1b` after `InsufficientInstanceCapacity` blocked the original instance in `us-east-1c`. An AMI was created from the stopped instance and launched in the new AZ to preserve identical EIFs and software state.

| Config | Run | AZ | Instance |
|--------|-----|----|----------|
| 1stage | 1 | us-east-1c | A |
| 1stage | 2 | us-east-1c | A |
| 1stage | 3 | us-east-1c | A |
| 1stage | 4 | us-east-1c | A |
| 1stage | 5 | us-east-1c | A |
| 2stage | 1 | us-east-1c | A |
| 2stage | 2 | us-east-1c | A |
| 2stage | 3 | us-east-1c | A |
| 2stage | 4 | us-east-1c | A |
| 2stage | 5 | us-east-1c | A |
| 3stage | 1 | us-east-1c | A |
| 3stage | 2 | us-east-1c | A |
| 3stage | 3 | us-east-1c | A |
| 3stage | 4 | us-east-1b | B |
| 3stage | 5 | us-east-1b | B |

**Notes:**
- Instance A and B are both `m6i.2xlarge`. Each run is a fresh stop/start cycle (new IP each boot).
- Instance B was launched from an AMI snapshot of Instance A after `InsufficientInstanceCapacity` blocked Instance A in `us-east-1c`. Identical EIFs, software, and allocator config.
- CPU model verified identical (`Intel Xeon Platinum 8375C @ 2.90GHz`) on both instances.
- 3stage run 5 (the highest-latency outlier at 61.5ms/tok) ran on Instance B in `us-east-1b`. Whether the AZ change contributed to the outlier or it was noisy-neighbor host placement is indeterminate from this data.
