# GPT-2 Pipeline Latency — Azure DC4ads_v5 (SEV-SNP)

## Platform

| Property | Value |
|----------|-------|
| Provider | Azure |
| VM SKU | Standard_DC4ads_v5 |
| CPU | AMD EPYC 7763 (Milan) [SEV-SNP masked] |
| vCPUs | 4 |
| Memory | 16 GiB |
| Security | ConfidentialVM (SEV-SNP) |
| Kernel | 6.8.0-1044-azure-fde |
| Region | eastus (zone 2) |
| Date | 2026-02-10 |

## Configuration

- **Model:** GPT-2 small (124M params, 12 transformer layers)
- **Hidden dim:** 768
- **Prompt:** "The capital of France is"
- **Max tokens:** 20 (19 generation + 1 prompt)
- **Decoding:** Greedy
- **KV-cache:** Enabled
- **Transport:** TCP with encrypted SecureChannel (MockProvider attestation)
- **Runs:** 5 per configuration

## Results

| Metric | 1-stage (12 layers) | 2-stage (6+6) |
|--------|---------------------|---------------|
| TTFT | 57.1 ± 2.6ms | 59.7 ± 3.0ms |
| Gen avg | 26.6 ± 0.4ms/tok | 27.7 ± 1.6ms/tok |
| Gen p50 | 26.5 ± 0.5ms/tok | 27.7 ± 1.7ms/tok |
| Gen p95 | 27.7 ± 0.7ms/tok | 28.9 ± 2.1ms/tok |
| Gen p99 | 27.7 ± 0.7ms/tok | 28.9 ± 2.1ms/tok |
| Tokens/sec | 37.6 ± 0.6 | 36.2 ± 2.0 |
| **2-stage overhead** | — | **+1.1ms (4.1%)** |

## Cross-Platform Comparison

| Metric | Azure DC4ads_v5 (SEV-SNP, TCP) | AWS m6i.2xlarge (Nitro VSock) |
|--------|-------------------------------|-------------------------------|
| CPU | AMD EPYC 7763 (Milan) | Intel Xeon 8375C (Ice Lake) |
| vCPUs | 4 | 8 (6 enclave + 2 host) |
| Security | SEV-SNP (full VM encryption) | Nitro Enclave (isolated VM) |
| **1-stage gen avg** | **26.6ms/tok** | 41.9ms/tok |
| **1-stage tokens/sec** | **37.6** | 23.9 |
| **2-stage gen avg** | **27.7ms/tok** | 44.1ms/tok |
| **2-stage tokens/sec** | **36.2** | 22.7 |
| **2-stage overhead** | **4.1%** | 5.2% |

### Notes on Cross-Platform Comparison

The Azure numbers are **not directly comparable** to AWS Nitro for several reasons:

1. **Transport difference:** Azure uses TCP loopback (zero network hop); Nitro uses VSock (hypervisor-mediated virtio transport with ~0.17ms RTT per hop)
2. **Isolation difference:** Azure runs stages as processes on the same CVM; Nitro runs each stage in a separate enclave VM with its own kernel
3. **CPU difference:** AMD EPYC 7763 vs Intel Xeon 8375C — different IPC, cache, and memory bandwidth
4. **vCPU allocation:** Azure stages share 4 vCPUs; Nitro allocates 2 dedicated vCPUs per enclave

The Azure result measures **pure pipeline overhead** (SecureChannel encryption + scheduling + relay) without cross-VM transport costs. The Nitro result measures **end-to-end confidential pipeline** including enclave isolation overhead.

The key finding is consistent: **2-stage pipeline overhead is 4-5%** regardless of platform, confirming that the pipeline scheduling and encrypted relay adds minimal latency on top of the per-stage compute.
