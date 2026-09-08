# Qwen3.8-27B FP8 on 2× Intel Arc Pro B70 — R187 MTP4 recipe

This is the production recipe I am using for **Qwen3.8-27B FP8** on **two Intel Arc Pro B70 32 GB GPUs** with vLLM XPU, tensor parallelism across both cards, and Qwen's native MTP speculative decoding.

The goal is not a synthetic peak. The selected profile is the one that gave the best balance of **single-user generation speed, multi-user throughput, 64K context, prompt processing speed, and output identity** on the actual deployment.

## Result

Validated host:

- 2× Intel Arc Pro B70, 32 GB each
- PCIe 5.0 ×8 per GPU
- AMD Ryzen 9 9900X
- 64 GB system RAM
- Linux + Intel Xe/Level Zero stack
- Docker
- vLLM XPU
- Official `Qwen/Qwen3.8-27B-FP8` checkpoint
- TP=2
- R187 whole-graph deterministic profile
- MTP depth 4
- 64K maximum context

### Qualified single-user result

The R187 qualification campaign measured:

| MTP depth | Qualified TG |
|---:|---:|
| MTP0 | 33.097 tok/s |
| MTP1 | 54.935 tok/s |
| MTP2 | 70.142 tok/s |
| MTP3 | 79.183 tok/s |
| **MTP4** | **82.396 tok/s** |
| MTP5 | 86.182 tok/s |
| MTP6 | ~87.1 tok/s in later strict pairs |

The MTP4 strict pair was **82.447 / 82.345 tok/s**, with all 12 complete token arrays matching the same-configuration MTP0 oracle.

### Fresh production-style 1/2/4-user check

A later three-pass live test on the 64K MTP4 service produced:

| Concurrent users | Aggregate throughput | Median per-user TG | Median TTFT | Output identity |
|---:|---:|---:|---:|---:|
| **1** | **75.12 tok/s** | **79.54 tok/s** | **93 ms** | 3/3 exact |
| **2** | **119.34 tok/s** | **61.41 tok/s** | **98 ms** | 6/6 exact |
| **4** | **206.41 tok/s** | **65.22 tok/s** | **147 ms** | 12/12 exact |

The first 2-user pass hit a cold/new-batch-shape path at 79.47 tok/s aggregate; the next two passes were 119.56 and 119.34 tok/s. The table uses the stable median.

### Prompt processing / prefill

Effective service-level prefill on the R187 line was approximately **3.0K–3.5K prompt tokens/s** through 32K active context. This is calculated as `active prompt tokens / HTTP TTFT`; it includes scheduling and first-token work and is not a kernel-only PP number.

| Active context | TTFT | Effective PP |
|---:|---:|---:|
| 2K | 0.60 s | ~3,413 tok/s |
| 4K | 1.17 s | ~3,501 tok/s |
| 8K | 2.37 s | ~3,457 tok/s |
| 16K | 4.94 s | ~3,317 tok/s |
| 24K | 7.74 s | ~3,175 tok/s |
| 32K | 10.73 s | ~3,054 tok/s |

## Why MTP4 is the production choice

MTP5 and MTP6 are faster on the fixed single-user qualification suite, but they did **not** win on the practical 1/2/4-user workload used for this deployment.

| Profile | 1-user TG | 2-user aggregate | 4-user aggregate |
|---|---:|---:|---:|
| **MTP4** | **79.54 tok/s** | **119.34 tok/s** | **206.41 tok/s** |
| MTP5 | 74.70 tok/s | 114.03 tok/s | 202.03 tok/s |
| MTP6 | 77.01 tok/s | 113.52 tok/s | 196.02 tok/s |

All three profiles passed complete token-ID identity in that test. MTP4 simply had the best real deployment balance.

## Credits and source of the optimization work

This recipe uses the R187 work from Steve Seguin's **b70-optimization-lab**. The upstream lab did the heavy lifting on the Intel B70/XPU integration, block-W8A16 path, deterministic GDN handling, oneCCL ordering, row-invariant oneDNN strategy, MTP draft-head optimization, whole-graph compile profile, and qualification harness.

Upstream:

- https://github.com/steveseguin/b70-optimization-lab
- Snapshot used for this deployment documentation: `5e04c9e00cf9e82ec898234a523bb9053d76fd08`
- Upstream package: `packages/qwen38-27b-fp8-tp2-b70/`
- Upstream reproduction guide: `repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/README.md`

The purpose of this document is to record the exact configuration that was deployed and re-tested on my dual-B70 host, not to re-claim the upstream optimization work.

## 1. Clone the upstream lab at the tested snapshot

```bash
git clone https://github.com/steveseguin/b70-optimization-lab.git
cd b70-optimization-lab
git checkout 5e04c9e00cf9e82ec898234a523bb9053d76fd08
```

## 2. Download the exact Qwen FP8 model revision

```bash
huggingface-cli download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a \
  --local-dir /path/to/Qwen3.8-27B-FP8
```

Do not copy model weights into this repository.

## 3. Pull the exact prebuilt R187 runtime image

The upstream lab publishes the exact image used by the qualified R187 line on GHCR.

```bash
docker pull \
  ghcr.io/steveseguin/vllm-openai-xpu-qwen38-fp8@sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0

docker tag \
  ghcr.io/steveseguin/vllm-openai-xpu-qwen38-fp8@sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0 \
  neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-gdn-split-mixed-r156
```

Image digest / expected image ID:

```text
sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0
```

If you prefer to build from source, follow the upstream reproduction guide instead of the GHCR shortcut. The source route is considerably more involved because it rebuilds the pinned XPU kernel / oneDNN / vLLM overlay chain.

## 4. Preflight the host and model

Use the upstream contract checker before serving:

```bash
IMAGE=neural-download/vllm-openai-xpu:qwen38-fp8-mtp1-gdn-split-mixed-r156 \
IMAGE_CONTRACT_PROFILE=mtp1-serial-fa-split-gdn \
MODEL_DIR=/path/to/Qwen3.8-27B-FP8 \
  repro/qwen38-27b-fp8-vllm-tp2-asrock-b70/preflight.sh
```

The preflight checks the model identities, backing-store reads, Docker access, render devices, memory boundary, and installed container contract.

## 5. Launch the production MTP4 / 64K profile

Use a dedicated vLLM cache directory. The deployment used a 64K service with four active slots and an 8,192-token batching ceiling.

```bash
export MODEL_DIR=/path/to/Qwen3.8-27B-FP8
export VLLM_CACHE_DIR=/path/to/vllm-cache/r187-mtp4-64k
export EXPECTED_IMAGE_ID=sha256:173660ec18c6e98a14b9a4f573922abe9d3414999056f07ab5c3c14b55d6ceb0

export PORT=8027
export MAX_MODEL_LEN=65536
export MAX_NUM_SEQS=4
export MAX_NUM_BATCHED_TOKENS=8192
export GPU_MEMORY_UTILIZATION=0.95

experiments/qwen38-27b-b70/scripts/\
run-20260903-qwen38-fp8-mtp4-whole-graph-r187-server.sh
```

The wrapper selects:

```text
served model: qwen38-fp8-block-w8a16-mtp4-whole-graph-r187
tensor parallel: 2
speculative method: qwen3_next_mtp
num_speculative_tokens: 4
model dtype: FP16 activations / official block-FP8 weights
KV dtype: auto
XPU Graph: disabled
whole-graph torch.compile: splitting_ops=[]
draft vocabulary projection: INT4 G128
target verifier vocabulary projection: FP16
block size: 64
prefix caching: disabled
```

The important environment gates in the qualified stack include the block-W8A16 path, mixed-step GDN split, deterministic Inductor settings, explicit oneCCL ordering, and draft-only INT4 LM head. Use the upstream wrapper rather than manually reconstructing those flags unless you are deliberately doing kernel research.

## 6. Verify the API

```bash
curl -fsS http://127.0.0.1:8027/health
curl -fsS http://127.0.0.1:8027/v1/models
```

Expected model entry:

```text
qwen38-fp8-block-w8a16-mtp4-whole-graph-r187
max_model_len: 65536
```

## 7. Re-run the 1/2/4-user identity benchmark

The following uses the upstream concurrency harness, disables cache reuse, requests 128 generated tokens per request, and requires each concurrent output to match its sequential oracle exactly.

```bash
mkdir -p /tmp/r187-mtp4-final

python3 scripts/bench-openai-concurrency-oracle.py \
  --base-url http://127.0.0.1:8027 \
  --model qwen38-fp8-block-w8a16-mtp4-whole-graph-r187 \
  --api-mode completions \
  --suite experiments/qwen38-27b-b70/data/2026-08-25-qwen38-q4km-tp2-http-smallctx-suite.json \
  --concurrency 1,2,4 \
  --repeats 3 \
  --max-tokens 128 \
  --seed 42 \
  --timeout 600 \
  --request-extra-json '{"ignore_eos":true,"temperature":0}' \
  --return-token-ids \
  --require-output-identity \
  --out /tmp/r187-mtp4-final/ladder.json
```

Do not report a speed number if the identity gate fails. This stack was built specifically to avoid presenting a faster but numerically unstable speculative result as production-ready.

## Notes

- **MTP4 is intentional.** Deeper speculation is not automatically faster once multiple users are active.
- **64K is the configured maximum context**, not the context length used for the short-prompt throughput ladder.
- The **3K–3.5K PP figure is effective HTTP prefill**, not a raw GEMM/kernel number.
- The strict 82.396 tok/s headline and the later ~79.5 tok/s live 1-user measurement use different benchmark shapes; both are valid within their stated methodology.
- The current production profile keeps the **target verifier FP16** and only quantizes the MTP draft vocabulary projection to INT4.
- Prefix caching was disabled for qualification so cached prompts could not inflate PP or latency results.
- Intel XPU software changes quickly. Pin the model revision, container digest, and upstream lab snapshot when reproducing results.

## Bottom line

For this dual-B70 host, the production configuration is:

```text
Qwen3.8-27B official FP8
2 × Intel Arc Pro B70 32 GB
TP2
R187 whole-graph deterministic XPU build
MTP4
64K context
~3.0K–3.5K effective PP
~80–82 tok/s single-user TG depending on benchmark shape
~119 tok/s aggregate at 2 users
~206 tok/s aggregate at 4 users
output-identity qualified on the tested workloads
```

That is the recipe I would reproduce before experimenting with DFlash, deeper MTP, or a different 27B quantization path.
