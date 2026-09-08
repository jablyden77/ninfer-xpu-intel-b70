# NInfer-XPU for Intel Arc Pro B70

Experimental Intel XPU/SYCL port of [Neroued/ninfer](https://github.com/Neroued/ninfer), focused on running Qwen3.8-27B on Intel Arc Pro B70.

This repository is a public engineering handoff, not a production-ready inference engine. The CUDA/NVIDIA upstream remains dramatically faster today. The goal here is to give Intel/XPU contributors a working baseline they can profile, replace, and improve rather than starting from zero.

## Current result

Test hardware: 1x Intel Arc Pro B70 32 GB, PCIe, single-GPU TP=1.

- Real Qwen3.8-27B GPTQ INT4 G128 model executes all 64 layers.
- BF16 activations, FP32 accumulation/control state.
- 48 Gated DeltaNet layers execute end-to-end.
- 16 full-attention/GQA layers execute end-to-end.
- Real target-only generation reached **9.39 tok/s** after lossless row-major GPTQ repacking.
- Earlier native GPTQ packing reached **7.30 tok/s**.
- A tuned vLLM-XPU implementation on the same class of B70 hardware reaches roughly **77-85 tok/s**, so this port still has a large optimization gap.

There is no claim that NInfer-XPU currently outperforms vLLM on Intel hardware.

## What works

- XPU backend selection in CMake (`NINFER_BACKEND=XPU`).
- Intel oneAPI/SYCL device selection by PCI address.
- USM-backed device arena and tensors.
- BF16 RMSNorm and gated RMSNorm.
- RoPE.
- Packed Q4, W8/Q8, and GPTQ4 G128 T=1 GEMV paths.
- Lossless GPTQ G128 row-major repacking for Intel-friendly decode access.
- Qwen3.8 GDN projection, causal conv state update, gating/control projection, recurrent Gated DeltaNet update, and post-GDN normalization.
- Qwen3.8 full-attention projection, Q/K normalization, RoPE, contiguous single-user KV cache, GQA causal attention, output gating, and MLP tail.
- A simple 64-layer Qwen3.8 runner with greedy argmax generation.
- GPTQ checkpoint exporter that does not include model weights in this repository.

Correctness tests are included for the individual kernels and synthetic complete GDN/full-attention layers.

## Apply this port to upstream NInfer

This public handoff is patch-based so the upstream NVIDIA/CUDA project remains the source of truth.

```bash
git clone https://github.com/Neroued/ninfer.git
cd ninfer
git checkout 3f2e1422e87f41fc421cd487dd6daa0858e301a7
cat /path/to/ninfer-xpu-intel-b70/patches/ninfer-xpu-intel-b70.patch.part-* | git apply -
```

Patch SHA-256 after concatenation:

```text
df5b8844e3d304dcaf038ca428ea24e5c8b56e66a76c78ae08bfbeded5d67f37
```

## Environment used

- Intel oneAPI compiler: 2026.1
- SYCL/OpenCL backend for the working path
- Intel Arc Pro B70 32 GB
- Linux host with Intel `xe` driver
- C++20

Level Zero was tested early in the port but a simple SYCL kernel returned `UR_RESULT_ERROR_UNSUPPORTED_FEATURE`. OpenCL worked and became the bring-up path. Re-establishing a proper Level Zero path is one of the highest-value contribution areas.

## Build

```bash
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-xpu -GNinja \
  -DNINFER_BACKEND=XPU \
  -DCMAKE_C_COMPILER=icx \
  -DCMAKE_CXX_COMPILER=icpx
cmake --build build-xpu -j
```

The XPU tests and runner are built under `src/xpu/`.

## Model export

Model files are **not** included. The development target used a Qwen3.8-27B GPTQ INT4 symmetric G128 checkpoint.

The exporter under `tools/xpu_export_qwen38_gptq.py` reads a local Hugging Face/GPTQ checkpoint and creates an XPU-specific artifact. Large transformer matrices remain the same quantized values; the row-major GPTQ path changes storage order only. Embedding and output-head records are converted to W8G32 for the experimental runner.

Do not commit generated `.ninfer`, `.safetensors`, or other model-weight files.

## Where contributors can make the biggest difference

1. Restore and optimize a Level Zero runtime path.
2. Replace generic SYCL/OpenCL kernels with XeTLA/XeTile/XMX-friendly implementations.
3. Reduce per-layer kernel-launch overhead and add graph capture.
4. Fuse RMSNorm/projection, activation/projection, and residual paths.
5. Optimize GPTQ G128 decode beyond the current simple row-parallel GEMV.
6. Add a real batched/paged KV cache and concurrency path.
7. Implement optimized prefill; the current runner is decode-first and does not provide a meaningful production PP benchmark.
8. Port MTP/speculative decoding after target-only decode is competitive.

A useful first target is simply to beat **10 tok/s**, then 25, 50, and eventually the ~77-85 tok/s vLLM-XPU reference range on one B70.

## Benchmark history

| Stage | Real-model TG |
|---|---:|
| Native GPTQ checkpoint packing | 7.30 tok/s |
| Lossless row-major GPTQ packing | **9.39 tok/s** |
| Tuned vLLM-XPU reference on B70 | ~77-85 tok/s |

Microbenchmarks showed substantially better raw kernel bandwidth than the end-to-end runner, which strongly suggests launch/runtime/fusion overhead remains a major part of the gap. Treat microbenchmarks as diagnostic data, not model-throughput claims.

## Upstream and license

This work is derived from **NInfer** by Neroued and retains the upstream Apache License 2.0. See `LICENSE`.

Upstream project: https://github.com/Neroued/ninfer

This Intel/XPU port is experimental and is not affiliated with or endorsed by Intel, NVIDIA, Alibaba/Qwen, or the upstream NInfer maintainers.

Contributions, profiling results, alternative kernels, and reproducible B70 benchmarks are welcome.
