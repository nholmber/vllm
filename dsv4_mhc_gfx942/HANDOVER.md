# DeepSeek-V4 MHC on AMD gfx942 (MI300X) — fix handover

Self-contained. Everything you need is in this folder + this vLLM branch
(`fix/dsv4-mhc-gfx942`). No external repos required.

## Problem
On gfx942, DeepSeek-V4 emitted gibberish (GSM8K ~1%). The MHC (hyper-connection)
layer was miscomputing the HC residual stream. Two independent kernel families
were affected:
- **tilelang** fused MHC kernels: wrong at *every* token count (32-lane-warp
  assumption on a 64-lane wavefront). Not used.
- **aiter** MHC kernels: correct at 1 token but wrong for ≥4 tokens because the
  shipped aiter build was missing an accuracy fix (details below).

## Root cause of the aiter miscompute
The deployed aiter build was missing **ROCm/aiter #3417** ("Fix mhc_pre_big_fuse
accuracy"). In `mhc_pre_big_fuse`, the RMS-reduction guard let out-of-range lanes
skip the whole reduction block, so they dropped out of the warp `shfl` reduction
and corrupted `layer_input` for any >1-token input. (`mhc_pre` runs `big_fuse`
after the sqrsum step, so the earlier #3033 sqrsum-race fix alone was not enough.)

The corrected kernel source is `aiter_mhc_kernels.cu` here (= upstream `main`,
includes both #3033 and #3417). The one-logical-line fix appears twice:

```cpp
// BROKEN:
if(warp_id < hc_mult3_reduce_warp_num && lane_id < warp_num_pow2 * num_rows) {
    float sum = s_pre_rms_partial[lane_id];
    if (lane_id % warp_num_pow2 >= warp_num) { sum = 0.0f; }
// FIXED:
if(warp_id < hc_mult3_reduce_warp_num) {
    float sum = 0.0f;
    if(lane_id < warp_num_pow2 * num_rows && lane_id % warp_num_pow2 < warp_num) {
        sum = s_pre_rms_partial[lane_id];
    }
```

## The fix (two files)
1. `aiter_mhc_kernels.cu` → `…/aiter_meta/csrc/kernels/mhc_kernels.cu` (then drop
   the cached `…/aiter/jit/module_mhc.so` so it rebuilds).
2. `mhc.py` → `…/vllm/model_executor/layers/mhc.py` (identical to this branch's
   `vllm/model_executor/layers/mhc.py`). Adds `_mhc_aiter_enabled()` and routes
   `MHCPreOp`/`MHCPostOp`/`MHCFusedPostPreOp` through the aiter ops on ROCm.

## Build the baked image
```bash
# from this folder; BASE = your working DSv4 ROCm image
docker build -t vllm-rocm:dsv4-mhc-aiter-fixed \
  --build-arg BASE=<your-dsv4-rocm-image> .
```
A prebuilt image is already on this host: **`vllm-rocm:dsv4-mhc-aiter-fixed`**.

## Run
```bash
docker run -d --name dsv4 \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined --ipc=host -e HIP_VISIBLE_DEVICES=0,1,2,3 \
  -v <hf-cache>:/huggingface -p 8001:8001 \
  vllm-rocm:dsv4-mhc-aiter-fixed \
  serve deepseek-ai/DeepSeek-V4-Flash --tensor-parallel-size 4 \
  --kv-cache-dtype fp8_e4m3 --max-model-len 32768 --trust-remote-code \
  --tokenizer-mode deepseek_v4 --moe-backend triton_unfused \
  --gpu-memory-utilization 0.85 --distributed-executor-backend mp \
  --max-num-batched-tokens 8192 --host 0.0.0.0 --port 8001
```
First start triggers a one-time ~30s aiter `module_mhc` JIT build (the
from-scratch image only; the prebuilt image already contains it).

## Runtime switches (env)
- `VLLM_DSV4_MHC_AITER=1|0` — force / disable aiter MHC (default: on for ROCm).
- `VLLM_DSV4_MHC_TORCH=1|0` — force torch (correct, slow) / tilelang (broken).
- aiter ops require `hidden_size % 256 == 0` (DSv4-Flash = 7168, OK); otherwise
  it falls back automatically.

## Verify
```bash
# numerics (no model load): expect aiter_rel ~1e-3, all finite
docker run --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD":/h:ro vllm-rocm:dsv4-mhc-aiter-fixed -lc 'python3 /h/verify_numerics.py'

# correctness (server up): expect ~0.95
lm_eval --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Flash,base_url=http://<host>:8001/v1/completions,num_concurrent=32,tokenized_requests=False,trust_remote_code=True,tokenizer=deepseek-ai/DeepSeek-V4-Flash \
  --tasks gsm8k --num_fewshot 5 --batch_size 32

# throughput (per concurrency C in 4 8 16 32)
vllm bench serve --backend vllm --base-url http://<host>:8001 \
  --model deepseek-ai/DeepSeek-V4-Flash --tokenizer deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --dataset-name random --random-input-len 1000 --random-output-len 1000 \
  --num-prompts $((3*C)) --max-concurrency $C --ignore-eos --seed 0
```

## Validation results (DSv4-Flash, TP4, MI300X, compile mode)
| metric | broken (tilelang) | torch fallback | **aiter (this fix)** |
|---|---|---|---|
| GSM8K | 1.29% | 95.38% | **95.6%** |
| throughput @ conc 32 | 833 tok/s (gibberish) | 408 tok/s | **782 tok/s** |
| TPOT @ conc 32 | 37 ms | 77 ms | **40 ms** |

aiter is correct **and** ~2× the torch fallback (recovers the ~46–51% penalty),
matching the fast-but-broken tilelang path.

## Files here
- `aiter_mhc_kernels.cu` — corrected aiter MHC kernel source (#3033 + #3417).
- `mhc.py` — wired vLLM MHC dispatch (copy of this branch's layer file).
- `Dockerfile` — self-contained bake from a DSv4 base image.
- `verify_numerics.py` — aiter-vs-torch micro check.

## Upstreaming / follow-ups
- The vLLM change is ready to upstream (re-enables the aiter blocks the original
  PR left commented). Long term, bump the aiter dependency to a version that
  already includes #3417 instead of patching `mhc_kernels.cu`.
- Optional: fix the tilelang kernel to be wavefront-size aware (lower priority —
  aiter already gives correct + fast).
