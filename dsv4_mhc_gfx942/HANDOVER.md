# DeepSeek-V4 MHC fix on AMD gfx942 (MI300X) — handover

**Problem:** DSv4 produced gibberish on gfx942 (GSM8K ~1%). The MHC layer miscomputed the HC
residual stream.

**Root cause:** The shipped aiter build was missing `ROCm/aiter #3417` (`mhc_pre_big_fuse`
accuracy). Its RMS-reduction guard let out-of-range lanes skip the reduction block, dropping them
from the warp shfl reduction → corrupted `layer_input` for any >1-token input. (The earlier `#3033`
sqrsum-race fix alone wasn't enough — `mhc_pre` runs big_fuse after the sqrsum step.) The separate
tilelang MHC kernels are *also* broken (64- vs 32-lane wavefront assumption) and are not used.

**Fix:** Use aiter MHC kernels built with **both #3033 + #3417**, and wire vLLM to prefer them on
ROCm. Two files:
- corrected `mhc_kernels.cu` → aiter source (+ drop cached `module_mhc.so` so it rebuilds)
- wired `mhc.py` → `_mhc_aiter_enabled()` gate; routes MHCPre/Post/FusedPostPre through aiter
  (needs `hidden_size % 256 == 0`, DSv4-Flash=7168 OK)

## aiter vs tilelang — correctness
`mhc_pre` `layer_input` max rel-err vs torch (ground truth):
```
T      aiter(#3033+#3417)   tilelang
1      6.5e-08  OK           ~1.0    BROKEN
4      8.0e-05  OK           0.94    BROKEN
16     2.9e-03  OK           ~1-2.4  BROKEN
64     1.1e-03  OK           ~1-1.9  BROKEN
256    1.9e-03  OK           1.25    BROKEN
1024   1.4e-03  OK           ~1-1.9  BROKEN
```
aiter = bf16-accurate everywhere → GSM8K 95.6%. tilelang = order-1 wrong at every T (incl T=1) →
GSM8K 1.29% (gibberish). tilelang's bug is structural: launches `threads=96`, guards reductions with
`if tid < 32:` (32-lane NVIDIA-warp assumption); on gfx942's 64-lane wavefront that's half a wave and
the in-`if` `sync_threads()` is unsafe.

## Throughput (1k/1k, compile mode, tok/s)
tilelang numbers are produced while emitting garbage; torch-fix is the correct-but-slow baseline:
```
conc   tilelang(broken)   torch-fix   aiter     aiter vs torch-fix
4      110.28             59.65       118.94    +99%
8      224.62             118.64      229.08    +93%
16     428.66             224.41      434.99    +94%
32     833.83             408.10      781.75    +91%
```
GSM8K: broken 1.29% | torch-fix 95.38% | **aiter 95.6%**.
TPOT @ conc32 (ms): tilelang 37 | torch-fix 77 | **aiter 40**.

aiter is correct **and** ~2× the torch fallback (recovers the ~46–51% penalty), matching the
fast-but-broken tilelang up to conc 16 and only ~6% behind at conc 32. The conc-32 gap is because
tilelang has one fused `mhc_fused_post_pre` kernel while aiter composes `mhc_post`→`mhc_pre`
(one extra launch + HBM round-trip). **aiter supersedes the torch fallback; tilelang stays unused.**

## Where everything is (self-contained in the vLLM fork — no other repo needed)
- `nholmber/vllm` @ `fix/dsv4-mhc-gfx942`
- code: `vllm/model_executor/layers/mhc.py`
- bundle: `dsv4_mhc_gfx942/` → `HANDOVER.md`, `aiter_mhc_kernels.cu`, `mhc.py`, `Dockerfile`, `verify_numerics.py`
- commits: `7b118688d` (torch fallback) → `b795ba16d` (enable aiter) → `1102d07b1` (handover + bake)

## Build / run
Prebuilt image on the host: **`vllm-rocm:dsv4-mhc-aiter-fixed`** (aiter prebuilt, ready to run).
Or rebuild from the bundle against your own DSv4 base image:
```
docker build -t vllm-rocm:dsv4-mhc-aiter-fixed --build-arg BASE=<your-dsv4-image> dsv4_mhc_gfx942/
```
Serve (TP4, port 8001):
```
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

## Runtime switches
- `VLLM_DSV4_MHC_AITER=1|0` — force / disable aiter MHC (default: on for ROCm).
- `VLLM_DSV4_MHC_TORCH=1|0` — force torch (correct, slow) / tilelang (broken).

## Verify
```
# numerics (no model load): expect aiter_rel ~1e-3, all finite
docker run --rm --entrypoint bash \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --security-opt seccomp=unconfined -e HIP_VISIBLE_DEVICES=0 \
  -v "$PWD/dsv4_mhc_gfx942":/h:ro vllm-rocm:dsv4-mhc-aiter-fixed -lc 'python3 /h/verify_numerics.py'

# GSM8K (expect ~0.95)
lm_eval --model local-completions \
  --model_args model=deepseek-ai/DeepSeek-V4-Flash,base_url=http://<host>:8001/v1/completions,num_concurrent=32,tokenized_requests=False,trust_remote_code=True,tokenizer=deepseek-ai/DeepSeek-V4-Flash \
  --tasks gsm8k --num_fewshot 5 --batch_size 32

# throughput (per concurrency C in 4 8 16 32)
vllm bench serve --backend vllm --base-url http://<host>:8001 \
  --model deepseek-ai/DeepSeek-V4-Flash --tokenizer deepseek-ai/DeepSeek-V4-Flash --trust-remote-code \
  --dataset-name random --random-input-len 1000 --random-output-len 1000 \
  --num-prompts $((3*C)) --max-concurrency $C --ignore-eos --seed 0
```

## Open item
Bump the aiter dependency to a release that already includes #3417 (so no `mhc_kernels.cu` patch is
needed) — then the `mhc.py` change is upstreamable as-is. Optional/low priority: make the tilelang
kernel wavefront-size aware (higher effort, no advantage over aiter).
