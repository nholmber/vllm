# Self-contained bake of the DeepSeek-V4 MHC gfx942 fix.
#
# Applies the two changes needed for correct + fast MHC on gfx942 (MI300X):
#   1. aiter_mhc_kernels.cu -> aiter's MHC kernel source (upstream main; includes
#      ROCm/aiter #3033 sqrsum-race + #3417 mhc_pre_big_fuse reduction fixes).
#   2. mhc.py              -> vLLM dispatch that prefers the aiter MHC ops on gfx942.
# The aiter `module_mhc` JIT object is dropped so it rebuilds from the patched
# source on first server start (~30s, guarded by aiter's FileBaton lock so the
# TP workers don't race). No GPU needed at build time.
#
# Build (context = this folder):
#   docker build -t vllm-rocm:dsv4-mhc-aiter-fixed .
#
# BASE must be a working DeepSeek-V4 vLLM/ROCm image (e.g. the PR-42893 build).
ARG BASE=vllm-rocm:nightly-pr42893-aiter0115src
FROM ${BASE}

ARG AITER_META=/usr/local/lib/python3.12/dist-packages/aiter_meta/csrc/kernels/mhc_kernels.cu
ARG AITER_JIT=/usr/local/lib/python3.12/dist-packages/aiter/jit
ARG VLLM_MHC=/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/mhc.py

COPY aiter_mhc_kernels.cu ${AITER_META}
COPY mhc.py ${VLLM_MHC}
RUN rm -rf ${AITER_JIT}/build/module_mhc ${AITER_JIT}/module_mhc.so
