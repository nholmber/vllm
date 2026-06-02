# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the FP8 a8w8 blockscale SplitK zero-init fusion (ROCm/aiter#3457).

The fusion folds the GEMM output zero-init (needed before a SplitK atomic-add)
into the upstream activation group-quant producer, so the GEMM can run with
``y_is_zeroed=True``. On the vLLM side this is exposed as the single fused op
``rocm_aiter_fp8_blockscale_group_quant_gemm`` and gated behind
``VLLM_ROCM_USE_AITER_FP8_BLOCKSCALE_FUSED_ZERO_INIT``.

This file is skipped off ROCm / without aiter. The numerical case is further
skipped on aiter builds that predate the required ``gemm_out_zero_init`` /
``y_is_zeroed`` arguments.
"""

import importlib.util

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("This test can only run on ROCm.", allow_module_level=True)

if importlib.util.find_spec("aiter") is None:
    pytest.skip("These tests require AITER to run.", allow_module_level=True)

# Import to ensure the custom ops are registered.
from vllm._aiter_ops import rocm_aiter_ops  # noqa: E402

FP8_DTYPE = current_platform.fp8_dtype()
GROUP_SIZE = 128


def _fused_zero_init_supported() -> bool:
    return rocm_aiter_ops._probe_fused_zero_init_support()


def test_fused_op_registration():
    assert hasattr(torch.ops.vllm, "rocm_aiter_fp8_blockscale_group_quant_gemm")
    assert callable(torch.ops.vllm.rocm_aiter_fp8_blockscale_group_quant_gemm)


def test_enabled_gating_respects_env_flag(monkeypatch):
    """The fusion stays off unless both linear and the opt-in flag are set, and
    only turns on when the aiter build actually exposes the new args."""
    monkeypatch.setattr(rocm_aiter_ops, "_AITER_ENABLED", True)
    monkeypatch.setattr(rocm_aiter_ops, "_LINEAR_ENABLED", True)

    # Off when the opt-in flag is unset, regardless of API support.
    monkeypatch.setattr(rocm_aiter_ops, "_FP8_BLOCKSCALE_FUSED_ZERO_INIT", False)
    monkeypatch.setattr(
        rocm_aiter_ops, "_FP8_BLOCKSCALE_FUSED_ZERO_INIT_SUPPORTED", None
    )
    assert rocm_aiter_ops.is_fp8_blockscale_fused_zero_init_enabled() is False

    # When opted in, the result tracks the (cached) API-support probe.
    monkeypatch.setattr(rocm_aiter_ops, "_FP8_BLOCKSCALE_FUSED_ZERO_INIT", True)
    monkeypatch.setattr(
        rocm_aiter_ops, "_FP8_BLOCKSCALE_FUSED_ZERO_INIT_SUPPORTED", True
    )
    assert rocm_aiter_ops.is_fp8_blockscale_fused_zero_init_enabled() is True

    monkeypatch.setattr(
        rocm_aiter_ops, "_FP8_BLOCKSCALE_FUSED_ZERO_INIT_SUPPORTED", False
    )
    assert rocm_aiter_ops.is_fp8_blockscale_fused_zero_init_enabled() is False


@pytest.mark.skipif(
    not _fused_zero_init_supported(),
    reason="aiter build lacks gemm_out_zero_init / y_is_zeroed (ROCm/aiter#3457).",
)
@pytest.mark.parametrize(
    "M, N, K", [(8, 5120, 2048), (8, 2048, 4096), (16, 1024, 2048)]
)
def test_fused_matches_unfused_reference(M, N, K):
    """The fused quant+GEMM (producer-zeroed Y, y_is_zeroed=True) must match an
    unfused reference where the GEMM does its own zero-init."""
    from aiter import gemm_a8w8_blockscale
    from aiter.ops.quant import per_group_quant_hip

    torch.manual_seed(0)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w_bf16 = torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.1
    w_q = w_bf16.to(FP8_DTYPE)
    w_scale = torch.ones(
        (N + 127) // 128, (K + 127) // 128, device="cuda", dtype=torch.float32
    )

    # Unfused reference: quant without the zero-init side effect, GEMM zeroes Y.
    x_q, x_scale = per_group_quant_hip(
        x,
        quant_dtype=FP8_DTYPE,
        group_size=GROUP_SIZE,
        transpose_scale=True,
    )
    y_ref = gemm_a8w8_blockscale(x_q, w_q, x_scale, w_scale, dtype=torch.bfloat16)

    # Fused path under test.
    y_fused = rocm_aiter_ops.fp8_blockscale_group_quant_gemm(
        x, w_q, w_scale, GROUP_SIZE, output_dtype=torch.bfloat16
    )
    torch.accelerator.synchronize()

    assert y_fused.shape == y_ref.shape
    ymax = y_ref.abs().max().item()
    assert ymax > 0, "degenerate reference output"
    rel = (y_fused - y_ref).abs().max().item() / ymax
    # Only difference vs. reference is bf16 SplitK atomic-add reordering noise.
    assert rel < 2e-2, f"fused result diverged from reference: rel={rel:.2e}"
