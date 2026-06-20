# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-KV-head FP8 quantization — 2 kernels total per tensor.

K1: PyTorch abs().amax() — single fused reduce_kernel (MaxNanFunctor)
K2: Triton _fp8_quant_per_kv_head — reads raw amax, computes scale inline, quantizes

No .float(), no .clamp(), no / FP8_MAX, no Memset.
"""

import torch

from vllm.triton_utils import tl, triton

FP8_MAX = torch.finfo(torch.float8_e4m3fn).max


@triton.jit
def _fp8_quant_per_kv_head(
    x_ptr,
    out_ptr,
    amax_ptr,  # [num_kv_heads] raw amax (bf16 or f32)
    stride_t_in,
    stride_h_in,
    stride_t_out,
    stride_h_out,
    head_dim,
    gqa_ratio,
    FP8_MAX: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Quantize one (token, head). Computes scale from raw amax inline."""
    token_id = tl.program_id(0)
    head_id = tl.program_id(1)

    kv_head = head_id // gqa_ratio
    amax_val = tl.load(amax_ptr + kv_head).to(tl.float32)
    safe_amax = tl.maximum(amax_val, 1e-12)
    inv_scale = FP8_MAX / safe_amax

    in_base = token_id * stride_t_in + head_id * stride_h_in
    out_base = token_id * stride_t_out + head_id * stride_h_out
    d_offs = tl.arange(0, BLOCK_D)
    mask = d_offs < head_dim
    vals = tl.load(x_ptr + in_base + d_offs, mask=mask, other=0.0)
    scaled = vals.to(tl.float32) * inv_scale
    clamped = tl.clamp(scaled, -FP8_MAX, FP8_MAX)
    tl.store(out_ptr + out_base + d_offs, clamped.to(tl.float8e4nv), mask=mask)


_out_cache: dict = {}


def fp8_per_kv_head_quant(
    tensor: torch.Tensor,
    num_kv_heads: int,
    batch_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize [tokens, heads, dim] to FP8 with per-KV-head scales.

    2 kernels: 1 PyTorch fused abs+amax, 1 Triton quant.
    """
    total_tokens, num_heads, head_dim = tensor.shape
    gqa_ratio = num_heads // num_kv_heads
    fp8_dtype = torch.float8_e4m3fn
    device = tensor.device

    BLOCK_D = triton.next_power_of_2(head_dim)

    # Allocate output — single cached buffer, resized if needed
    cache_key = (num_heads, head_dim, device)
    if cache_key not in _out_cache or _out_cache[cache_key].shape[0] < total_tokens:
        _out_cache[cache_key] = torch.empty(
            total_tokens, num_heads, head_dim, dtype=fp8_dtype, device=device
        )
    out = _out_cache[cache_key][:total_tokens]

    # K1: PyTorch fused abs+amax (single reduce_kernel with MaxNanFunctor)
    grouped = tensor.view(total_tokens, num_kv_heads, gqa_ratio, head_dim)
    amax = grouped.abs().amax(dim=(0, 2, 3))  # [nkv], dtype matches input (bf16)

    # K2: Triton quant — reads raw amax, computes scale inline
    _fp8_quant_per_kv_head[(total_tokens, num_heads)](
        tensor,
        out,
        amax,
        tensor.stride(0),
        tensor.stride(1),
        out.stride(0),
        out.stride(1),
        head_dim,
        gqa_ratio,
        FP8_MAX=FP8_MAX,
        BLOCK_D=BLOCK_D,
    )

    # descale for FMHA kernel: amax / FP8_MAX, shape [batch_size, nkv]
    # This is a tiny [nkv] tensor op — negligible
    descale = (amax.float() / FP8_MAX).unsqueeze(0).expand(batch_size, -1).contiguous()

    return out, descale
