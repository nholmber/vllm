# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.triton_utils import tl, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.utils.platform_utils import num_compute_units

from vllm import _custom_ops as ops
from vllm._aiter_ops import rocm_aiter_ops
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    get_fp8_min_max,
    group_broadcast,
    prep_scale_for_group_broadcast,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import (
    DeepGemmQuantScaleFMT,
    is_deep_gemm_e8m0_used,
    is_deep_gemm_supported,
)

_FP8_DTYPE = current_platform.fp8_dtype()
_FP8_MIN, _FP8_MAX = get_fp8_min_max()
_FP8_MIN_SCALING_FACTOR = 1.0 / (_FP8_MAX * 512.0)


def calc_rows_per_block(M: int, device: torch.device) -> int:
    sm_count = num_compute_units(device.index)
    rows_per_block = next_power_of_2(cdiv(M, 2 * sm_count))
    rows_per_block = min(rows_per_block, 4)
    return rows_per_block

@triton.jit
def silu_mul_input_quant_fp8_kernel(
    X,  # pointer to the input, shape (M, N) where N = 2*D
    Y_quant,  # pointer to the quantized output, shape (M, D)
    S,  # pointer to the scales, shape (M, NG)
    stride_x_row,
    stride_x_col,
    stride_y_row,
    stride_s_row,
    M,  # number of rows
    D,  # half of N (output dim)
    G: tl.constexpr,  # group size
    BLOCK_G: tl.constexpr,  # next_power_of_2(G)
    ROWS_PER_BLOCK: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN_SCALING_FACTOR: tl.constexpr,
):
    # program_id(0) -> row tile, program_id(1) -> group index within D
    rows = tl.program_id(0) * ROWS_PER_BLOCK + tl.arange(0, ROWS_PER_BLOCK)
    my_group = tl.program_id(1)
    row_mask = rows < M
    cols = tl.arange(0, BLOCK_G)

    # Column offsets within D for this group
    col_off = my_group * G + cols
    mask = row_mask[:, None] & ((col_off < D) & (cols < G))[None, :]

    # Load gate (first half) and up (second half)
    gate_ptr = X + rows[:, None] * stride_x_row + col_off[None, :] * stride_x_col
    up_ptr = X + rows[:, None] * stride_x_row + (col_off[None, :] + D) * stride_x_col

    gate = tl.load(gate_ptr, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(up_ptr, mask=mask, other=0.0).to(tl.float32)

    # SiLU(gate) * up
    y = (gate * tl.sigmoid(gate)) * up

    # Per-group FP8 quantization
    group_absmax = tl.max(tl.where(mask, tl.abs(y), 0.0), axis=1)
    scale_raw = group_absmax / FP8_MAX
    if USE_UE8M0:
        scale_raw = tl.exp2(tl.ceil(tl.log2(scale_raw)))
    scale = tl.maximum(scale_raw, FP8_MIN_SCALING_FACTOR)

    # Store scales (one per row per group)
    tl.store(S + rows * stride_s_row + my_group, scale, mask=row_mask)

    # Quantize and store
    y_scaled = y / scale[:, None]
    y_quant = tl.maximum(tl.minimum(y_scaled, FP8_MAX), FP8_MIN)

    y_ptr = Y_quant + rows[:, None] * stride_y_row + col_off[None, :]
    tl.store(y_ptr, y_quant.to(Y_quant.dtype.element_ty), mask=mask)


@triton.heuristics(
    {
        "HAS_BIAS": lambda args: args["B"] is not None,
        "HAS_Z": lambda args: args["Z"] is not None,
    }
)
@triton.jit
def rms_norm_input_quant_fp8_kernel(
    X,  # pointer to the input
    W,  # pointer to the weights
    B,  # pointer to the biases
    Z,  # pointer to the other branch
    Y_quant, # pointer to the quantized output
    Scales, # pointer to the scales
    stride_x_row,  # how much to increase the pointer when moving by 1 row
    stride_z_row,
    stride_y_row,
    M,  # number of rows in X
    N: tl.constexpr,  # number of columns in X
    eps,  # epsilon to avoid division by zero
    BLOCK_N: tl.constexpr,
    ROWS_PER_BLOCK: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_Z: tl.constexpr,
    NORM_BEFORE_GATE: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    USE_UE8M0: tl.constexpr,
    FP8_MIN_SCALING_FACTOR: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    # Map the program id to the starting row of X and Y it should compute.
    row_start = tl.program_id(0) * ROWS_PER_BLOCK
    group = tl.program_id(1)

    # Create 2D tile: [ROWS_PER_BLOCK, BLOCK_N]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)
    cols = tl.arange(0, BLOCK_N)

    # Compute offsets for 2D tile
    row_offsets = rows[:, None] * stride_x_row
    col_offsets = cols[None, :] + group * BLOCK_N

    # Base pointers
    X_base = X + row_offsets + col_offsets
    Y_base = Y_quant + rows[:, None] * stride_y_row + col_offsets
    S_base = Scales + rows

    # Create mask for valid rows and columns
    row_mask = rows[:, None] < M
    col_mask = cols[None, :] < N
    mask = row_mask & col_mask

    # Load input data with 2D tile
    x = tl.load(X_base, mask=mask, other=0.0).to(tl.float32)

    if HAS_Z and not NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            x *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            x *= tl.sigmoid(z)

    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N  # Shape: [ROWS_PER_BLOCK]
    rstd = tl.rsqrt(var + eps)  # Shape: [ROWS_PER_BLOCK]

    # Load weights and biases (broadcast across rows)
    w_offsets = cols + group * BLOCK_N
    w_mask = w_offsets < N
    w = tl.load(W + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    if HAS_BIAS:
        b = tl.load(B + w_offsets, mask=w_mask, other=0.0).to(tl.float32)

    # Normalize and apply linear transformation
    x_hat = x * rstd[:, None]

    y = x_hat * w[None, :] + b[None, :] if HAS_BIAS else x_hat * w[None, :]

    if HAS_Z and NORM_BEFORE_GATE:
        Z_base = Z + rows[:, None] * stride_z_row + col_offsets
        z = tl.load(Z_base, mask=mask, other=0.0).to(tl.float32)
        if ACTIVATION == "swish" or ACTIVATION == "silu":
            y *= z * tl.sigmoid(z)
        elif ACTIVATION == "sigmoid":
            y *= tl.sigmoid(z)

    ## Now we got y, we next quantize y

    # Compute per-row absmax (only considering valid elements)
    abs_y = tl.where(mask, tl.abs(y), 0.0)
    absmax = tl.max(abs_y, axis=1)  # Shape: [ROWS_PER_BLOCK]
    
    # Compute scales
    scales_raw = absmax / FP8_MAX
    # TODO: Add USE_UE8M0 as a constexpr parameter if needed:
    if USE_UE8M0:
        scales_raw = tl.exp2(tl.ceil(tl.log2(scales_raw)))
    scales = tl.maximum(scales_raw, FP8_MIN_SCALING_FACTOR)  # Shape: [ROWS_PER_BLOCK]
    
    # Quantize: divide by scale (broadcast to match y shape) and clamp
    y_scaled = y / scales[:, None]  # Broadcast scales from [ROWS_PER_BLOCK] to [ROWS_PER_BLOCK, BLOCK_N]
    y_quant = tl.maximum(tl.minimum(y_scaled, FP8_MAX), FP8_MIN)
    
    # Store quantized output
    tl.store(Y_base, y_quant.to(Y_quant.dtype.element_ty), mask=mask)
    
    # Store scales (one per row)
    scales_row_mask = rows < M
    tl.store(S_base, scales, mask=scales_row_mask)


# --8<-- [start:quant_fp8]
@CustomOp.register("quant_fp8")
class QuantFP8(CustomOp):
    """
    Quantize input tensor to FP8 (per-tensor, per-token, per-channel, or per-group).
    This CustomOp supports both static and dynamic quantization.
    """

    # --8<-- [end:quant_fp8]

    def __init__(
        self,
        static: bool,
        group_shape: GroupShape,
        num_token_padding: int | None = None,
        column_major_scales: bool = False,
        tma_aligned_scales: bool = False,
        use_ue8m0: bool | None = None,  # for Torch compile
        compile_native: bool = True,
    ):
        """
        :param static: static or dynamic quantization
        :param group_shape: quantization group shape (PER_TOKEN, PER_TENSOR,
            PER_CHANNEL, or arbitrary block size)
        :param num_token_padding: Pad the token dimension of output to this
            size
        :param tma_aligned_scales: For group quantization, output scales in
            TMA-aligned layout
        :param column_major_scales: For group quantization, output scales in
            column major format
        :param compile_native: Manually compile forward_native if compile mode > None
        """
        super().__init__(compile_native=compile_native)
        self.static = static
        self.group_shape = group_shape
        self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
        self.num_token_padding = num_token_padding
        self.column_major_scales = column_major_scales
        self.tma_aligned_scales = tma_aligned_scales
        self.use_ue8m0 = is_deep_gemm_e8m0_used() if use_ue8m0 is None else use_ue8m0
        self.use_deep_gemm_supported = is_deep_gemm_supported()

        self.use_aiter = rocm_aiter_ops.is_linear_fp8_enabled()

        self.is_group_quant = group_shape.is_per_group()
        if self.is_group_quant:
            self.group_size = group_shape.col
        else:
            self.use_per_token_if_dynamic = group_shape == GroupShape.PER_TOKEN
            if not static:
                assert group_shape in (GroupShape.PER_TOKEN, GroupShape.PER_TENSOR), (
                    "Only per-token or per-tensor scales are supported for dynamic "
                    "non-group quantization."
                )

    def forward_cuda(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
        rms_norm_parameters: dict | None = None,
        silu_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        from vllm.model_executor.layers.quantization.utils import fp8_utils

        if (
            self.is_group_quant
            and self.use_ue8m0
            and self.use_deep_gemm_supported
            and (DeepGemmQuantScaleFMT.from_oracle() == DeepGemmQuantScaleFMT.UE8M0)
        ):
            return fp8_utils.per_token_group_quant_fp8_packed_for_deepgemm(
                x,
                group_size=self.group_size,
                use_ue8m0=True,
            )

        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"

            return fp8_utils.per_token_group_quant_fp8(
                x,
                group_size=self.group_size,
                column_major_scales=self.column_major_scales,
                tma_aligned_scales=self.tma_aligned_scales,
                dtype=_FP8_DTYPE,
                use_ue8m0=self.use_ue8m0,
            )

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        return ops.scaled_fp8_quant(
            x,
            scale,
            num_token_padding=self.num_token_padding,
            scale_ub=scale_ub,
            use_per_token_if_dynamic=self.use_per_token_if_dynamic,
            group_shape=(self.group_shape.row, self.group_shape.col)
            if self.static
            else None,
        )

    def forward_hip(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
        rms_norm_parameters: dict | None = None,
        silu_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.is_group_quant and use_triton:
            assert scale is None, "Dynamic group quantization does not use scale"

            return torch.ops.vllm.triton_per_token_group_quant_fp8(x, self.group_size)

        use_aiter_quant = self.use_aiter and scale_ub is None and x.is_contiguous()
        use_aiter_per_tensor_quant = (
            use_aiter_quant and self.group_shape.is_per_tensor()
        )
        use_aiter_per_token_quant = use_aiter_quant and self.group_shape.is_per_token()

        use_aiter_per_group_quant = use_aiter_quant and self.group_shape.is_per_group()

        if use_aiter_per_group_quant:
            return rocm_aiter_ops.group_fp8_quant(x, self.group_size)
        if use_aiter_per_tensor_quant:
            return rocm_aiter_ops.per_tensor_quant(x, _FP8_DTYPE, scale)
        if use_aiter_per_token_quant:
            return rocm_aiter_ops.per_token_quant(x, _FP8_DTYPE, scale)

        # Fallback to native implementation for group quantization.
        if self.is_group_quant:
            assert scale is None, "Dynamic group quantization does not use scale"
            return self._quantize_group_native(x)

        # Fallback to CUDA implementation
        return self.forward_cuda(x, scale, scale_ub)

    def forward_xpu(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
        rms_norm_parameters: dict | None = None,
        silu_mul: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # XPU can use same code path as CUDA.
        return self.forward_cuda(x, scale, scale_ub, use_triton)

    def forward_native(
        self,
        x: torch.Tensor,
        scale: torch.Tensor | None = None,
        scale_ub: torch.Tensor | None = None,
        use_triton: bool = False,
        rms_norm_parameters: dict | None = None,
        silu_mul: bool = False,
    ):
        if self.is_group_quant and not self.static:
            assert scale is None, "Dynamic group quantization does not use scale"
            if rms_norm_parameters is not None:
                return self._rmsnorm_quantize_group_native(x, rms_norm_parameters)
            elif silu_mul:
                return self._silu_mul_quantize_group_native(x)
            else:
                return self._quantize_group_native(x)

        assert (scale is not None) == self.static
        assert scale_ub is None or (
            not self.static
            and self.group_shape == GroupShape.PER_TOKEN
            and scale_ub.numel() == 1
        )

        if scale is None:
            if self.group_shape == GroupShape.PER_TOKEN:
                x_max, _ = x.abs().max(dim=-1)
                x_max = x_max.unsqueeze(-1).to(torch.float32)
                if scale_ub is not None:
                    x_max = x_max.clamp(max=scale_ub)
            else:
                x_max = x.abs().max().unsqueeze(-1).to(torch.float32)

            scale = (x_max / _FP8_MAX).clamp(min=_FP8_MIN_SCALING_FACTOR)
        else:
            scale = prep_scale_for_group_broadcast(scale, x, self.group_shape)

        # Even for dynamic per-token scales,
        # reciprocal performs slightly better than division
        out = (
            x.to(torch.float32)
            * group_broadcast(scale.to(torch.float32), x.shape[-2:]).reciprocal()
        )
        out = out.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        # This currently generates an extra Triton kernel in compilation.
        # Fortunately, we don't use padding if compiling.
        # TODO(luka): benchmark torch._scaled_mm to hopefully remove padding
        #  in general.
        if self.num_token_padding is not None:
            padding = max(self.num_token_padding - out.size(0), 0)
            out = F.pad(out, (0, 0, 0, padding), "constant", 0.0)

        return out, scale

    def _quantize_group_native(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_shape = x.shape
        hidden_dim = x.shape[-1]
        num_groups = (hidden_dim + self.group_size - 1) // self.group_size
        padded_dim = num_groups * self.group_size

        if padded_dim != hidden_dim:
            padding = padded_dim - hidden_dim
            x = F.pad(x, (0, padding), mode="constant", value=0.0)

        x_grouped = x.view(-1, num_groups, self.group_size)
        absmax = x_grouped.abs().max(dim=-1, keepdim=True)[0].float()
        scales_raw = absmax / _FP8_MAX
        if self.use_ue8m0:
            scales_raw = torch.exp2(torch.ceil(torch.log2(scales_raw)))
        scales = (scales_raw).clamp(min=_FP8_MIN_SCALING_FACTOR)

        x_scaled = x_grouped / scales
        x_quant = x_scaled.clamp(_FP8_MIN, _FP8_MAX).to(_FP8_DTYPE)

        x_quant = x_quant.view(-1, padded_dim)
        if padded_dim != hidden_dim:
            x_quant = x_quant[..., :hidden_dim]
        x_quant = x_quant.view(orig_shape)

        scales = scales.squeeze(-1)
        scales = scales.reshape(orig_shape[:-1] + (num_groups,))

        if self.column_major_scales:
            scales = scales.transpose(-2, -1).contiguous().transpose(-1, -2)

        return x_quant, scales

    def _rmsnorm_quantize_group_native(
        self, 
        x: torch.Tensor, # (n, h, d)
        rms_norm_parameters: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert not self.column_major_scales, "column major for scales is not supported in this kernel"
        assert rms_norm_parameters is not None, "rms norm parameters must not be None"
        z = rms_norm_parameters["z"]
        weight = rms_norm_parameters["weight"]
        bias = rms_norm_parameters["bias"]
        norm_group_size = rms_norm_parameters["group_size"]
        eps = rms_norm_parameters["eps"]
        norm_before_gate = rms_norm_parameters["norm_before_gate"]
        activation = rms_norm_parameters["activation"]

        assert norm_group_size is None, "group_size in rms norm should be None"
        assert norm_before_gate, "norm_before_gate should be True"

        ## following asserts are based on the fact that x and z are created by torch.zeros()
        assert x.is_contiguous(), "x should be contiguous"
        assert z.is_contiguous(), "z should be contiguous"

        num_tokens, num_heads, head_dim = z.shape
        x = x.view(-1, head_dim)
        z = z.view(-1, head_dim)
        assert x.size() == z.size(), "x and z should have the same shape"
        assert self.group_size == head_dim, f"the kernel is only supported for group_size == head_dim, got group_size {self.group_size} and head_dim {head_dim}"

        weight = weight.contiguous()
        if bias is not None:
            bias = bias.contiguous()

        ## now we do the job
        M = x.shape[0]
        group_size = head_dim
        ngroups = 1
        # Less than 64KB per feature: enqueue fused kernel
        MAX_FUSED_SIZE = 65536 // x.element_size()
        BLOCK_N = min(MAX_FUSED_SIZE, triton.next_power_of_2(group_size))
        if group_size > BLOCK_N:
            raise RuntimeError("This layer norm doesn't support feature dim >= 64KB.")
        # heuristics for number of warps
        num_warps = min(max(BLOCK_N // 256, 1), 8)
        # Calculate rows per block based on SM count
        rows_per_block = calc_rows_per_block(M, x.device)

        x_quant = torch.empty_like(x, dtype=_FP8_DTYPE)
        scales = torch.empty(M, dtype=torch.float32, device=x.device)

        grid = (cdiv(M, rows_per_block), ngroups)
        rms_norm_input_quant_fp8_kernel[grid](
            x,
            weight,
            bias,
            z,
            x_quant,
            scales,
            x.stride(0),
            z.stride(0) if z is not None else 0,
            x_quant.stride(0),
            M,
            group_size,
            eps,
            BLOCK_N=BLOCK_N,
            ROWS_PER_BLOCK=rows_per_block,
            NORM_BEFORE_GATE=norm_before_gate,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            USE_UE8M0=self.use_ue8m0,
            FP8_MIN_SCALING_FACTOR=_FP8_MIN_SCALING_FACTOR,
            num_warps=num_warps,
            ACTIVATION=activation,
        )

        x_quant = x_quant.view(num_tokens, -1)
        scales = scales.view(num_tokens, -1)

        return x_quant, scales

    def _silu_mul_quantize_group_native(
        self, 
        x: torch.Tensor,  # (M, N) where N = 2*D
    ) -> tuple[torch.Tensor, torch.Tensor]:

        assert not self.column_major_scales, "column major for scales is not supported in this kernel"

        x_shape = x.shape
        x = x.view(-1, x_shape[-1])
        M, N = x.shape
        assert N % 2 == 0, f"N must be even for silu_mul, got N={N}"
        D = N // 2  # output dim after silu_mul

        ng = cdiv(D, self.group_size)
        BLOCK_G = triton.next_power_of_2(self.group_size)

        MAX_FUSED_SIZE = 65536 // x.element_size()
        if BLOCK_G > MAX_FUSED_SIZE:
            raise RuntimeError("This kernel doesn't support group_size >= 64KB.")

        num_warps = min(max(BLOCK_G // 256, 1), 8)
        rows_per_block = calc_rows_per_block(M, x.device)

        y_quant = torch.empty((M, D), dtype=_FP8_DTYPE, device=x.device)
        scales = torch.empty((M, ng), dtype=torch.float32, device=x.device)

        grid = (cdiv(M, rows_per_block), ng)
        silu_mul_input_quant_fp8_kernel[grid](
            x,
            y_quant,
            scales,
            x.stride(0),
            x.stride(1),
            y_quant.stride(0),
            scales.stride(0),
            M,
            D,
            self.group_size,
            BLOCK_G=BLOCK_G,
            ROWS_PER_BLOCK=rows_per_block,
            FP8_MIN=_FP8_MIN,
            FP8_MAX=_FP8_MAX,
            USE_UE8M0=self.use_ue8m0,
            FP8_MIN_SCALING_FACTOR=_FP8_MIN_SCALING_FACTOR,
            num_warps=num_warps,
        )

        out_shape = x_shape[:-1] + (D,)
        y_quant = y_quant.view(out_shape)
        scales = scales.view(x_shape[:-1] + (ng,))

        return y_quant, scales
