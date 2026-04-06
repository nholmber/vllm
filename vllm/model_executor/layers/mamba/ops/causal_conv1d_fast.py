# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py


import numpy as np
import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

@triton.jit()
def _causal_conv1d_update_fast_kernel(
    # Pointers to matrices
    x_ptr,  # (batch, dim, seqlen)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (batch, dim, seqlen)
    # Matrix dimensions
    batch: int,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    # [BLOCK_N,] elements along the feature-dimension (channel)
    idx_feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)

    if IS_APC_ENABLED:
        # Get the state from the initial_state_idx
        conv_state_init = tl.load(initial_state_idx + idx_seq)
        current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
    else:
        conv_state_init = 0
        current_last_index = 0

    # cache_idx
    conv_states_input_coord = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
    ).to(tl.int64)

    if USE_PAD_SLOT:  # noqa
        if conv_states_input_coord == pad_slot_id:
            # not processing as this is not the actual sequence
            return

    # IS_VARLEN is False
    query_start_index = idx_seq * seqlen
    query_end_index = query_start_index + seqlen
    x_offset = idx_seq * stride_x_seq
    o_offset = idx_seq * stride_o_seq

    if query_start_index == query_end_index:
        return

    # IS_SPEC_DECODING is False
    conv_state_token_offset = 0

    # STEP 1: READ init_state data
    # note: NP2_STATELEN = triton.next_power_of_2(KERNEL_WIDTH - 1)
    idx_cols = tl.arange(0, NP2_STATELEN)
    conv_state_ptrs_cols = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[:, None]
        + (idx_cols * stride_conv_state_tok)[None, :]
    )  # [BLOCK_N, NP2_STATELEN]
    mask_cols = (
        (conv_states_input_coord < num_cache_lines)
        & (idx_feats < dim)[:, None]
        & (idx_cols < KERNEL_WIDTH - 1)[None, :]
    )
    cols = tl.load(conv_state_ptrs_cols, mask_cols, other=0.0)

    # STEP 2: assume state_len > seqlen
    idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

    # With speculative decoding, the conv_state updates works in a sliding
    # window manner, at each forward pass, the tokens are shift by 1, so we
    # load since idx_tokens + 1.
    conv_state_ptrs_source = (
        conv_state_ptr
        + (conv_states_input_coord * stride_conv_state_seq)
        + conv_state_token_offset * stride_conv_state_tok
        + (idx_feats * stride_conv_state_dim)[None, :]
        + ((idx_tokens + seqlen) * stride_conv_state_tok)[
            :, None
        ]
    )  # [BLOCK_M, BLOCK_N]
    mask = (
        (conv_states_input_coord < num_cache_lines)
        & ((idx_tokens + seqlen) < state_len)[:, None]
        & (idx_feats < dim)[None, :]
    )
    conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

    VAL = state_len - seqlen
    x_base = x_ptr + x_offset + (idx_feats * stride_x_dim)  # [BLOCK_N]

    x_ptrs = (
        x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
    )  # [BLOCK_M, BLOCK_N]

    mask_x = (
        (idx_tokens - VAL >= 0)[:, None]
        & (idx_tokens - VAL < seqlen)[:, None]
        & (idx_feats < dim)[None, :]
    )  # token-index  # token-index  # feature-index
    loaded_x = tl.load(x_ptrs, mask_x, 0.0)
    tl.debug_barrier()

    new_conv_state = tl.where(mask, conv_state, loaded_x)

    # Get the state from the initial_state_idx
    # cache_idx
    conv_states_offset = tl.load(
        conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
    ).to(tl.int64)
    conv_state_ptrs_target = (
        conv_state_ptr
        + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
        + (idx_feats * stride_conv_state_dim)
    )[None, :] + (  # [BLOCK_N,]
        idx_tokens * stride_conv_state_tok
    )[:, None]
    mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
    tl.store(conv_state_ptrs_target, new_conv_state, mask)

    # STEP 3: init accumulator, not necessary
    #if HAS_BIAS:
    #    bias = bias_ptr + idx_feats
    #    mask_bias = idx_feats < dim
    #    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
    #        tl.float32
    #    )  # [BLOCK_N]
    #else:
    #    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # STEP 4:
    # LOAD WEIGHTS and compute
    w_cols_ptrs = w_ptr + (idx_feats * stride_w_dim)[:, None] + (idx_cols * stride_w_width)[None, :]
    mask_w_cols = (idx_feats < dim)[:, None] & (idx_cols < KERNEL_WIDTH - 1)[None, :]
    w_cols = tl.load(w_cols_ptrs, mask_w_cols, other=0.0)  # [BLOCK_N, NP2_STATELEN]

    w_last_ptrs = w_ptr + (idx_feats * stride_w_dim) + (KERNEL_WIDTH - 1) * stride_w_width
    w_last = tl.load(w_last_ptrs, idx_feats < dim, other=0.0) # [BLOCK_N]

    # For the convolution output: dot(weights, [state_cols | x])
    # cols is [BLOCK_N, NP2_STATELEN] = conv_state history
    # We need x as 1D [BLOCK_N] for the last weight column
    x_1d = tl.load(x_base, mask=(idx_feats < dim), other=0.0)  # [BLOCK_N], reload as 1D
    acc = tl.sum((w_cols * cols).to(tl.float32), axis=1) + (w_last * x_1d).to(tl.float32)

    if HAS_BIAS:
        bias = bias_ptr + idx_feats
        acc += tl.load(bias, idx_feats < dim, other=0.0).to(
            tl.float32
        )  # [BLOCK_N]

    if SILU_ACTIVATION:
        acc = acc / (1 + tl.exp(-acc))
    mask_1d = idx_feats < dim
    o_ptrs = (
        o_ptr + o_offset + (idx_feats * stride_o_dim)
    )

    tl.store(o_ptrs, acc, mask=mask_1d)

def causal_conv1d_update_fast(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    x: Input tensor which can take the following shapes:

    - `[batch, dim]` - single token prediction
    - `[batch, dim, seqlen]` - single or multiple tokens prediction
    - `[num_tokens, dim]` - continuous batching, where num_tokens is
        the total tokens of all sequences in that batch

    conv_state: (..., dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim,
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into conv_state_indices, where the last cache block to be filled is located.
    initial_state_idx: (batch,), dtype int32
        The pointer into conv_state_indices, where the cache block containing the initial state is located.
    num_accepted_tokens: (batch,), dtype int32
        In this implementation, it must be None
        If not None, it indicates the number of accepted tokens for each
        sequence in the batch.
        This is used in speculative decoding, where the conv_state is updated
        in a sliding window manner.
    query_start_loc: (batch + 1,) int32
        In this implementation, it must be None
        If not None, the inputs is given in a varlen fashion and this indicates
        the starting index of each sequence in the batch.
    max_query_len: int
        If query_start_loc is not None, this indicates the maximum query
        length in the batch.
    pad_slot_id: int
        if conv_state_indices is passed, lets the kernel identify padded
        entries that will not be processed,
        for example: conv_state_indices = [pad_slot_id, 1 ,20 ,pad_slot_id]
        in this case, the kernel will not process entries at
        indices 0 and 3
    out: (batch, dim) or (batch, dim, seqlen) or (num_tokens, dim), same shape as `x`
    """
    assert num_accepted_tokens is None, f"num_accepted_tokens must be None, got {num_accepted_tokens}"
    assert query_start_loc is None, f"query_start_loc must be None, got {query_start_loc}"
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    if query_start_loc is None:
        batch, dim, seqlen = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this

    # adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    out = x
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        # X (batch, dim, seqlen)
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # X (dim, cu_seqlen)
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)

    def grid(META):
        return (
            batch,
            triton.cdiv(dim, META["BLOCK_N"]),
        )

    _causal_conv1d_update_fast_kernel[grid](
        # Pointers to matrices
        x,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        # Matrix dimensions
        batch,
        dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        # others
        pad_slot_id,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    return out.to(original_x_dtype)


@triton.jit()
def _reshape_causal_conv1d_update_fast_kernel(
    # Pointers to matrices
    x_ptr,  # (num_tokens, dim+z_dim, seqlen) where seqlen=1
    ba_ptr,
    z_ptr, # (num_tokens, num_v_heads, head_v_dim)
    b_ptr, # (num_accepted_tokens, num_v_heads)
    a_ptr, # (num_accepted_tokens, num_v_heads)
    w_ptr,  # (dim, width)
    bias_ptr,
    conv_state_ptr,
    conv_state_indices_ptr,
    block_idx_last_scheduled_token,  # (batch,)
    initial_state_idx,  # (batch,)
    o_ptr,  # (num_accepted_tokens, dim, seqlen)
    # Matrix dimensions
    batch: int,
    num_k_heads: tl.constexpr,
    num_v_heads: tl.constexpr,
    head_k_dim: tl.constexpr,
    head_v_dim: tl.constexpr,
    dim: tl.constexpr,
    head_qkvz_dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines: tl.constexpr,  # added to support vLLM larger cache lines
    # Strides
    stride_x_seq: tl.constexpr,
    stride_x_dim: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_dim: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_conv_state_seq: tl.constexpr,
    stride_conv_state_dim: tl.constexpr,
    stride_conv_state_tok: tl.constexpr,
    stride_state_indices: tl.constexpr,
    stride_o_seq: tl.constexpr,
    stride_o_dim: tl.constexpr,
    stride_o_token: tl.constexpr,
    stride_z_seq: tl.constexpr,
    stride_ba_seq: tl.constexpr,
    stride_ba_token: tl.constexpr,
    stride_b_seq: tl.constexpr,
    # others
    pad_slot_id: tl.constexpr,
    num_program_write_z: tl.constexpr,
    BLOCK_Z: tl.constexpr,
    HV: tl.constexpr,
    # Meta-parameters
    HAS_BIAS: tl.constexpr,
    KERNEL_WIDTH: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    IS_APC_ENABLED: tl.constexpr,
    NP2_STATELEN: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # ruff: noqa: E501
    idx_seq = tl.program_id(0)
    if idx_seq >= batch:
        return

    ## write b, a
    if tl.program_id(1) == 0:
        ## HV = triton.next_power_of_2(num_v_heads)
        idx_hv = tl.arange(0, HV)
        ## map idx_hv to source idx
        idx_h = idx_hv // (num_v_heads // num_k_heads)
        idx_v = idx_hv % (num_v_heads // num_k_heads)
        b_source_offset = idx_h * (2 * num_v_heads // num_k_heads) + idx_v
        a_source_offset = idx_h * (2 * num_v_heads // num_k_heads) + num_v_heads // num_k_heads + idx_v

        b_source_ptrs = ba_ptr + idx_seq * stride_ba_seq + b_source_offset * stride_ba_token
        a_source_ptrs = ba_ptr + idx_seq * stride_ba_seq + a_source_offset * stride_ba_token
        mask_ba = idx_hv < num_v_heads
        b = tl.load(b_source_ptrs, mask=mask_ba, other=0.0)
        a = tl.load(a_source_ptrs, mask=mask_ba, other=0.0)
        ## b, a should be contiguous so the last stride is 1
        b_ptrs = b_ptr + idx_seq * stride_b_seq + idx_hv
        a_ptrs = a_ptr + idx_seq * stride_b_seq + idx_hv
        tl.store(b_ptrs, b, mask_ba)
        tl.store(a_ptrs, a, mask_ba)
    ## write z
    elif tl.program_id(1) < 1 + num_program_write_z:
        idx_z = (tl.program_id(1) -  1) * BLOCK_Z + tl.arange(0, BLOCK_Z)
        ## map idx_z to source idx
        idx_z_x = idx_z // (num_v_heads // num_k_heads * head_v_dim) * head_qkvz_dim + 2 * head_k_dim + num_v_heads // num_k_heads * head_v_dim + idx_z % (num_v_heads // num_k_heads * head_v_dim)
        z_source_ptrs = x_ptr + idx_seq * stride_x_seq + idx_z_x * stride_x_dim
        mask_z = idx_z < num_v_heads * head_v_dim
        z = tl.load(z_source_ptrs, mask=mask_z, other=0.0)
        z_ptrs = z_ptr + idx_seq * stride_z_seq + idx_z
        tl.store(z_ptrs, z, mask=mask_z)
    ## do regular causal conv1d udpate
    else:
        # [BLOCK_N,] elements along the feature-dimension (channel)
        idx_feats = (tl.program_id(1) - 1 - num_program_write_z) * BLOCK_N + tl.arange(0, BLOCK_N)
        ## map idx_feats to idx_feats_x
        idx_feats_x = (
            (idx_feats < num_k_heads * head_k_dim).to(tl.int64) * (idx_feats // head_k_dim * head_qkvz_dim + idx_feats % head_k_dim)
            + ((idx_feats >= num_k_heads * head_k_dim) & (idx_feats < num_k_heads * head_k_dim * 2)).to(tl.int64) * ((idx_feats - num_k_heads * head_k_dim) // head_k_dim * head_qkvz_dim + head_k_dim + (idx_feats - num_k_heads * head_k_dim) % head_k_dim)
            + (idx_feats >= num_k_heads * head_k_dim * 2).to(tl.int64) * ((idx_feats - num_k_heads * head_k_dim * 2) // (num_v_heads // num_k_heads * head_v_dim) * head_qkvz_dim + 2 * head_k_dim + (idx_feats - num_k_heads * head_k_dim * 2) % (num_v_heads // num_k_heads * head_v_dim))
        )

        if IS_APC_ENABLED:
            # Get the state from the initial_state_idx
            conv_state_init = tl.load(initial_state_idx + idx_seq)
            current_last_index = tl.load(block_idx_last_scheduled_token + idx_seq)
        else:
            conv_state_init = 0
            current_last_index = 0

        # cache_idx
        conv_states_input_coord = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + conv_state_init
        ).to(tl.int64)

        if USE_PAD_SLOT:  # noqa
            if conv_states_input_coord == pad_slot_id:
                # not processing as this is not the actual sequence
                return

        # IS_VARLEN is False
        query_start_index = idx_seq * seqlen
        query_end_index = query_start_index + seqlen
        x_offset = idx_seq * stride_x_seq
        o_offset = idx_seq * stride_o_seq

        if query_start_index == query_end_index:
            return

        # STEP 1: READ init_state data
        # note: NP2_STATELEN = triton.next_power_of_2(KERNEL_WIDTH - 1)
        idx_cols = tl.arange(0, NP2_STATELEN)
        conv_state_ptrs_cols = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[:, None]
            + (idx_cols * stride_conv_state_tok)[None, :]
        )  # [BLOCK_N, NP2_STATELEN]
        mask_cols = (
            (conv_states_input_coord < num_cache_lines)
            & (idx_feats < dim)[:, None]
            & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        )
        cols = tl.load(conv_state_ptrs_cols, mask_cols, other=0.0)

        # STEP 2: assume state_len > seqlen
        idx_tokens = tl.arange(0, NP2_STATELEN)  # [BLOCK_M]

        # With speculative decoding, the conv_state updates works in a sliding
        # window manner, at each forward pass, the tokens are shift by 1, so we
        # load since idx_tokens + 1.
        conv_state_ptrs_source = (
            conv_state_ptr
            + (conv_states_input_coord * stride_conv_state_seq)
            + (idx_feats * stride_conv_state_dim)[None, :]
            + ((idx_tokens + seqlen) * stride_conv_state_tok)[
                :, None
            ]
        )  # [BLOCK_M, BLOCK_N]
        mask = (
            (conv_states_input_coord < num_cache_lines)
            & ((idx_tokens + seqlen) < state_len)[:, None]
            & (idx_feats < dim)[None, :]
        )
        conv_state = tl.load(conv_state_ptrs_source, mask, other=0.0)

        VAL = state_len - seqlen
        x_base = x_ptr + x_offset + (idx_feats_x * stride_x_dim)  # [BLOCK_N]

        x_ptrs = (
            x_base[None, :] + ((idx_tokens - VAL) * stride_x_token)[:, None]
        )  # [BLOCK_M, BLOCK_N]

        mask_x = (
            (idx_tokens - VAL >= 0)[:, None]
            & (idx_tokens - VAL < seqlen)[:, None]
            & (idx_feats < dim)[None, :]
        )  # token-index  # token-index  # feature-index
        loaded_x = tl.load(x_ptrs, mask_x, 0.0)
        tl.debug_barrier()

        new_conv_state = tl.where(mask, conv_state, loaded_x)

        # Get the state from the initial_state_idx
        # cache_idx
        conv_states_offset = tl.load(
            conv_state_indices_ptr + idx_seq * stride_state_indices + current_last_index
        ).to(tl.int64)
        conv_state_ptrs_target = (
            conv_state_ptr
            + (conv_states_offset * stride_conv_state_seq)  # Offset from seq
            + (idx_feats * stride_conv_state_dim)
        )[None, :] + (  # [BLOCK_N,]
            idx_tokens * stride_conv_state_tok
        )[:, None]
        mask = (idx_tokens < state_len)[:, None] & (idx_feats < dim)[None, :]
        tl.store(conv_state_ptrs_target, new_conv_state, mask)

        # STEP 3: init accumulator, not necessary
        #if HAS_BIAS:
        #    bias = bias_ptr + idx_feats
        #    mask_bias = idx_feats < dim
        #    acc_preload = tl.load(bias, mask=mask_bias, other=0.0).to(
        #        tl.float32
        #    )  # [BLOCK_N]
        #else:
        #    acc_preload = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # STEP 4:
        # LOAD WEIGHTS and compute
        w_cols_ptrs = w_ptr + (idx_feats * stride_w_dim)[:, None] + (idx_cols * stride_w_width)[None, :]
        mask_w_cols = (idx_feats < dim)[:, None] & (idx_cols < KERNEL_WIDTH - 1)[None, :]
        w_cols = tl.load(w_cols_ptrs, mask_w_cols, other=0.0)  # [BLOCK_N, NP2_STATELEN]

        w_last_ptrs = w_ptr + (idx_feats * stride_w_dim) + (KERNEL_WIDTH - 1) * stride_w_width
        w_last = tl.load(w_last_ptrs, idx_feats < dim, other=0.0) # [BLOCK_N]

        # For the convolution output: dot(weights, [state_cols | x])
        # cols is [BLOCK_N, NP2_STATELEN] = conv_state history
        # We need x as 1D [BLOCK_N] for the last weight column
        x_1d = tl.load(x_base, mask=(idx_feats < dim), other=0.0)  # [BLOCK_N], reload as 1D
        acc = tl.sum((w_cols * cols).to(tl.float32), axis=1) + (w_last * x_1d).to(tl.float32)

        if HAS_BIAS:
            bias = bias_ptr + idx_feats
            acc += tl.load(bias, idx_feats < dim, other=0.0).to(
                tl.float32
            )  # [BLOCK_N]

        if SILU_ACTIVATION:
            acc = acc / (1 + tl.exp(-acc))
        mask_1d = idx_feats < dim
        o_ptrs = (
            o_ptr + o_offset + (idx_feats * stride_o_dim)
        )

        tl.store(o_ptrs, acc, mask=mask_1d)

def fused_reshape_causal_conv1d_update_fast(
    x: torch.Tensor, 
    num_actual_tokens: int,
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
    ba: torch.Tensor, 
    z_out: torch.Tensor, 
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,

):
    """
    Inputs (part):
        x: qkvz, can be viewed (num_tokens, qkvz_dim)
        ba: can be viewed (num_tokens, 2 * num_v_heads)
        z_out: (num_tokens, num_v_heads, head_v_dim)
    Outputs:
        mixed_qkv (out): (num_actual_tokens, dim)
        b: (num_actual_tokens, num_v_heads)
        a: (num_actual_tokens, num_v_heads)
        z_out (in-place modified): (num_tokens, num_v_heads, head_v_dim)
    """
    assert num_accepted_tokens is None, f"num_accepted_tokens must be None, got {num_accepted_tokens}"
    assert query_start_loc is None, f"query_start_loc must be None, got {query_start_loc}"
    assert z_out.is_contiguous(), "z_out should be contiguous if it was created by torch.zeros()"
    x = x.view(x.shape[0], -1)
    ba = ba.view(ba.shape[0], -1)
    original_z_shape = z_out.shape
    z_out = z_out.view(original_z_shape[0], -1)
    if validate_data:
        assert pad_slot_id is not None
        assert x.stride(1) == 1
    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)
    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        # make it (batch, dim, seqlen) with seqlen == 1
        x = x.unsqueeze(-1)
    _, qkvz_dim, seqlen = x.shape
    batch = num_actual_tokens
    _, width = weight.shape
    # conv_state: (..., dim, state_len), where state_len >= width - 1
    ## dim for qkv
    head_dim = (
        head_k_dim
        + head_k_dim 
        + head_v_dim * num_v_heads // num_k_heads 
    )
    head_qkvz_dim = head_dim + head_v_dim * num_v_heads // num_k_heads
    dim = num_k_heads * head_dim
    expected_qkvz_dim = num_k_heads * head_qkvz_dim
    assert qkvz_dim == expected_qkvz_dim, f"ERROR: expect qkvz_dim to be {expected_qkvz_dim}, got {qkvz_dim}"
    num_cache_lines, _, state_len = conv_state.size()

    if validate_data:
        assert dim == weight.size(0)
        assert conv_state.stride(-2) == 1, (
            f"ERROR: expect contiguous along feat-dim of conv_state (currently stride={conv_state.stride()})"
        )
        assert state_len >= width - 1
        # when above happens, we don't shift-left to keep any records in conv_state
        assert dim == conv_state.size(1)
        if conv_state_indices is None:
            assert conv_state.size(0) >= batch
        else:
            assert (batch,) == conv_state_indices.shape

        assert num_cache_lines >= batch
        assert weight.stride(1) == 1  # Need this

    out = torch.empty((num_actual_tokens, dim, seqlen), dtype=x.dtype, device=x.device)
    b_out = torch.empty((num_actual_tokens, num_v_heads), dtype=ba.dtype, device=ba.device)
    a_out = torch.empty((num_actual_tokens, num_v_heads), dtype=ba.dtype, device=ba.device)
    ## adopt the strategy in vLLM that overwrite on 'x' directly, rather than creating a new tensor 'o'
    #out = x
    stride_w_dim, stride_w_width = weight.stride()

    if query_start_loc is None:
        # X (batch, dim, seqlen)
        stride_x_seq, stride_x_dim, stride_x_token = x.stride()
        stride_o_seq, stride_o_dim, stride_o_token = out.stride()
    else:
        # X (dim, cu_seqlen)
        stride_x_token, stride_x_dim = x.stride()
        stride_x_seq = 0
        stride_o_token, stride_o_dim = out.stride()
        stride_o_seq = 0

    stride_z_seq = z_out.stride(0)
    stride_ba_seq, stride_ba_token = ba.stride()
    stride_b_seq = b_out.stride(0)

    stride_istate_seq, stride_istate_dim, stride_istate_token = conv_state.stride()
    stride_state_indices = (
        conv_state_indices.stride(0) if conv_state_indices is not None else 0
    )
    if num_accepted_tokens is not None:
        state_len = width - 1 + (seqlen - 1)  # effective state_len needed
    else:
        state_len = width - 1
    np2_statelen = triton.next_power_of_2(state_len)
    HV = triton.next_power_of_2(num_v_heads)
    BLOCK_Z=512
    num_program_write_z = triton.cdiv(num_v_heads * head_v_dim, BLOCK_Z)

    ## additional programs for z and ba
    def grid(META):
        return (
            batch,
            1 + num_program_write_z + triton.cdiv(dim, META["BLOCK_N"]),
        )

    _reshape_causal_conv1d_update_fast_kernel[grid](
        # Pointers to matrices
        x,
        ba,
        z_out,
        b_out,
        a_out,
        weight,
        bias,
        conv_state,
        conv_state_indices,
        block_idx_last_scheduled_token,
        initial_state_idx,
        out,
        # Matrix dimensions
        batch,
        num_k_heads,
        num_v_heads,
        head_k_dim,
        head_v_dim,
        dim,
        head_qkvz_dim,
        seqlen,
        state_len,
        num_cache_lines,
        # stride
        stride_x_seq,
        stride_x_dim,
        stride_x_token,
        stride_w_dim,
        stride_w_width,
        stride_istate_seq,
        stride_istate_dim,
        stride_istate_token,
        stride_state_indices,
        stride_o_seq,
        stride_o_dim,
        stride_o_token,
        stride_z_seq,
        stride_ba_seq,
        stride_ba_token,
        stride_b_seq,
        # others
        pad_slot_id,
        num_program_write_z,
        BLOCK_Z,
        HV=HV,
        # META
        HAS_BIAS=bias is not None,
        KERNEL_WIDTH=width,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        IS_APC_ENABLED=block_idx_last_scheduled_token is not None,
        NP2_STATELEN=np2_statelen,
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=256,
    )
    if unsqueeze:
        out = out.squeeze(-1)
    z_out = z_out.view(original_z_shape)
    return out.to(original_x_dtype), b_out, a_out

