# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Layout dispatch tests for ``QwenGatedDeltaNetAttention._forward_core_rocm``.

The AITER fused reshape+conv kernel learned Qwen3.5's flat ``[q|k|v|z]``
packing in https://github.com/ROCm/aiter/pull/3251, so flat-layout models take
the decode fast path too. Builds predating that only understand Qwen3-Next's
interleaved packing and would read the wrong columns from a flat tensor, so
flat layouts must keep falling back to the generic path there.

These tests run host-side on CPU: ``_forward_core_rocm`` is bound to a stub
layer whose ``_forward_core_decode_aiter``/``_forward_core`` record which
branch ran, so no GPU or AITER install is needed.
"""

from __future__ import annotations

import types
from typing import cast
from unittest.mock import patch

import pytest
import torch

from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
    _resolve_aiter_conv_layout_kwargs,
)
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

PREFIX = "model.layers.0.linear_attn"
H = 2  # num key heads
HV = 4  # num value heads
K = 8  # head_k_dim
V = 8  # head_v_dim


def _make_metadata(
    *,
    num_prefills: int = 0,
    num_decodes: int = 2,
    spec_sequence_masks: torch.Tensor | None = None,
):
    """Only the fields the dispatch condition reads carry meaningful values."""
    return GDNAttentionMetadata(
        num_prefills=num_prefills,
        num_prefill_tokens=num_prefills,
        num_decodes=num_decodes,
        num_decode_tokens=num_decodes,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=num_prefills + num_decodes,
        spec_sequence_masks=spec_sequence_masks,
    )


def _make_layer(gqa_interleaved_layout: bool, aiter_supports_layout: bool):
    """Stub layer running the real ``_forward_core_rocm``, recording dispatch."""
    layer = types.SimpleNamespace()
    layer.prefix = PREFIX
    layer.gqa_interleaved_layout = gqa_interleaved_layout
    layer.qkvz_layout = "interleaved" if gqa_interleaved_layout else "flat"
    with patch.object(
        qwen_gdn_linear_attn,
        "GDN_AITER_SUPPORTS_QKVZ_LAYOUT",
        aiter_supports_layout,
    ):
        layer._aiter_conv_layout_kwargs = _resolve_aiter_conv_layout_kwargs(
            layer.qkvz_layout
        )

    layer.calls = []
    layer._forward_core_decode_aiter = lambda **kw: layer.calls.append("aiter")
    layer._forward_core = lambda **kw: layer.calls.append("generic")
    layer.prepare_gdn_attention_core_inputs = lambda qkvz, ba, n: (
        torch.zeros(n, H * K * 2 + HV * V),
        torch.zeros(n, HV, V),
        torch.zeros(n, HV),
        torch.zeros(n, HV),
    )
    layer._forward_core_rocm = types.MethodType(
        QwenGatedDeltaNetAttention._forward_core_rocm, layer
    )
    return layer


def _run(layer, meta) -> str:
    num_tokens = max(meta.num_actual_tokens, 1)
    ctx = types.SimpleNamespace(attn_metadata={PREFIX: meta})
    with patch.object(qwen_gdn_linear_attn, "get_forward_context", return_value=ctx):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, 2 * H * K + 2 * HV * V),
            ba=torch.zeros(num_tokens, 2 * HV),
            z_out=torch.zeros(num_tokens, HV, V),
            core_attn_out=torch.zeros(num_tokens, HV, V),
        )
    assert len(layer.calls) == 1
    return layer.calls[0]


@pytest.mark.parametrize("aiter_supports_layout", [True, False])
def test_interleaved_always_takes_the_fast_path(aiter_supports_layout: bool) -> None:
    """Qwen3-Next reached the fast path before ``qkvz_layout`` existed."""
    layer = _make_layer(True, aiter_supports_layout)
    assert _run(layer, _make_metadata()) == "aiter"


def test_flat_takes_the_fast_path_when_aiter_understands_the_layout() -> None:
    layer = _make_layer(False, True)
    with patch.object(qwen_gdn_linear_attn, "GDN_AITER_SUPPORTS_QKVZ_LAYOUT", True):
        assert _run(layer, _make_metadata()) == "aiter"


def test_flat_falls_back_on_aiter_without_qkvz_layout() -> None:
    """Feeding flat tensors to an interleaved-only kernel destroys accuracy."""
    layer = _make_layer(False, False)
    with patch.object(qwen_gdn_linear_attn, "GDN_AITER_SUPPORTS_QKVZ_LAYOUT", False):
        assert _run(layer, _make_metadata()) == "generic"


@pytest.mark.parametrize("gqa_interleaved_layout", [True, False])
@pytest.mark.parametrize(
    "meta_kwargs",
    [
        {"num_prefills": 1},
        {"num_decodes": 0},
        {"spec_sequence_masks": torch.zeros(1, dtype=torch.bool)},
    ],
    ids=["has_prefill", "no_decode", "spec_decode"],
)
def test_non_pure_decode_batches_use_the_generic_path(
    gqa_interleaved_layout: bool, meta_kwargs: dict
) -> None:
    layer = _make_layer(gqa_interleaved_layout, True)
    with patch.object(qwen_gdn_linear_attn, "GDN_AITER_SUPPORTS_QKVZ_LAYOUT", True):
        assert _run(layer, _make_metadata(**meta_kwargs)) == "generic"


@pytest.mark.parametrize("qkvz_layout", ["flat", "interleaved"])
def test_layout_kwarg_is_forwarded_when_supported(qkvz_layout: str) -> None:
    with patch.object(qwen_gdn_linear_attn, "GDN_AITER_SUPPORTS_QKVZ_LAYOUT", True):
        assert _resolve_aiter_conv_layout_kwargs(qkvz_layout) == {
            "qkvz_layout": qkvz_layout
        }


@pytest.mark.parametrize("qkvz_layout", ["flat", "interleaved"])
def test_layout_kwarg_is_omitted_when_unsupported(qkvz_layout: str) -> None:
    """Passing it to an old AITER would raise TypeError; its default matches."""
    with patch.object(qwen_gdn_linear_attn, "GDN_AITER_SUPPORTS_QKVZ_LAYOUT", False):
        assert _resolve_aiter_conv_layout_kwargs(qkvz_layout) == {}


def test_fused_gdr_norm_capability_is_signature_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def old_wrapper(core_attn_out=None):
        return core_attn_out

    def new_wrapper(
        core_attn_out=None,
        output_gate=None,
        norm_weight=None,
        norm_eps=None,
    ):
        return core_attn_out

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "GDN_AITER_TRITON_AVAILABLE",
        True,
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule",
        old_wrapper,
        raising=False,
    )
    assert not qwen_gdn_linear_attn._gdn_aiter_supports_fused_gdr_rmsnorm_silu()

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule",
        new_wrapper,
    )
    assert qwen_gdn_linear_attn._gdn_aiter_supports_fused_gdr_rmsnorm_silu()


def test_forward_hip_uses_fused_gdr_norm_output_directly() -> None:
    num_tokens = 2
    local_value_heads = 16
    head_v_dim = 128
    hidden_states = torch.zeros(num_tokens, 1, dtype=torch.bfloat16)
    qkvz = torch.zeros(num_tokens, 8, dtype=torch.bfloat16)
    ba = torch.zeros(num_tokens, 4, dtype=torch.bfloat16)
    captured = {}

    layer = types.SimpleNamespace(
        prefix=PREFIX,
        num_v_heads=local_value_heads,
        tp_size=1,
        head_v_dim=head_v_dim,
        enable_aiter_fused_gdr_norm=True,
        norm=types.SimpleNamespace(weight=torch.ones(head_v_dim, dtype=torch.bfloat16)),
        in_proj_qkvz=lambda _: (qkvz, None),
        in_proj_ba=lambda _: (ba, None),
        out_proj=lambda x: (x, None),
        _output_projection=lambda *_: pytest.fail(
            "standalone RMSNormGated should be skipped"
        ),
    )
    layer.forward_hip = types.MethodType(QwenGatedDeltaNetAttention.forward_hip, layer)

    def fake_attention_core(*args, **kwargs):
        captured.update(kwargs)
        args[3].fill_(1)

    with (
        patch.object(qwen_gdn_linear_attn, "GDN_AITER_TRITON_AVAILABLE", True),
        patch.object(
            torch.ops.vllm,
            "qwen_gdn_attention_core",
            side_effect=fake_attention_core,
        ),
    ):
        output = layer.forward_hip(hidden_states)

    assert captured["use_aiter"] is True
    assert captured["fuse_norm"] is True
    torch.testing.assert_close(output, torch.ones_like(output))


def test_rocm_fused_norm_mode_normalizes_generic_fallback() -> None:
    layer = _make_layer(gqa_interleaved_layout=True, aiter_supports_layout=True)
    metadata = _make_metadata(num_prefills=1)
    num_tokens = metadata.num_actual_tokens
    z = torch.full((num_tokens, HV, V), 3.0)
    layer.prepare_gdn_attention_core_inputs = lambda qkvz, ba, n: (
        torch.zeros(n, H * K * 2 + HV * V),
        z,
        torch.zeros(n, HV),
        torch.zeros(n, HV),
    )
    layer._forward_core = lambda **kwargs: kwargs["core_attn_out"].fill_(2.0)
    layer.norm = types.SimpleNamespace(forward_native=lambda x, gate: x + gate)

    z_out = torch.zeros_like(z)
    core_attn_out = torch.zeros_like(z)
    context = types.SimpleNamespace(attn_metadata={PREFIX: metadata})
    with patch.object(
        qwen_gdn_linear_attn, "get_forward_context", return_value=context
    ):
        layer._forward_core_rocm(
            qkvz=torch.zeros(num_tokens, 2 * H * K + 2 * HV * V),
            ba=torch.zeros(num_tokens, 2 * HV),
            z_out=z_out,
            core_attn_out=core_attn_out,
            fuse_norm=True,
        )

    torch.testing.assert_close(z_out, z)
    torch.testing.assert_close(core_attn_out, torch.full_like(core_attn_out, 5.0))


def test_decode_aiter_forwards_fused_norm_epilogue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    num_tokens = 2
    local_h = 2
    local_hv = 16
    dim = 128
    qkv_dim = 2 * local_h * dim + local_hv * dim
    state_indices = torch.tensor([3, 1], dtype=torch.int32)
    cu_seqlens = torch.arange(num_tokens + 1, dtype=torch.int32)
    mixed_qkv = torch.zeros(num_tokens, qkv_dim, dtype=torch.bfloat16)
    b = torch.zeros(num_tokens, local_hv, dtype=torch.bfloat16)
    a = torch.zeros_like(b)
    z_out = torch.zeros(num_tokens, local_hv, dim, dtype=torch.bfloat16)
    core_attn_out = torch.zeros_like(z_out)
    norm_weight = torch.ones(dim, dtype=torch.bfloat16)
    captured = {}

    layer = types.SimpleNamespace(
        tp_size=1,
        num_k_heads=local_h,
        num_v_heads=local_hv,
        head_k_dim=dim,
        head_v_dim=dim,
        key_dim=local_h * dim,
        value_dim=local_hv * dim,
        activation="silu",
        A_log=torch.zeros(local_hv, dtype=torch.float32),
        dt_bias=torch.zeros(local_hv, dtype=torch.bfloat16),
        conv1d=types.SimpleNamespace(
            weight=torch.zeros(qkv_dim, 1, 4, dtype=torch.bfloat16),
            bias=None,
        ),
        kv_cache=(
            torch.empty(4, 1),
            torch.empty(4, local_hv, dim, dim, dtype=torch.float32),
        ),
        norm=types.SimpleNamespace(weight=norm_weight, eps=1e-6),
        _aiter_conv_layout_kwargs={},
    )
    metadata = types.SimpleNamespace(
        non_spec_query_start_loc=cu_seqlens,
        non_spec_state_indices_tensor=state_indices,
        num_actual_tokens=num_tokens,
        num_decodes=num_tokens,
    )

    def fake_conv(*args, **kwargs):
        return mixed_qkv, b, a

    def fake_gdr(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "is_conv_state_dim_first",
        lambda: True,
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "gdn_aiter_fused_reshape_causal_conv1d_update_single_token",
        fake_conv,
        raising=False,
    )
    monkeypatch.setattr(
        qwen_gdn_linear_attn,
        "gdn_aiter_fused_rearrange_sigmoid_gated_delta_rule",
        fake_gdr,
        raising=False,
    )

    QwenGatedDeltaNetAttention._forward_core_decode_aiter(
        cast(QwenGatedDeltaNetAttention, layer),
        qkvz=torch.zeros(num_tokens, 1, dtype=torch.bfloat16),
        ba=torch.zeros(num_tokens, 1, dtype=torch.bfloat16),
        z_out=z_out,
        core_attn_out=core_attn_out,
        attn_metadata=cast(GDNAttentionMetadata, metadata),
        fuse_norm=True,
    )

    assert captured["output_gate"].data_ptr() == z_out.data_ptr()
    assert captured["norm_weight"] is norm_weight
    assert captured["norm_eps"] == 1e-6
    assert captured["use_qk_l2norm_in_kernel"] is True
    assert captured["ssm_state_indices"] is state_indices
