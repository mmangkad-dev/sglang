# SPDX-License-Identifier: Apache-2.0
"""Shared FP8 post-load weight processing.

Six quantization families run the same sequence after an FP8 checkpoint is
loaded: settle the weight scale to the granularity the kernel wants,
normalize to ``e4m3fnuz`` on ROCm, then commit the weight and its scale as
non-trainable parameters in the layout the GEMM expects. Every family had its
own copy, and the copies had drifted.

This module is deliberately not inside ``fp8/``: the callers live in
``compressed_tensors/``, ``quark/``, ``modelslim/``, ``modelopt_quant.py``,
``w8a8_fp8.py`` and ``fp8.py``, so a home under any one of them would be
wrong.
"""

from typing import Literal

import torch
from torch.nn import Parameter

from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.layers.quantization.fp8_utils import normalize_e4m3fn_to_e4m3fnuz
from sglang.srt.layers.quantization.utils import requantize_with_max_scale

__all__ = ["process_fp8_linear_after_loading"]

Granularity = Literal["tensor", "channel"]


def process_fp8_linear_after_loading(
    layer: torch.nn.Module,
    *,
    granularity: Granularity,
    aiter_shuffle: bool = False,
) -> None:
    """Settle an FP8 linear layer's weight and weight scale after loading.

    Reads ``layer.weight`` and ``layer.weight_scale`` and rebinds both, plus
    ``layer.input_scale`` when the ROCm normalization rescales it.

    :param granularity: ``"tensor"`` requantizes a fused module's N per-tensor
        scales down to one max scale, so the kernel can always run per tensor.
        ``"channel"`` takes the loaded scales as they are, already lined up
        one per output channel.
    :param aiter_shuffle: commit the weight in aiter's shuffled ``(N, K)``
        layout instead of transposing it. Only meaningful on ROCm with aiter
        enabled, and the caller owns that decision.

    Deciding whether the layer's activation scale survives is the caller's
    job: a static-input scheme reduces it to its max, a dynamic one drops it,
    and some schemes never register one at all.
    """
    if granularity == "tensor":
        weight_scale, weight = requantize_with_max_scale(
            weight=layer.weight,
            weight_scale=layer.weight_scale,
            logical_widths=layer.logical_widths,
        )
        if is_fp8_fnuz():
            weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight,
                weight_scale=weight_scale,
                input_scale=getattr(layer, "input_scale", None),
            )
            if input_scale is not None:
                layer.input_scale = Parameter(input_scale, requires_grad=False)
    elif granularity == "channel":
        weight = layer.weight
        if is_fp8_fnuz():
            weight, weight_scale, input_scale = normalize_e4m3fn_to_e4m3fnuz(
                weight=weight,
                weight_scale=layer.weight_scale,
                input_scale=getattr(layer, "input_scale", None),
            )
            if input_scale is not None:
                layer.input_scale = Parameter(input_scale, requires_grad=False)
        else:
            weight_scale = layer.weight_scale.data
    else:
        raise ValueError(f"Unknown FP8 weight granularity {granularity!r}")

    if aiter_shuffle:
        from aiter.ops.shuffle import shuffle_weight

        # Keep the weight as (N, K); the shuffled layout is what the kernel
        # indexes, so it must not be transposed on top.
        layer.weight = Parameter(shuffle_weight(weight, (16, 16)), requires_grad=False)
    else:
        layer.weight = Parameter(weight.t(), requires_grad=False)

    # required by torch.compile to be torch.nn.Parameter
    layer.weight_scale = Parameter(weight_scale, requires_grad=False)
