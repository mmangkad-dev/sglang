"""Config-time override declarations for cohere_compass."""

from typing import Any

from sglang.srt.arg_groups.model_override_base import _register_for, resolving_view
from sglang.srt.runtime_context import attn_dp_enabled_of


@_register_for("CohereCompassForConditionalGeneration")
def _cohere_compass_overrides(server_args: Any, hf_config: Any) -> dict:
    # The decoder shards attention and its parallel MLP by the global TP group,
    # while DP attention sizes the KV pool by the attention-TP group.
    if attn_dp_enabled_of(resolving_view(server_args)):
        raise ValueError("CohereCompass does not support DP attention.")
    return {}
