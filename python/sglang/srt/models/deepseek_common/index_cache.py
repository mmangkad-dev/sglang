from __future__ import annotations

from numbers import Integral
from typing import Any, MutableMapping, Optional

import torch

from sglang.srt.utils import ceil_align

_INDEX_CACHE_PATTERN_ATTR = "_sglang_index_cache_pattern"
_NSA_ARCHITECTURES = {
    "DeepseekV3ForCausalLM",
    "DeepseekV32ForCausalLM",
    "GlmMoeDsaForCausalLM",
}

TopkIndices = torch.Tensor | tuple[torch.Tensor, torch.Tensor]


def _get_config_attr(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(name, default)
    return getattr(config, name, default)


def _is_deepseek_nsa_config(config: Any) -> bool:
    architectures = _get_config_attr(config, "architectures")
    index_topk = _get_config_attr(config, "index_topk")
    return (
        architectures is not None
        and architectures[0] in _NSA_ARCHITECTURES
        and index_topk is not None
    )


def resolve_index_cache_pattern(
    config: Any,
    *,
    use_nsa: Optional[bool] = None,
    is_nextn: bool = False,
) -> Optional[str]:
    if is_nextn:
        return None

    if use_nsa is None:
        use_nsa = _is_deepseek_nsa_config(config)
    if not use_nsa:
        return None

    if not isinstance(config, dict):
        cached_pattern = getattr(config, _INDEX_CACHE_PATTERN_ATTR, None)
        if cached_pattern is not None:
            return cached_pattern

    num_hidden_layers = _get_config_attr(config, "num_hidden_layers")
    if num_hidden_layers is None:
        raise ValueError(
            "IndexCache requires `num_hidden_layers` to resolve the layer reuse pattern."
        )

    pattern = _get_config_attr(config, "index_topk_pattern")
    if pattern is not None:
        if not isinstance(pattern, str):
            raise TypeError(
                f"`index_topk_pattern` must be a string, got {type(pattern).__name__}."
            )
        resolved_pattern = pattern.strip().upper()
        if not resolved_pattern:
            raise ValueError("`index_topk_pattern` must not be empty.")
        invalid_chars = sorted(set(resolved_pattern) - {"F", "S"})
        if invalid_chars:
            raise ValueError(
                "`index_topk_pattern` may only contain 'F' and 'S', "
                f"got invalid entries: {invalid_chars}."
            )
        if len(resolved_pattern) != num_hidden_layers:
            raise ValueError(
                "`index_topk_pattern` length must match the number of decoder layers: "
                f"got {len(resolved_pattern)} vs {num_hidden_layers}."
            )
    else:
        index_topk_freq = _get_config_attr(config, "index_topk_freq", 1)
        if isinstance(index_topk_freq, bool) or not isinstance(
            index_topk_freq, Integral
        ):
            raise TypeError(
                "`index_topk_freq` must be a positive integer, "
                f"got {type(index_topk_freq).__name__}."
            )
        index_topk_freq = int(index_topk_freq)
        if index_topk_freq < 1:
            raise ValueError(f"`index_topk_freq` must be >= 1, got {index_topk_freq}.")
        resolved_pattern = "".join(
            "F" if layer_id % index_topk_freq == 0 else "S"
            for layer_id in range(num_hidden_layers)
        )

    if resolved_pattern[0] != "F":
        raise ValueError(
            "The first IndexCache layer must be 'F'; the first DSA layer cannot reuse "
            "top-k indices from a previous layer."
        )

    if not isinstance(config, dict):
        setattr(config, _INDEX_CACHE_PATTERN_ATTR, resolved_pattern)
    return resolved_pattern


def get_index_cache_skip_flags(
    resolved_pattern: Optional[str], layer_id: int
) -> tuple[bool, bool]:
    if resolved_pattern is None:
        return False, False
    if layer_id < 0 or layer_id >= len(resolved_pattern):
        raise ValueError(
            f"Layer id {layer_id} is out of range for IndexCache pattern "
            f"of length {len(resolved_pattern)}."
        )
    skip_topk = resolved_pattern[layer_id] == "S"
    next_skip_topk = (
        layer_id + 1 < len(resolved_pattern) and resolved_pattern[layer_id + 1] == "S"
    )
    return skip_topk, next_skip_topk


def get_index_cache_topk_buffer_width(config: Any) -> Optional[int]:
    if not _is_deepseek_nsa_config(config):
        return None
    resolved_pattern = resolve_index_cache_pattern(config)
    if resolved_pattern is None or "S" not in resolved_pattern:
        return None
    return ceil_align(_get_config_attr(config, "index_topk"), 2048)


def has_reusable_topk_indices(topk_indices: Optional[TopkIndices]) -> bool:
    if topk_indices is None:
        return False
    if isinstance(topk_indices, tuple):
        return any(has_reusable_topk_indices(item) for item in topk_indices)
    return topk_indices.numel() > 0 and bool((topk_indices != -1).any().item())


def extract_topk_indices_from_pp_proxy_tensors(
    pp_proxy_tensors: Any,
) -> Optional[TopkIndices]:
    if pp_proxy_tensors is None:
        return None

    tensors = getattr(pp_proxy_tensors, "tensors", pp_proxy_tensors)
    if tensors is None:
        return None

    if "topk_indices" in tensors:
        return tensors["topk_indices"]

    has_prev = "topk_indices_prev" in tensors
    has_next = "topk_indices_next" in tensors
    if has_prev != has_next:
        raise ValueError(
            "Pipeline proxy tensors must provide both `topk_indices_prev` and "
            "`topk_indices_next` together."
        )
    if has_prev:
        return tensors["topk_indices_prev"], tensors["topk_indices_next"]

    return None


def add_topk_indices_to_pp_proxy_tensors(
    tensors: MutableMapping[str, torch.Tensor],
    topk_indices: Optional[TopkIndices],
) -> None:
    if topk_indices is None:
        return
    if isinstance(topk_indices, tuple):
        tensors["topk_indices_prev"] = topk_indices[0]
        tensors["topk_indices_next"] = topk_indices[1]
    else:
        tensors["topk_indices"] = topk_indices
