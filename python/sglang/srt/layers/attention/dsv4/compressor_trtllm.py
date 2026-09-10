"""Unfused compressor store for the trtllm backend's uniform-FP8 KV pool.

The FlashMLA epilogue writes a different packed layout. Keep this pipeline
separate until the fused uniform-FP8 store in PR #32975 replaces it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from sglang.srt.layers.attention.dsv4.compressor import Compressor
    from sglang.srt.layers.attention.dsv4.compressor_v2 import CompressorBackendMixin
    from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool


def forward_compress_uniform_fp8(
    backend: CompressorBackendMixin,
    *,
    token_to_kv_pool: DeepSeekV4TokenToKVPool,
    kv_score_input: torch.Tensor,
    state_pool,
    compressor: Compressor,
    layer_id: int,
) -> None:
    """Compress, normalize, apply RoPE, and store as uniform e4m3."""
    from sglang.srt.layers.attention.dsv4.compressor_v2 import (
        _use_online_compress,
        prepare_unfused_compress_store,
    )

    assert not compressor.is_in_indexer
    assert compressor.head_dim == 512, f"{compressor.head_dim=}"
    assert not _use_online_compress(compressor.ratio), (
        "SGLANG_OPT_USE_ONLINE_COMPRESS is not supported with the "
        "uniform-FP8 KV layout yet."
    )

    prepared = prepare_unfused_compress_store(
        backend=backend,
        kv_score_input=kv_score_input,
        state_pool=state_pool,
        compressor=compressor,
    )
    if prepared is None:
        return
    kv_compressed, out_loc_to_store = prepared

    token_to_kv_pool.set_extra_key_buffer_fused(
        layer_id=layer_id,
        loc=out_loc_to_store,
        cache_k=kv_compressed,
    )
