"""CPU regression tests for the CohereCompass (North Micro Vision) invariants.

Two properties of the model are not expressible in its config and would fail
silently if broken, so they are pinned here:

* the vision tower must interpolate its position embeddings corner-aligned, or
  enabling the ViT CUDA graph changes the embeddings instead of only changing
  how they are computed, and
* the decoder's DeepStack tap count must match the vision tower's, or visual
  residuals land on the wrong layers.
"""

import os
import socket

import numpy as np
import pytest
import torch

from sglang.srt.configs.cohere_compass import (
    CohereCompassConfig,
    CohereCompassVisionConfig,
)
from sglang.srt.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from sglang.srt.models.cohere_compass import (
    CohereCompassForConditionalGeneration,
    CohereCompassTextModel,
)
from sglang.srt.models.qwen3_vl import Qwen3VLMoeVisionModel
from sglang.srt.runtime_context import get_context, get_exec, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=25, suite="base-a-test-cpu")

# A non-square grid whose sides are both below num_grid_per_side, so the
# position grid is genuinely resampled; a native-resolution grid interpolates
# to itself and would hide an alignment mismatch.
GRID_THW = [[1, 6, 4]]
DEEPSTACK_INDEXES = [0, 1]


def _vision_config(**overrides):
    kwargs = dict(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=2,
        out_hidden_size=64,
        num_position_embeddings=64,
        deepstack_visual_indexes=list(DEEPSTACK_INDEXES),
    )
    kwargs.update(overrides)
    return kwargs


def _text_config(**overrides):
    kwargs = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        eos_token_id=511,
        max_position_embeddings=256,
        sliding_window=8,
        layer_types=["sliding_attention"] * 3 + ["full_attention"],
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "full_attention": None,
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 50000,
                "mrope_interleaved": True,
                "mrope_section": [4, 2, 2],
            },
        },
    )
    kwargs.update(overrides)
    return kwargs


@pytest.fixture(scope="module")
def gloo_world():
    """a one-rank cpu process group, so the parallel layers can be built"""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    init_distributed_environment(
        world_size=1,
        rank=0,
        local_rank=0,
        distributed_init_method=f"tcp://127.0.0.1:{port}",
        backend="gloo",
    )
    initialize_model_parallel(tensor_model_parallel_size=1, backend="gloo")
    yield
    destroy_model_parallel()
    destroy_distributed_environment()


@pytest.fixture
def single_rank(gloo_world):
    with (
        get_parallel().override(tp_size=1, tp_rank=0, attn_tp_size=1, attn_tp_rank=0),
        get_context().override_server_args(),
    ):
        yield


def _build_model():
    return CohereCompassForConditionalGeneration(
        CohereCompassConfig(text_config=_text_config(), vision_config=_vision_config())
    )


def _build_tower():
    tower = Qwen3VLMoeVisionModel(CohereCompassVisionConfig(**_vision_config()))
    torch.nn.init.normal_(tower.pos_embed.weight, std=1.0)
    return tower


def test_model_pins_corner_aligned_interpolation(single_rank):
    """The tower must be corner-aligned regardless of the global flag.

    ``enable_precise_embedding_interpolation`` defaults off, and the tower reads
    it in its constructor, so without the model-owned override the ViT
    CUDA-graph path would resample the position grid differently from eager.
    """
    assert not get_exec().kernel.enable_precise_embedding_interpolation
    model = _build_model()
    assert model.visual.align_corners is True


def test_graph_interpolation_indices_match_the_eager_path(single_rank):
    """The graph path's resampling indices must equal eager's hardcoded linspace."""
    tower = _build_tower()
    CohereCompassForConditionalGeneration._pin_vision_interpolation(tower)

    side = tower.num_grid_per_side
    for dim in (GRID_THW[0][1], GRID_THW[0][2]):
        np.testing.assert_array_equal(
            tower._get_interpolation_indices(dim),
            np.linspace(0, side - 1, dim, dtype=np.float32),
        )


def test_graph_and_eager_position_embeddings_agree(single_rank):
    """Pinned, the two paths agree; unpinned, the ViT graph changes the result."""
    tower = _build_tower()
    grid = torch.tensor(GRID_THW, dtype=torch.int32)

    eager = tower.fast_pos_embed_interpolate_from_list(GRID_THW)
    scale = eager.abs().max()

    CohereCompassForConditionalGeneration._pin_vision_interpolation(tower)
    pinned = tower.fast_pos_embed_interpolate(grid)
    # Not bit-identical: the eager path factors the bilinear weights as
    # ``w00 = 1 - dh - w01``, which only reassociates the same arithmetic.
    assert ((eager - pinned).abs().max() / scale).item() < 1e-5

    tower.align_corners = False
    unpinned = tower.fast_pos_embed_interpolate(grid)
    assert ((eager - unpinned).abs().max() / scale).item() > 1e-2


def test_deepstack_tap_count_is_joined_across_configs(single_rank):
    model = _build_model()
    assert model.num_deepstack_embeddings == len(DEEPSTACK_INDEXES)
    assert model.model.num_deepstack_embeddings == len(DEEPSTACK_INDEXES)
    assert list(model.model.deepstack_embed_to_decoder_layer) == list(
        range(len(DEEPSTACK_INDEXES))
    )


def test_decoder_rejects_an_unjoined_text_config(single_rank):
    """A text config built without the vision join must not silently lose DeepStack."""
    from sglang.srt.configs.cohere_compass import CohereCompassTextConfig

    config = CohereCompassTextConfig(**_text_config())
    assert not hasattr(config, "deepstack_visual_indexes")
    with pytest.raises(AttributeError, match="deepstack_visual_indexes"):
        CohereCompassTextModel(config=config)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
