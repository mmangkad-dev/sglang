# Copyright 2026 Cohere Inc.
# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Config and processor classes for CohereCompass (North Micro Vision).

The checkpoint (``CohereLabs/North-Micro-Vision-Instruct``) declares
``model_type: cohere_compass``, which only Transformers >= 5.16 knows about.
SGLang pins an older Transformers, so the config / processor classes are
defined and registered here instead.

Architecturally CohereCompass is:

* a Qwen3-VL native-resolution vision tower (identical config fields, DeepStack
  taps, ``spatial_merge_size`` patch merger), and
* a Cohere2/Command-A style text decoder: parallel attention+MLP residual, a
  mean-centred ``LayerNorm`` without bias, interleaved sliding-window layers
  carrying interleaved M-RoPE and full-attention layers carrying *no* position
  embedding at all (NoPE).

so the processors are thin subclasses of the Qwen2-VL / Qwen3-VL ones, and only
exist so the ``*_processor_type`` strings in the checkpoint resolve.
"""

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoProcessor,
    AutoVideoProcessor,
    PretrainedConfig,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

try:
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import (
        Qwen2VLImageProcessorPil,
    )
except ImportError:  # pragma: no cover - older Transformers layouts
    Qwen2VLImageProcessorPil = None


class CohereCompassVisionConfig(PretrainedConfig):
    """Native-resolution vision tower. Field-for-field a Qwen3-VL vision config."""

    model_type = "cohere_compass_vision"
    base_config_key = "vision_config"

    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=2048,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes


class CohereCompassTextConfig(PretrainedConfig):
    """North Micro LLM: Command-A style decoder with interleaved SWA / NoPE layers.

    ``rope_parameters`` is keyed by layer type: ``sliding_attention`` carries the
    interleaved M-RoPE parameters, and ``full_attention`` is ``None``, meaning
    those layers run without any positional embedding.
    """

    model_type = "cohere_compass_text"
    base_config_key = "text_config"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=262144,
        hidden_size=2048,
        intermediate_size=6144,
        logit_scale=0.25,
        num_hidden_layers=28,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=500000,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=255001,
        tie_word_embeddings=True,
        rope_parameters=None,
        attention_bias=False,
        attention_dropout=0.0,
        sliding_window=4096,
        layer_types=None,
        pooling=None,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.logit_scale = logit_scale
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = (
            num_attention_heads if num_key_value_heads is None else num_key_value_heads
        )
        self.head_dim = head_dim or hidden_size // num_attention_heads
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.sliding_window = sliding_window
        self.pooling = pooling

        self.layer_types = layer_types or ["full_attention"] * num_hidden_layers
        normalized_rope_parameters = _normalize_rope_parameters(rope_parameters)

        # Interleaved sliding-window / full-attention layers, so SGLang can run
        # the two KV pools separately. ``1`` marks a sliding-window layer, which
        # is what the generic hybrid-SWA fallback in ``ModelConfig`` expects.
        self.is_hybrid_swa = "sliding_attention" in self.layer_types
        self.hybrid_layer_pattern = [
            1 if layer_type == "sliding_attention" else 0
            for layer_type in self.layer_types
        ]

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Assigned after ``super().__init__`` so Transformers' single-layer-type
        # RoPE validator does not warn about the per-layer-type entries.
        self.rope_parameters = normalized_rope_parameters


def _normalize_rope_parameters(rope_parameters):
    """Fill in defaults and hoist the M-RoPE keys to the top level.

    The checkpoint nests per-layer-type RoPE parameters, but SGLang's generic
    "is this an mrope model?" probe (``ModelConfig.model_is_mrope``) looks for
    ``mrope_section`` directly on ``rope_parameters``. Hoisting the sliding-window
    layers' M-RoPE keys keeps that probe working; the model code always reads the
    per-layer-type entries, so the hoisted copies are informational only.
    """
    rope_parameters = dict(rope_parameters or {})
    rope_parameters.setdefault("rope_type", "default")
    rope_parameters.setdefault("rope_theta", 10000.0)
    # ``full_attention: None`` means NoPE, and must survive as an explicit None.
    rope_parameters.setdefault("full_attention", None)
    rope_parameters.setdefault(
        "sliding_attention",
        {
            "rope_type": "default",
            "rope_theta": 50000,
            "mrope_interleaved": True,
            "mrope_section": [24, 20, 20],
        },
    )

    swa_rope = rope_parameters.get("sliding_attention") or {}
    if "mrope_section" in swa_rope:
        rope_parameters.setdefault("mrope_section", swa_rope["mrope_section"])
        rope_parameters.setdefault(
            "mrope_interleaved", swa_rope.get("mrope_interleaved", False)
        )
    return rope_parameters


class CohereCompassConfig(PretrainedConfig):
    model_type = "cohere_compass"
    sub_configs = {
        "vision_config": CohereCompassVisionConfig,
        "text_config": CohereCompassTextConfig,
    }
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        text_config=None,
        vision_config=None,
        image_token_id=255031,
        video_token_id=255032,
        vision_start_token_id=255028,
        vision_end_token_id=255029,
        tie_word_embeddings=True,
        **kwargs,
    ):
        if isinstance(vision_config, dict):
            self.vision_config = self.sub_configs["vision_config"](**vision_config)
        elif vision_config is None:
            self.vision_config = self.sub_configs["vision_config"]()
        else:
            self.vision_config = vision_config

        if isinstance(text_config, dict):
            self.text_config = self.sub_configs["text_config"](**text_config)
        elif text_config is None:
            self.text_config = self.sub_configs["text_config"]()
        else:
            self.text_config = text_config

        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.vision_start_token_id = vision_start_token_id
        self.vision_end_token_id = vision_end_token_id
        # ``fusion_config`` is metadata about how the projector was trained; it
        # carries no runtime knobs, so it is kept only via ``kwargs``.
        super().__init__(**kwargs, tie_word_embeddings=tie_word_embeddings)


class CohereCompassImageProcessor(Qwen2VLImageProcessor):
    """Only exists so ``image_processor_type`` in the checkpoint resolves."""


class CohereCompassVideoProcessor(Qwen3VLVideoProcessor):
    """Only exists so ``video_processor_type`` in the checkpoint resolves."""


class CohereCompassProcessor(Qwen3VLProcessor):
    """Qwen3-VL processing with Cohere's upper-case vision placeholder tokens."""

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            chat_template=chat_template,
            **kwargs,
        )
        self.image_token = "<|IMAGE_PAD|>"
        self.video_token = "<|VIDEO_PAD|>"
        self.vision_start_token = "<|VISION_START|>"
        self.vision_end_token = "<|VISION_END|>"
        if tokenizer is not None:
            self.image_token_id = tokenizer.convert_tokens_to_ids(self.image_token)
            self.video_token_id = tokenizer.convert_tokens_to_ids(self.video_token)
            self.vision_start_token_id = tokenizer.convert_tokens_to_ids(
                self.vision_start_token
            )
            self.vision_end_token_id = tokenizer.convert_tokens_to_ids(
                self.vision_end_token
            )


AutoConfig.register("cohere_compass", CohereCompassConfig, exist_ok=True)
AutoConfig.register("cohere_compass_text", CohereCompassTextConfig, exist_ok=True)
AutoConfig.register("cohere_compass_vision", CohereCompassVisionConfig, exist_ok=True)

_image_processor_classes = {"torchvision": CohereCompassImageProcessor}
if Qwen2VLImageProcessorPil is not None:

    class CohereCompassImageProcessorPil(Qwen2VLImageProcessorPil):
        """PIL-backend counterpart, selected by ``use_fast=False``."""

    _image_processor_classes["pil"] = CohereCompassImageProcessorPil

AutoImageProcessor.register(
    CohereCompassConfig,
    image_processor_classes=_image_processor_classes,
    exist_ok=True,
)
AutoVideoProcessor.register(
    CohereCompassConfig, CohereCompassVideoProcessor, exist_ok=True
)
AutoProcessor.register(CohereCompassConfig, CohereCompassProcessor, exist_ok=True)
