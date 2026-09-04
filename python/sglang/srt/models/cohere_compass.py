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
"""Inference-only CohereCompass (North Micro Vision) model.

Checkpoint: ``CohereLabs/North-Micro-Vision-Instruct``.

The vision tower is a Qwen3-VL native-resolution encoder with DeepStack taps, so
it is reused verbatim. The text decoder is Cohere's Command-A style block:
one mean-centred ``LayerNorm`` feeding attention and MLP in parallel, summed back
into a single residual, with three interleaved sliding-window layers (interleaved
M-RoPE) per global layer, and the global layers carrying no position embedding at
all (NoPE).
"""

from typing import Optional, Tuple, Union

import torch
from torch import nn

from sglang.srt.configs.cohere_compass import (
    CohereCompassConfig,
    CohereCompassTextConfig,
)
from sglang.srt.distributed.parallel_state import get_pp_group
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.dp_attention import is_dp_attention_enabled
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.commandr import LayerNorm as CohereCompassLayerNorm
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import add_prefix, make_layers


class CohereCompassMLP(nn.Module):
    def __init__(
        self,
        config: CohereCompassTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class CohereCompassAttention(nn.Module):
    """Per-layer-type attention: sliding window + interleaved M-RoPE, or global + NoPE."""

    def __init__(
        self,
        config: CohereCompassTextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        tp_size = get_parallel().tp_size
        self.config = config
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        if self.total_num_kv_heads >= tp_size:
            assert self.total_num_kv_heads % tp_size == 0
        else:
            assert tp_size % self.total_num_kv_heads == 0
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.layer_type = config.layer_types[layer_id]
        rope_parameters = config.rope_parameters.get(self.layer_type)
        # ``None`` for the global layers: they run without any position embedding.
        if rope_parameters is None:
            self.rotary_emb = None
        else:
            rope_scaling = {
                key: value
                for key, value in rope_parameters.items()
                if key != "rope_theta"
            }
            self.rotary_emb = get_rope(
                head_size=self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=rope_parameters["rope_theta"],
                rope_scaling=rope_scaling,
                is_neox_style=True,
            )

        self.sliding_window_size = (
            config.sliding_window if self.layer_type == "sliding_attention" else -1
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=self.sliding_window_size,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class CohereCompassDecoderLayer(nn.Module):
    def __init__(
        self,
        config: CohereCompassTextConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.self_attn = CohereCompassAttention(
            config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = CohereCompassMLP(
            config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = CohereCompassLayerNorm(
            param_shape=(config.hidden_size), eps=config.layer_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Cohere's parallel block: attention and MLP read the same normalized
        input, and their outputs are summed into one residual.

        Follows SGLang's deferred-residual convention -- ``hidden_states`` is
        this layer's contribution and ``residual`` the running stream, folded
        together at the start of the next layer -- so the pair matches the
        pipeline-parallel proxy tensors the runtime allocates.
        """
        residual = hidden_states if residual is None else residual + hidden_states
        normed, _ = self.input_layernorm(residual)
        attn_output = self.self_attn(
            positions=positions,
            hidden_states=normed,
            forward_batch=forward_batch,
        )
        return attn_output + self.mlp(normed), residual


class CohereCompassTextModel(nn.Module):
    def __init__(
        self,
        *,
        config: CohereCompassTextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                use_attn_tp_group=is_dp_attention_enabled(),
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: CohereCompassDecoderLayer(
                config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = CohereCompassLayerNorm(
                param_shape=(config.hidden_size), eps=config.layer_norm_eps
            )
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # DeepStack: the visual features tapped at ``deepstack_visual_indexes``
        # of the vision tower are added to decoder layers 0..N-1, in order.
        self.num_deepstack_embeddings = len(
            getattr(config, "deepstack_visual_indexes", None) or []
        )
        self.deepstack_embed_to_decoder_layer = range(self.num_deepstack_embeddings)
        if not self.pp_group.is_first_rank:
            assert self.start_layer >= self.num_deepstack_embeddings, (
                "start_layer should be greater than or equal to the number of "
                "deepstack visual indexes"
            )

        # For EAGLE3 support
        self.layers_to_capture = []
        self.capture_aux_hidden_states = False

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_input_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            hidden_states = (
                self.embed_tokens(input_ids) if input_embeds is None else input_embeds
            )
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        aux_hidden_states = []
        for layer_id in range(self.start_layer, self.end_layer):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual = self.layers[layer_id](
                positions,
                hidden_states,
                forward_batch,
                residual,
            )
            if (
                input_deepstack_embeds is not None
                and layer_id in self.deepstack_embed_to_decoder_layer
            ):
                # ``hidden_states`` is this layer's own contribution and gets
                # folded into ``residual`` next, so adding here lands the
                # DeepStack features on the same running sum HF adds them to.
                sep = self.hidden_size * layer_id
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            hidden_states, _ = self.norm(hidden_states + residual)

        if len(aux_hidden_states) == 0:
            return hidden_states
        return hidden_states, aux_hidden_states

    def set_dflash_layers_to_capture(self, layer_ids: list):
        self.capture_aux_hidden_states = True
        self.layers_to_capture = layer_ids


class CohereCompassForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """North Micro Vision: Qwen3-VL vision tower + Cohere Command-A text decoder."""

    def __init__(
        self,
        config: CohereCompassConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        # The text config needs the vision tower's DeepStack tap count before the
        # language model is built, since it decides which decoder layers get a
        # visual residual added.
        config.text_config.deepstack_visual_indexes = (
            config.vision_config.deepstack_visual_indexes
        )
        super().__init__(
            config,
            quant_config=quant_config,
            prefix=prefix,
            language_model_cls=CohereCompassTextModel,
        )
        # Cohere scales logits before sampling.
        self.logit_scale = config.text_config.logit_scale
        self.logits_processor = LogitsProcessor(
            self.config, logit_scale=self.logit_scale
        )


EntryClass = CohereCompassForConditionalGeneration
