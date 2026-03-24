import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.models.deepseek_common.index_cache import (
    add_topk_indices_to_pp_proxy_tensors,
    extract_topk_indices_from_pp_proxy_tensors,
    get_index_cache_skip_flags,
    get_index_cache_topk_buffer_width,
    has_reusable_topk_indices,
    resolve_index_cache_pattern,
)


def _make_nsa_config(**overrides):
    values = dict(
        architectures=["DeepseekV32ForCausalLM"],
        num_hidden_layers=6,
        index_topk=2048,
        index_topk_freq=1,
        index_topk_pattern=None,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


class TestIndexCache(unittest.TestCase):
    def test_resolve_index_cache_pattern_from_frequency(self):
        config = _make_nsa_config(index_topk_freq=2)
        pattern = resolve_index_cache_pattern(config)

        self.assertEqual(pattern, "FSFSFS")
        self.assertEqual(get_index_cache_skip_flags(pattern, 0), (False, True))
        self.assertEqual(get_index_cache_skip_flags(pattern, 1), (True, False))
        self.assertEqual(get_index_cache_skip_flags(pattern, 4), (False, True))
        self.assertEqual(get_index_cache_skip_flags(pattern, 5), (True, False))

    def test_custom_pattern_overrides_frequency(self):
        config = _make_nsa_config(
            num_hidden_layers=4,
            index_topk_freq=99,
            index_topk_pattern="fsSf",
        )
        pattern = resolve_index_cache_pattern(config)

        self.assertEqual(pattern, "FSSF")
        self.assertEqual(get_index_cache_skip_flags(pattern, 1), (True, True))
        self.assertEqual(get_index_cache_skip_flags(pattern, 2), (True, False))

    def test_pattern_must_start_with_full_layer(self):
        config = _make_nsa_config(index_topk_pattern="SFFFFF")

        with self.assertRaisesRegex(ValueError, "must be 'F'"):
            resolve_index_cache_pattern(config)

    def test_topk_buffer_width_only_needed_when_sharing(self):
        self.assertIsNone(
            get_index_cache_topk_buffer_width(_make_nsa_config(index_topk_freq=1))
        )
        self.assertEqual(
            get_index_cache_topk_buffer_width(_make_nsa_config(index_topk_freq=4)),
            2048,
        )

    def test_non_target_architectures_do_not_activate_index_cache(self):
        for arch in [
            "DeepseekV3ForCausalLMNextN",
            "MistralLarge3ForCausalLM",
            "PixtralForConditionalGeneration",
        ]:
            with self.subTest(architecture=arch):
                self.assertIsNone(
                    resolve_index_cache_pattern(
                        _make_nsa_config(architectures=[arch], index_topk_freq=4)
                    )
                )
                self.assertIsNone(
                    get_index_cache_topk_buffer_width(
                        _make_nsa_config(architectures=[arch], index_topk_freq=4)
                    )
                )

    def test_pipeline_proxy_helpers_round_trip(self):
        tensor_topk = torch.tensor([[0, -1], [3, 1]], dtype=torch.int64)
        tensor_proxy = {}
        add_topk_indices_to_pp_proxy_tensors(tensor_proxy, tensor_topk)
        self.assertTrue(
            torch.equal(
                extract_topk_indices_from_pp_proxy_tensors(tensor_proxy), tensor_topk
            )
        )

        tuple_topk = (
            torch.tensor([[0, -1]], dtype=torch.int64),
            torch.tensor([[2, 1]], dtype=torch.int64),
        )
        tuple_proxy = {}
        add_topk_indices_to_pp_proxy_tensors(tuple_proxy, tuple_topk)
        extracted_tuple = extract_topk_indices_from_pp_proxy_tensors(tuple_proxy)
        self.assertIsInstance(extracted_tuple, tuple)
        self.assertTrue(torch.equal(extracted_tuple[0], tuple_topk[0]))
        self.assertTrue(torch.equal(extracted_tuple[1], tuple_topk[1]))

    def test_has_reusable_topk_indices(self):
        self.assertFalse(has_reusable_topk_indices(None))
        self.assertFalse(
            has_reusable_topk_indices(torch.full((2, 4), -1, dtype=torch.int64))
        )
        self.assertTrue(
            has_reusable_topk_indices(torch.tensor([[0, -1, -1]], dtype=torch.int64))
        )

    def test_has_reusable_topk_indices_skips_content_check_during_capture(self):
        with mock.patch(
            "sglang.srt.models.deepseek_common.index_cache._is_graph_capture_mode",
            return_value=True,
        ):
            self.assertTrue(
                has_reusable_topk_indices(torch.full((2, 4), -1, dtype=torch.int64))
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
