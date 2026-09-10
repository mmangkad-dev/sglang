import unittest

import torch

from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4UniformFP8KVPool
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

_DIM = 512
_ROWS = 1024


def _store_into_blank_pool(loc):
    """Run the uniform-FP8 store over a zeroed pool; return it as uint8."""
    pool = DeepSeekV4UniformFP8KVPool.__new__(DeepSeekV4UniformFP8KVPool)
    pool.kv_cache_total_dim = _DIM
    pool.kv_buffer = [torch.zeros(_ROWS, _DIM, dtype=torch.float8_e4m3fn)]
    cache_k = torch.full((loc.shape[0], _DIM), 2.0)
    pool.set_key_buffer_fused(layer_id=0, loc=loc, cache_k=cache_k)
    return pool.kv_buffer[0].view(torch.uint8)


class TestUniformFP8StoreSentinel(CustomTestCase):
    """A negative KV location must not land on a live row of the pool.

    BUG REGRESSION. `fused_k_norm_rope_flashmla` skips rows whose out_loc is
    negative -- the -1 sentinel the full->SWA translation emits for
    out-of-window tokens and padded rows, and DSpark's uncommitted draft
    slots. The uniform-FP8 setter indexed with those values directly, so -1
    wrapped to the last row of the flattened pool and overwrote whatever token
    occupied it. Negative locations now land on the reserved dummy slot 0,
    which the allocators never hand out.
    """

    def test_sentinel_does_not_wrap_to_the_end_of_the_pool(self):
        buf = _store_into_blank_pool(torch.tensor([-1], dtype=torch.int32))
        self.assertTrue(
            bool((buf[_ROWS - 1] == 0).all()),
            "a -1 location wrote to the last pool row",
        )
        self.assertTrue(bool((buf[1:] == 0).all()), "only slot 0 may absorb it")

    def test_valid_locations_still_land_where_asked(self):
        """An over-eager clamp here would redirect every real store to slot 0."""
        buf = _store_into_blank_pool(torch.tensor([7, 300], dtype=torch.int32))
        for row in (7, 300):
            self.assertTrue(bool((buf[row] != 0).all()), f"row {row} not written")
        untouched = [r for r in (1, 6, 8, 299, 301, _ROWS - 1)]
        for row in untouched:
            self.assertTrue(bool((buf[row] == 0).all()), f"row {row} clobbered")

    def test_mixed_batch_keeps_the_valid_rows(self):
        """Padded decode rows arrive interleaved with real ones, not alone."""
        buf = _store_into_blank_pool(torch.tensor([5, -1, 9], dtype=torch.int32))
        for row in (5, 9):
            self.assertTrue(bool((buf[row] != 0).all()), f"row {row} not written")
        self.assertTrue(bool((buf[_ROWS - 1] == 0).all()))

    def test_store_stays_free_of_host_synchronization(self):
        """The store runs per layer per forward and inside captured graphs.

        A `.item()` / `bool()` on the locations to filter sentinels would be a
        device sync on the decode hot path and would break graph capture.
        """
        loc = torch.tensor([3, -1], dtype=torch.int32)
        calls = []
        original = torch.Tensor.item

        def tracking_item(self):
            calls.append(True)
            return original(self)

        torch.Tensor.item = tracking_item
        try:
            _store_into_blank_pool(loc)
        finally:
            torch.Tensor.item = original
        self.assertEqual(calls, [], "the store pulled a value to the host")


if __name__ == "__main__":
    unittest.main()
