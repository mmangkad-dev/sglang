import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.kits.spec_server_kits import SpecLogprobKit
from sglang.test.server_fixtures.ngram_fixture import NgramServerBase

# Per-commit: Paged backend only.
# - FA3 base test archived to test/manual/spec/test_spec_ngram_fa3.py
# - Triton + Flashinfer moved to test_spec_ngram_extra.py
register_cuda_ci(est_time=82, stage="base-b", runner_config="1-gpu-small")


class TestNgramSpeculativeDecodingPaged(
    NgramServerBase,
    GSM8KMixin,
    SpecLogprobKit,
    RegexConstrainedMixin,
    JSONConstrainedMixin,
):
    # Constrained mixins reuse this server; they cover the grammar verify path,
    # where the bitmask is built by walking the host draft tree.
    attention_backend = "flashinfer"
    # 0.7, not the fixture's 0.8: on a 32 GB card 0.8 leaves too little for
    # NCCL and activations after the KV pool and CUDA graphs.
    extra_args = ["--page-size", "64", "--mem-fraction-static", "0.7"]


if __name__ == "__main__":
    unittest.main()
