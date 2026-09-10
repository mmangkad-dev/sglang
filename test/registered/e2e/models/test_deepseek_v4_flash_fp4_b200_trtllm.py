"""B200 per-commit CI: DeepSeek-V4-Flash FP4 with the trtllm attention backend.

Three recipes over a uniform-FP8 KV pool and trtllm-gen sparse MLA, each
covering a path no other one reaches: TP-only spec decoding (draft-extend and
the multi-step draft backend), DP + DeepEP spec decoding (the DP-padded
draft-extend lengths), and DP + mixed-chunk + breakable prefill graph (the
graph-replay metadata refresh). A fourth non-spec DP recipe was dropped as a
subset of the breakable-graph one, whose topology and assertions it shared.
"""

import concurrent.futures
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.basic_decode_correctness_kit import BasicDecodeCorrectnessMixin
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.kits.spec_decoding_kit import SpecDecodingMixin
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=650, stage="base-c", runner_config="4-gpu-b200")

MODEL = "deepseek-ai/DeepSeek-V4-Flash"
SERVER_LAUNCH_TIMEOUT = 3600
DEEPEP_CONFIG = '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}'

_DEEPEP_ENV = {
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
}

# Long-prompt prefill fixtures. Mixed lengths cover c4/c128 selection and
# VarSeq packing; the longest prompt exceeds the 4096-token prefill chunk, so
# the chunk-crossing and cached-prefix paths run too.
_FILLER_SENTENCES = [
    "The expedition recorded water temperature, salinity, and current speed "
    "at every station along the transect. ",
    "Archival records from the observatory describe decades of nightly "
    "measurements taken with remarkable consistency. ",
    "Each greenhouse module recycles condensate through a gravel bed before "
    "returning it to the irrigation loop. ",
]
_LONG_PROMPT_QUESTION = (
    "\n\nIn one short sentence, what kind of activity do the paragraphs above describe?"
)


def _make_long_prompt(idx: int, target_chars: int) -> str:
    sentence = _FILLER_SENTENCES[idx % len(_FILLER_SENTENCES)]
    body = ""
    n = 0
    while len(body) < target_chars:
        body += f"[Entry {idx}-{n}] " + sentence
        n += 1
    return body + _LONG_PROMPT_QUESTION


# Roughly 2.5k, 4.5k, and 7k tokens.
LONG_PROMPTS = [
    _make_long_prompt(0, 10_000),
    _make_long_prompt(1, 18_000),
    _make_long_prompt(2, 28_000),
]
LONG_MAX_NEW_TOKENS = 32
MIN_PRINTABLE_ASCII_RATIO = 0.85
_REQUEST_TIMEOUT = 600


def _greedy_generate(base_url: str, prompt: str, max_new_tokens: int) -> str:
    resp = requests.post(
        base_url + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=_REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["text"]


def _printable_ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(32 <= ord(c) < 127 or c in "\n\t" for c in text) / len(text)


class LongPromptPrefillMixin:
    """Drive prompts past the prefill chunk bound through varlen prefill.

    The only coverage of chunk-crossing and cached-prefix extend for the
    uniform-FP8 pool; the decode-correctness kit's prompts all fit in one
    chunk. Outputs are sanity-checked rather than matched exactly because
    split-KV reduction order is not bit-reproducible across batch shapes.
    """

    def _assert_sane(self, out: str, what: str) -> None:
        self.assertGreater(len(out.strip()), 0, f"{what}: empty output")
        ratio = _printable_ascii_ratio(out)
        self.assertGreater(
            ratio,
            MIN_PRINTABLE_ASCII_RATIO,
            f"{what}: output looks like gibberish (ascii ratio={ratio:.2f}): {out!r}",
        )

    def test_long_prompt_varlen_prefill(self):
        with concurrent.futures.ThreadPoolExecutor(len(LONG_PROMPTS)) as pool:
            outs = list(
                pool.map(
                    lambda p: _greedy_generate(self.base_url, p, LONG_MAX_NEW_TOKENS),
                    LONG_PROMPTS,
                )
            )
        for i, out in enumerate(outs):
            print(f"[long-prefill] prompt_chars={len(LONG_PROMPTS[i])} out={out!r}")
            self._assert_sane(out, f"concurrent long prompt {i}")

        cached = _greedy_generate(self.base_url, LONG_PROMPTS[-1], LONG_MAX_NEW_TOKENS)
        print(f"[long-prefill] cached-prefix rerun out={cached!r}")
        self._assert_sane(cached, "cached-prefix extend")


class TestDSV4FlashFP4B200Trtllm(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    LongPromptPrefillMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """LowLatency recipe: TP=4, FP4 (mxfp4), EAGLE spec decoding."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 2.8
    bs_1_speed_thres = 220

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--dsv4-attn-backend",
                "trtllm",
                "--tp",
                "4",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "3",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "4",
                "--chunked-prefill-size",
                "4096",
                "--disable-flashinfer-autotune",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4B200BalancedTrtllm(
    SpecDecodingMixin,
    BasicDecodeCorrectnessMixin,
    GSM8KMixin,
    CustomTestCase,
):
    """Balanced recipe: TP=4, DP=4, DeepEP, EAGLE (1-step spec)."""

    gsm8k_accuracy_thres = 0.93
    accept_length_thres = 1.8
    bs_1_speed_thres = 100

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--dsv4-attn-backend",
                "trtllm",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--moe-a2a-backend",
                "deepep",
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-num-steps",
                "1",
                "--speculative-eagle-topk",
                "1",
                "--speculative-num-draft-tokens",
                "2",
                "--deepep-config",
                DEEPEP_CONFIG,
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


class TestDSV4FlashFP4BreakableCudaGraphB200Trtllm(
    BasicDecodeCorrectnessMixin, GSM8KMixin, CustomTestCase
):
    """BCG recipe: TP=4, DP=4, DeepEP, DP attention, mixed chunk."""

    gsm8k_accuracy_thres = 0.93

    @unittest.skip(
        "Flaky: temp-0 outputs are nondeterministic under this recipe "
        "(sparse-DP prefill replays the breakable CUDA graph with a "
        "fabricated idle-rank dummy extend; its hidden states vary run to "
        "run and perturb real tokens' logits through the shared EP grouped "
        "GEMMs at capture buckets 4/16). The cause is upstream of the "
        "attention backend, so the trtllm recipe inherits it. See #31125."
    )
    def test_determinism_temp_zero(self):
        pass

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(MODEL)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=[
                "--trust-remote-code",
                "--dsv4-attn-backend",
                "trtllm",
                "--tp",
                "4",
                "--dp",
                "4",
                "--enable-dp-attention",
                "--enable-mixed-chunk",
                "--cuda-graph-backend-prefill",
                "breakable",
                "--moe-a2a-backend",
                "deepep",
                "--deepep-config",
                DEEPEP_CONFIG,
                "--chunked-prefill-size",
                "4096",
                "--cuda-graph-max-bs-prefill",
                "1024",
                "--mem-fraction-static",
                "0.80",
                "--cuda-graph-max-bs-decode",
                "16",
                "--max-running-requests",
                "128",
                "--watchdog-timeout",
                "900",
            ],
            env=_DEEPEP_ENV,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
