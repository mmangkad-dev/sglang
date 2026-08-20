"""End-to-end GSM8K accuracy test for DSA cache layer split (GLM-5.2).

Layer split shards the DSA GPU KV/indexer cache layers across prefill CP ranks
(``--enable-dsa-cache-layer-split``); non-owner ranks read a layer via an
owner-broadcast into a small remote scratch buffer. It only applies to PD
prefill workers running DSA prefill-CP (a unified server would decode on the
same worker, where non-owner ranks lack the full cache), so this test drives a
PD-disaggregated GLM-5.2 deployment: a layer-split prefill worker running
interleave prefill-CP + layer split, and an ordinary decode worker that receives
full cache shards via PD transfer.

Runs on 8 GPUs split over two nodes (prefill TP=4 + decode TP=4), one worker
per node. The test runs on the prefill node and drives the decode node over
ssh, so only the prefill node needs this file. Each worker is a single-node TP
group, so neither side takes ``--nnodes``/``--dist-init-addr``.

Both nodes are assumed to sit in one MNNVL clique: KV transfer rides mooncake's
NVLink memory pool instead of RDMA, and no ``--disaggregation-ib-device`` is
selected. On an RDMA fabric, drop ``MC_FORCE_MNNVL`` /
``SGLANG_MOONCAKE_CUSTOM_MEM_POOL`` from ``WORKER_ENV`` and pass the IB devices
instead.

The cluster addresses are deployment-specific, so they come from the
environment and the test skips when they are unset:

    export GLM52_PD_PREFILL_HOST=<address the decode node dials this node on>
    export GLM52_PD_DECODE_HOST=<address this node dials the decode node on>
    export GLM52_PD_DECODE_SSH=<ssh target for the decode node>
    python3 test/registered/models_e2e/test_dsa_glm52_pd_mtp_cp_layersplit.py

Not CI-registered: a two-host deployment can't be scheduled onto a single
runner.
"""

import os
import shlex
import subprocess
import time
import unittest

import requests

from sglang.test.kits.eval_accuracy_kit import GSM8KMixin
from sglang.test.server_fixtures.disaggregation_fixture import (
    PDDisaggregationServerBase,
)
from sglang.test.test_utils import start_subprocess_fail_fast_watcher

# Deployment-specific, so there is no sensible default; unset means skip.
PREFILL_NODE_HOST = os.environ.get("GLM52_PD_PREFILL_HOST")
DECODE_NODE_HOST = os.environ.get("GLM52_PD_DECODE_HOST")
DECODE_NODE_SSH = os.environ.get("GLM52_PD_DECODE_SSH")

# Pinned rather than derived from CUDA_VISIBLE_DEVICES like the base fixture
# does: the two nodes have to agree on them. Override if they collide with
# anything already listening on either node.
LB_PORT = os.environ.get("GLM52_PD_LB_PORT", "21000")
PREFILL_PORT = os.environ.get("GLM52_PD_PREFILL_PORT", "21100")
DECODE_PORT = os.environ.get("GLM52_PD_DECODE_PORT", "21200")
PREFILL_NCCL_PORT = os.environ.get("GLM52_PD_PREFILL_NCCL_PORT", "21300")
DECODE_NCCL_PORT = os.environ.get("GLM52_PD_DECODE_NCCL_PORT", "21400")
BOOTSTRAP_PORT = os.environ.get("GLM52_PD_BOOTSTRAP_PORT", "21500")

# NVFP4 weight load plus warmup runs well past the 600s sglang default, and
# rank 0 additionally waits out rank 1's startup.
LAUNCH_TIMEOUT = float(os.environ.get("GLM52_PD_LAUNCH_TIMEOUT", "3600"))

REMOTE_LOG = "/tmp/glm52_pd_decode.log"
REMOTE_PID_FILE = "/tmp/glm52_pd_decode.pid"
REMOTE_DONE_FILE = "/tmp/glm52_pd_decode.done"

WORKER_ENV = {
    # Route mooncake KV transfer over the clique's multi-node NVLink instead of
    # RDMA: register an NVLink memory pool and keep the transfer on MNNVL.
    "SGLANG_MOONCAKE_CUSTOM_MEM_POOL": "NVLINK",
    "MC_FORCE_MNNVL": "True",
}
# ssh does not carry the caller's environment, so the decode node only sees the
# model cache root if it is forwarded explicitly.
if "HF_HOME" in os.environ:
    WORKER_ENV["HF_HOME"] = os.environ["HF_HOME"]


def _ssh(command: str, check: bool = True) -> str:
    result = subprocess.run(
        ["ssh", DECODE_NODE_SSH, command],
        capture_output=True,
        text=True,
        check=False,
    )
    if check and result.returncode != 0:
        raise RuntimeError(
            f"ssh {DECODE_NODE_SSH} {command!r} failed ({result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result.stdout.strip()


class TestGLM52DSACacheLayerSplit(PDDisaggregationServerBase, GSM8KMixin):
    model = "nvidia/GLM-5.2-NVFP4"

    # Full GSM8K test set (1319 questions) with a tight accuracy floor.
    gsm8k_accuracy_thres = 0.935
    gsm8k_num_questions = 1319
    gsm8k_num_threads = 200
    gsm8k_num_shots = 20

    extra_prefill_env = WORKER_ENV
    extra_decode_env = WORKER_ENV

    # rank 1 owns its own 4 GPUs, so decode starts at local GPU 0.
    decode_base_gpu_id = 0
    _decode_remote_pgid = None

    # Prefill worker: interleave prefill-CP + DSA cache layer split on 4 GPUs
    # (TP=4 -> attn_cp_size=4, so KV/indexer layers shard 4-way across CP ranks).
    extra_prefill_args = [
        "--tp",
        "4",
        "--attn-cp-size",
        "4",
        "--dsa-prefill-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--enable-dsa-cache-layer-split",
        "--enable-prefill-cp",
        "--cp-strategy",
        "interleave",
        "--mem-fraction-static",
        "0.85",
        "--chunked-prefill-size",
        "4096",
        "--max-prefill-tokens",
        "4096",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]
    # Decode worker: ordinary local decode cache, receives full shards via PD
    # transfer.
    extra_decode_args = [
        "--tp",
        "4",
        "--dsa-decode-backend",
        "trtllm",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.85",
        "--speculative-algorithm",
        "EAGLE",
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "6",
    ]

    @classmethod
    def _configure_cluster(cls):
        # The base fixture puts both workers on 127.0.0.1; here decode has to
        # reach the prefill bootstrap server across the cluster network.
        cls.lb_port = LB_PORT
        cls.prefill_port = PREFILL_PORT
        cls.decode_port = DECODE_PORT
        cls.prefill_nccl_port = PREFILL_NCCL_PORT
        cls.decode_nccl_port = DECODE_NCCL_PORT
        cls.bootstrap_port = BOOTSTRAP_PORT
        cls.base_host = PREFILL_NODE_HOST
        cls.prefill_url = f"http://{PREFILL_NODE_HOST}:{cls.prefill_port}"
        cls.decode_url = f"http://{DECODE_NODE_HOST}:{cls.decode_port}"
        cls.lb_url = f"http://{PREFILL_NODE_HOST}:{cls.lb_port}"
        cls.base_url = cls.lb_url

        # MNNVL, not RDMA, so no --disaggregation-ib-device. Layer split only
        # supports the mooncake transfer backend.
        cls.transfer_backend = ["--disaggregation-transfer-backend", "mooncake"]
        cls.rdma_devices = []
        print(f"{cls.prefill_url=} {cls.decode_url=} {cls.lb_url=}")

    @classmethod
    def setUpClass(cls):
        missing = [
            name
            for name, value in (
                ("GLM52_PD_PREFILL_HOST", PREFILL_NODE_HOST),
                ("GLM52_PD_DECODE_HOST", DECODE_NODE_HOST),
                ("GLM52_PD_DECODE_SSH", DECODE_NODE_SSH),
            )
            if not value
        ]
        if missing:
            raise unittest.SkipTest(
                f"two-node PD test requires {', '.join(missing)}; see the module "
                "docstring"
            )
        super().setUpClass()
        cls._configure_cluster()
        cls.launch_all()

    @classmethod
    def start_decode(cls):
        decode_args = [
            "--model-path",
            cls.model,
            "--trust-remote-code",
            "--disaggregation-mode",
            "decode",
            "--disaggregation-bootstrap-port",
            cls.bootstrap_port,
            "--nccl-port",
            cls.decode_nccl_port,
            "--tp",
            str(cls.decode_tp_size),
            "--base-gpu-id",
            str(cls.decode_base_gpu_id),
            *cls.extra_decode_args,
            *cls.transfer_backend,
            *cls.rdma_devices,
            "--host",
            DECODE_NODE_HOST,
            "--port",
            cls.decode_port,
        ]
        env_prefix = " ".join(
            f"{k}={shlex.quote(v)}" for k, v in sorted(cls.extra_decode_env.items())
        )
        launch = f"{env_prefix} python3 -m sglang.launch_server " + " ".join(
            shlex.quote(a) for a in decode_args
        )
        # Detach into its own session so a dropped ssh connection can't kill the
        # worker; $$ is the session leader, i.e. the pgid teardown signals. The
        # supervising shell deliberately does not exec, so it outlives the
        # server and records the exit code the readiness wait fails fast on.
        supervise = (
            f"echo $$ > {REMOTE_PID_FILE}; "
            f"{{ {launch}; }} > {REMOTE_LOG} 2>&1; "
            f"echo $? > {REMOTE_DONE_FILE}"
        )
        _ssh(
            f"rm -f {REMOTE_LOG} {REMOTE_PID_FILE} {REMOTE_DONE_FILE}; "
            f"setsid bash -c {shlex.quote(supervise)} </dev/null >/dev/null 2>&1 &"
        )
        for _ in range(30):
            pgid = _ssh(f"cat {REMOTE_PID_FILE} 2>/dev/null", check=False)
            if pgid:
                cls._decode_remote_pgid = int(pgid)
                break
            time.sleep(1)
        else:
            raise RuntimeError(
                f"decode worker on {DECODE_NODE_SSH} never wrote {REMOTE_PID_FILE}"
            )
        print(
            f"Launched decode worker on {DECODE_NODE_SSH} "
            f"(pgid={cls._decode_remote_pgid}, log {REMOTE_LOG})"
        )

    @classmethod
    def wait_remote_decode_ready(cls, timeout=LAUNCH_TIMEOUT):
        deadline = time.time() + timeout
        while time.time() < deadline:
            exit_code = _ssh(f"cat {REMOTE_DONE_FILE} 2>/dev/null", check=False)
            if exit_code:
                log_tail = _ssh(f"tail -n 50 {REMOTE_LOG}", check=False)
                raise RuntimeError(
                    f"decode worker on {DECODE_NODE_SSH} exited with {exit_code} "
                    f"before becoming ready:\n{log_tail}"
                )
            try:
                if (
                    requests.get(f"{cls.decode_url}/health", timeout=10).status_code
                    == 200
                ):
                    print(f"Server {cls.decode_url} is ready")
                    return
            except requests.RequestException:
                pass
            time.sleep(10)
        raise TimeoutError(
            f"decode worker on {DECODE_NODE_SSH} not ready after {timeout}s; "
            f"see {REMOTE_LOG} on that node"
        )

    @classmethod
    def launch_all(cls):
        cls.start_decode()
        cls.start_prefill()
        cls.wait_server_ready(
            cls.prefill_url + "/health",
            timeout=LAUNCH_TIMEOUT,
            process=cls.process_prefill,
        )
        cls.wait_remote_decode_ready()
        cls.launch_lb()
        # process_decode lives on rank 1, so only the local pair is watched.
        cls._fail_fast_stop = start_subprocess_fail_fast_watcher(
            [("prefill", cls.process_prefill), ("lb", cls.process_lb)]
        )

    @classmethod
    def stop_remote_decode(cls):
        if cls._decode_remote_pgid is None:
            return
        pgid = cls._decode_remote_pgid
        cls._decode_remote_pgid = None
        _ssh(f"kill -TERM -{pgid} 2>/dev/null; true", check=False)
        for _ in range(60):
            if not _ssh(f"kill -0 -{pgid} 2>/dev/null && echo alive", check=False):
                break
            time.sleep(1)
        else:
            _ssh(f"kill -KILL -{pgid} 2>/dev/null; true", check=False)

    @classmethod
    def tearDownClass(cls):
        try:
            cls.stop_remote_decode()
        finally:
            super().tearDownClass()


if __name__ == "__main__":
    unittest.main()
