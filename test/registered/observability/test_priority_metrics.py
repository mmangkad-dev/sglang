import sys
import unittest
from types import SimpleNamespace
from typing import Dict, List

import requests
from prometheus_client import CollectorRegistry, Gauge
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample

from sglang.srt.observability.label_transform import UNKNOWN_PRIORITY_VALUE
from sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerMetricsCollector,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(
    est_time=120,
    stage="base-b",
    runner_config="1-gpu-small",
)
register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd")
register_cpu_ci(est_time=240, suite="base-c-test-cpu")

_MODEL_NAME = "Qwen/Qwen3-0.6B"


def _parse_prometheus_metrics(metrics_text: str) -> Dict[str, List[Sample]]:
    result = {}
    for family in text_string_to_metric_families(metrics_text):
        for sample in family.samples:
            if sample.name not in result:
                result[sample.name] = []
            result[sample.name].append(sample)
    return result


def _get_samples_by_name(metrics: Dict[str, List[Sample]], name: str) -> List[Sample]:
    return metrics.get(name, [])


def _get_sample_value_by_labels(samples: List[Sample], labels: Dict[str, str]) -> float:
    for sample in samples:
        if all(sample.labels.get(k) == v for k, v in labels.items()):
            return sample.value
    raise KeyError(f"No sample found with labels {labels}")


def _assert_missing_priority_metrics(test_case: unittest.TestCase) -> None:
    """Exercise the production tokenizer and scheduler path without a default."""
    response = requests.post(
        f"{DEFAULT_URL_FOR_TEST}/generate",
        json={
            "text": "Hello world",
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": 512,
                "ignore_eos": True,
            },
        },
    )
    test_case.assertEqual(response.status_code, 200)

    metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
    test_case.assertEqual(metrics_response.status_code, 200)
    metrics = _parse_prometheus_metrics(metrics_response.text)

    e2e_count = _get_samples_by_name(
        metrics, "sglang:e2e_request_latency_seconds_count"
    )
    unknown_histograms = [
        sample
        for sample in e2e_count
        if sample.labels.get("priority") == UNKNOWN_PRIORITY_VALUE
    ]
    test_case.assertTrue(
        unknown_histograms,
        f"Expected tokenizer priority={UNKNOWN_PRIORITY_VALUE!r}, "
        f"got {[sample.labels.get('priority') for sample in e2e_count]}",
    )
    test_case.assertGreater(unknown_histograms[0].value, 0)

    for metric_name in ["sglang:num_running_reqs", "sglang:num_queue_reqs"]:
        samples = _get_samples_by_name(metrics, f"{metric_name}_by_priority")
        priorities = {sample.labels.get("priority") for sample in samples}
        test_case.assertIn(
            UNKNOWN_PRIORITY_VALUE,
            priorities,
            f"{metric_name}_by_priority used priorities {priorities}",
        )


class TestQueueCount(CustomTestCase):
    """Unit tests for QueueCount (no server needed)."""

    def test_queue_count_from_reqs(self):
        """QueueCount correctly counts per-priority breakdown."""
        reqs = [
            SimpleNamespace(priority=1, metrics_priority=1),
            SimpleNamespace(priority=1, metrics_priority=1),
            SimpleNamespace(priority=5, metrics_priority=5),
            SimpleNamespace(priority=5, metrics_priority=5),
            SimpleNamespace(priority=10, metrics_priority=10),
        ]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 5)
        self.assertEqual(qc.by_priority, {"1": 2, "5": 2, "10": 1})

    def test_queue_count_buckets_out_of_range_priorities(self):
        reqs = [
            SimpleNamespace(priority=100, metrics_priority=100),
            SimpleNamespace(priority=999, metrics_priority=999),
            SimpleNamespace(priority=-5, metrics_priority=-5),
        ]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 3)
        self.assertEqual(qc.by_priority, {"HIGH": 2, "LOW": 1})

    def test_queue_count_missing_priority(self):
        qc = QueueCount.from_reqs(
            [SimpleNamespace(priority=None, metrics_priority=None)] * 2, True
        )
        self.assertEqual(qc.by_priority, {UNKNOWN_PRIORITY_VALUE: 2})

    def test_queue_count_uses_api_priority_not_scheduler_sentinel(self):
        for effective_priority in [-sys.maxsize - 1, sys.maxsize]:
            with self.subTest(effective_priority=effective_priority):
                req = SimpleNamespace(
                    priority=effective_priority,
                    metrics_priority=None,
                )
                qc = QueueCount.from_reqs([req], enable_priority_scheduling=True)
                self.assertEqual(qc.by_priority, {UNKNOWN_PRIORITY_VALUE: 1})

    def test_queue_count_from_reqs_disabled(self):
        """Priority scheduling disabled → no breakdown."""
        reqs = [
            SimpleNamespace(priority=1, metrics_priority=1),
            SimpleNamespace(priority=5, metrics_priority=5),
        ]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=False)
        self.assertEqual(qc.total, 2)
        self.assertIsNone(qc.by_priority)

    def test_queue_count_empty(self):
        """Empty request list."""
        qc = QueueCount.from_reqs([], enable_priority_scheduling=True)
        self.assertEqual(qc.total, 0)
        self.assertEqual(qc.by_priority, {})

    def test_total_and_priority_breakdown_use_separate_gauges(self):
        registry = CollectorRegistry()
        total = Gauge("test_queue", "total", ["priority"], registry=registry)
        by_priority = Gauge(
            "test_queue_by_priority",
            "breakdown",
            ["priority"],
            registry=registry,
        )
        collector = SchedulerMetricsCollector.__new__(SchedulerMetricsCollector)
        collector.labels = {"priority": ""}
        collector._known_priorities = set()

        collector._log_gauge_queue_count(
            total,
            QueueCount(total=3, by_priority={"1": 2, "5": 1}),
            by_priority,
        )

        samples = {
            sample.name: sample
            for family in registry.collect()
            for sample in family.samples
        }
        self.assertEqual(samples["test_queue"].labels, {"priority": ""})
        self.assertEqual(samples["test_queue"].value, 3)
        breakdown = [
            sample
            for family in registry.collect()
            for sample in family.samples
            if sample.name == "test_queue_by_priority"
        ]
        self.assertEqual(sum(sample.value for sample in breakdown), 3)


class TestPriorityMetrics(CustomTestCase):
    """Test that priority-based metrics are correctly emitted when
    --enable-priority-scheduling is enabled."""

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            _MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-metrics",
                "--enable-priority-scheduling",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_priority_label_in_gauge_metrics(self):
        """Send requests with different priorities and verify that
        gauge metrics (num_running_reqs, num_queue_reqs) contain
        the priority label dimension."""

        # Send requests with different priorities to populate metrics
        for priority in [1, 5, 10]:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "Hello",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 5},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        # Fetch metrics
        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        # Aggregate metrics contain totals only; priority breakdowns use a
        # separate family so summing the aggregate cannot double count.
        for metric_name in ["sglang:num_running_reqs", "sglang:num_queue_reqs"]:
            samples = _get_samples_by_name(metrics, metric_name)
            self.assertGreater(len(samples), 0, f"No samples found for {metric_name}")
            priority_labels = {s.labels.get("priority", "") for s in samples}
            self.assertEqual(
                priority_labels,
                {""},
                f"{metric_name}: per-priority series leaked into aggregate",
            )

            by_priority = _get_samples_by_name(metrics, f"{metric_name}_by_priority")
            self.assertGreater(
                len(by_priority), 0, f"No samples found for {metric_name}_by_priority"
            )
            for sample in by_priority:
                self.assertNotEqual(sample.labels.get("priority"), "")
            for total in samples:
                keys = [key for key in total.labels if key != "priority"]
                matching = [
                    sample
                    for sample in by_priority
                    if all(sample.labels.get(key) == total.labels[key] for key in keys)
                ]
                self.assertAlmostEqual(
                    sum(sample.value for sample in matching), total.value
                )

    def test_priority_label_in_histogram_metrics(self):
        """Send requests with different priorities and verify that
        histogram metrics (TTFT, ITL, e2e latency) contain the priority label."""

        for priority in [1, 5]:
            response = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 20},
                    "priority": priority,
                },
            )
            self.assertEqual(response.status_code, 200)

        metrics_response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics")
        self.assertEqual(metrics_response.status_code, 200)
        metrics = _parse_prometheus_metrics(metrics_response.text)

        # Check histogram metrics have priority label with per-priority breakdown
        histogram_metrics = [
            "sglang:time_to_first_token_seconds",
            "sglang:e2e_request_latency_seconds",
        ]
        for metric_name in histogram_metrics:
            # Histogram metrics are emitted as _sum, _count, _bucket
            count_name = f"{metric_name}_count"
            samples = _get_samples_by_name(metrics, count_name)
            self.assertGreater(len(samples), 0, f"No samples found for {count_name}")
            # At least one sample should have a non-empty priority label
            priority_values = {s.labels.get("priority", "") for s in samples}
            non_empty = priority_values - {""}
            self.assertGreater(
                len(non_empty),
                0,
                f"{count_name}: expected per-priority samples, "
                f"got priority labels: {priority_values}",
            )
            # Verify that both priority="1" and priority="5" have count > 0
            for expected_priority in ["1", "5"]:
                matching = [
                    s for s in samples if s.labels.get("priority") == expected_priority
                ]
                self.assertGreater(
                    len(matching),
                    0,
                    f"{count_name}: no sample with priority='{expected_priority}'",
                )
                self.assertGreater(
                    matching[0].value,
                    0,
                    f"{count_name}: priority='{expected_priority}' count should be > 0",
                )

    def test_missing_priority_is_unknown(self):
        """No configured default is labeled UNKNOWN throughout the request."""
        _assert_missing_priority_metrics(self)


class TestPriorityMetricsLowValuesFirst(CustomTestCase):
    """Check missing-priority labels with the opposite scheduler ordering."""

    @classmethod
    def setUpClass(cls):
        cls.process = popen_launch_server(
            _MODEL_NAME,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--enable-metrics",
                "--enable-priority-scheduling",
                "--schedule-low-priority-values-first",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_missing_priority_is_unknown(self):
        _assert_missing_priority_metrics(self)


if __name__ == "__main__":
    unittest.main()
