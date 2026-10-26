"""Tests for the replicate aggregation used by the Phase 4 tracing comparison."""

from analysis_phases35 import median_pair


class StubRun:
    def __init__(self, p99: float, p999: float):
        self._latencies = {"p99": p99, "p999": p999}

    def percentile_ms(self, name: str) -> float:
        return self._latencies[name]


def test_should_ignore_one_outlier_replicate_in_the_p999_median():
    runs = [StubRun(70.0, 80.0), StubRun(71.0, 82.0), StubRun(72.0, 385.0)]
    assert median_pair(runs)[1] == 82.0


def test_should_return_the_shared_p99_when_replicates_agree():
    runs = [StubRun(60.0, 90.0), StubRun(60.0, 90.0), StubRun(60.0, 90.0)]
    assert median_pair(runs)[0] == 60.0
