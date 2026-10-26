"""Tests for run summaries, contamination detection, and the manifest that enables resumption."""

from pathlib import Path

import pytest

from host_gate import CONTAMINATION_CORES_LIMIT, background_cores, is_contaminated
from load_generator import OUTCOME_ERROR, OUTCOME_OK, OUTCOME_TIMEOUT, LoadResult
from run_storage import summarize
from storage_manifest import append_record, completed_run_ids

NS_PER_S = 1_000_000_000


def make_load(offsets_s, latencies_us, outcomes) -> LoadResult:
    return LoadResult(
        scheduled=len(offsets_s),
        request_index=list(range(len(offsets_s))),
        intended_offset_ns=[int(value * NS_PER_S) for value in offsets_s],
        lag_ns=[0] * len(offsets_s),
        latency_us=latencies_us,
        outcome=outcomes,
    )


def test_should_exclude_warm_up_requests_from_the_summary():
    load = make_load([0.5, 2.0, 3.0], [1000, 2000, 3000], [OUTCOME_OK, OUTCOME_OK, OUTCOME_OK])
    summary = summarize(load, warmup_ns=1 * NS_PER_S)
    assert summary["measured"] == 2.0


def test_should_count_timeouts_and_errors_separately_from_successes():
    load = make_load([2.0, 2.1, 2.2], [1000, 1000, 1000], [OUTCOME_OK, OUTCOME_TIMEOUT, OUTCOME_ERROR])
    summary = summarize(load, warmup_ns=0)
    assert (summary["ok"], summary["timeout"], summary["error"]) == (1.0, 1.0, 1.0)


def test_should_report_the_median_latency_from_the_histogram():
    load = make_load([2.0] * 101, [5000] * 101, [OUTCOME_OK] * 101)
    summary = summarize(load, warmup_ns=0)
    assert summary["p50.0_us"] == pytest.approx(5000.0, rel=0.01)


def make_rows(host_busy_delta_s: float, own_delta_s: float, elapsed_s: float):
    start = {"t_ns": 0, "host_busy_s": 0.0, "node_cpu_s": 0.0, "runner_cpu_s": 0.0}
    end = {
        "t_ns": int(elapsed_s * NS_PER_S),
        "host_busy_s": host_busy_delta_s,
        "node_cpu_s": own_delta_s,
        "runner_cpu_s": 0.0,
    }
    return [start, end]


def test_should_report_background_cores_as_host_busy_minus_own_processes():
    rows = make_rows(host_busy_delta_s=40.0, own_delta_s=10.0, elapsed_s=10.0)
    assert background_cores(rows) == 3.0


def test_should_flag_runs_whose_background_load_exceeds_the_limit():
    rows = make_rows(host_busy_delta_s=10.0 + (CONTAMINATION_CORES_LIMIT + 1.0) * 10.0, own_delta_s=10.0, elapsed_s=10.0)
    assert is_contaminated(rows)


def test_should_not_flag_runs_whose_background_load_is_within_the_limit():
    rows = make_rows(host_busy_delta_s=10.0 + 0.5 * 10.0, own_delta_s=10.0, elapsed_s=10.0)
    assert not is_contaminated(rows)


def test_should_list_only_runs_that_were_appended_to_the_manifest(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    append_record(manifest, {"run_id": "run-a"})
    append_record(manifest, {"run_id": "run-b"})
    assert completed_run_ids(manifest) == {"run-a", "run-b"}
