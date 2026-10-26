"""Tests for the dacshed_retry gate, the injected-error status, and the Phase 4b and 5b run matrices."""

import time
from collections import Counter

import mesh_service_pb2 as pb
from campaign_extra import phase4b_runs, phase5b_chain_runs, phase5b_diamond_runs
from node_common import STATUS_FAILED, STATUS_INJECTED, STATUS_OK
from node_retry import RETRY_BUDGET_FLOOR_MULTIPLE, RetryGate
from node_sync import call_child, hop_status, ChildTarget
from node_wire import CallResult
from overload_controller import DAC_THETA_HIGH
from run_interleaved import group_blocks

NS_PER_MS = 1_000_000
CHILD_FLOOR_MS = 20.0
AMPLE_BUDGET_MS = 1_000.0
LOW_GRADIENT = DAC_THETA_HIGH - 1.0
HIGH_GRADIENT = DAC_THETA_HIGH + 1.0
FIRST_ATTEMPT = 0
SECOND_ATTEMPT = 1
PHASE4B_RUNS_PER_BLOCK = 9
PHASE4B_BLOCKS_PER_LOAD = 5
PHASE4B_LOADS = 2
PHASE5B_CHAIN_RUNS = 75
PHASE5B_DIAMOND_RUNS = 24


def result_with_status(status: int) -> CallResult:
    return CallResult(pb.HopResponse(status_code=status), -1, -1, -1)


def no_response() -> CallResult:
    return CallResult(None, -1, -1, -1)


def deadline_in(budget_ms: float) -> int:
    return time.time_ns() + int(budget_ms * NS_PER_MS)


def gate(gradient: float) -> RetryGate:
    return RetryGate(lambda: gradient)


def test_should_allow_one_retry_after_an_injected_error_with_ample_budget_and_low_gradient():
    assert gate(LOW_GRADIENT).allows_retry(FIRST_ATTEMPT, result_with_status(STATUS_INJECTED),
                                           deadline_in(AMPLE_BUDGET_MS), CHILD_FLOOR_MS)


def test_should_refuse_retry_on_the_second_attempt():
    assert not gate(LOW_GRADIENT).allows_retry(SECOND_ATTEMPT, result_with_status(STATUS_INJECTED),
                                               deadline_in(AMPLE_BUDGET_MS), CHILD_FLOOR_MS)


def test_should_refuse_retry_when_there_is_no_response_as_for_shed_or_timeout():
    assert not gate(LOW_GRADIENT).allows_retry(FIRST_ATTEMPT, no_response(),
                                               deadline_in(AMPLE_BUDGET_MS), CHILD_FLOOR_MS)


def test_should_refuse_retry_for_a_failure_propagated_from_deeper_in_the_graph():
    assert not gate(LOW_GRADIENT).allows_retry(FIRST_ATTEMPT, result_with_status(STATUS_FAILED),
                                               deadline_in(AMPLE_BUDGET_MS), CHILD_FLOOR_MS)


def test_should_refuse_retry_when_budget_is_below_twice_the_child_floor():
    too_little_ms = RETRY_BUDGET_FLOOR_MULTIPLE * CHILD_FLOOR_MS - 1.0
    assert not gate(LOW_GRADIENT).allows_retry(FIRST_ATTEMPT, result_with_status(STATUS_INJECTED),
                                               deadline_in(too_little_ms), CHILD_FLOOR_MS)


def test_should_refuse_retry_when_the_latency_gradient_reaches_the_high_threshold():
    assert not gate(HIGH_GRADIENT).allows_retry(FIRST_ATTEMPT, result_with_status(STATUS_INJECTED),
                                                deadline_in(AMPLE_BUDGET_MS), CHILD_FLOOR_MS)


def test_should_report_injected_errors_apart_from_propagated_failures():
    assert hop_status(has_failed=False, failed_by_fault=True) == STATUS_INJECTED
    assert hop_status(has_failed=True, failed_by_fault=False) == STATUS_FAILED
    assert hop_status(has_failed=False, failed_by_fault=False) == STATUS_OK


class ScriptedCaller:
    def __init__(self, results):
        self._results = list(results)
        self.calls = 0

    def call(self, request):
        self.calls += 1
        return self._results.pop(0)


def test_should_return_the_retry_result_when_the_gate_allows_one_retry():
    caller = ScriptedCaller([result_with_status(STATUS_INJECTED), result_with_status(STATUS_OK)])
    target = ChildTarget(caller, 1, CHILD_FLOOR_MS)
    request = pb.HopRequest(absolute_deadline_ns=deadline_in(AMPLE_BUDGET_MS))
    outcome = call_child(caller, request, target, "", 1, gate(LOW_GRADIENT))
    assert outcome is not None and outcome.response.status_code == STATUS_OK
    assert caller.calls == 2


def test_should_not_retry_a_shed_decision_even_when_retries_are_allowed():
    caller = ScriptedCaller([no_response(), result_with_status(STATUS_OK)])
    target = ChildTarget(caller, 1, CHILD_FLOOR_MS)
    request = pb.HopRequest(absolute_deadline_ns=deadline_in(AMPLE_BUDGET_MS))
    assert call_child(caller, request, target, "", 1, gate(LOW_GRADIENT)) is None
    assert caller.calls == 1


def test_should_keep_naive_retry_behavior_when_no_gate_is_given():
    caller = ScriptedCaller([no_response(), no_response(), result_with_status(STATUS_OK)])
    target = ChildTarget(caller, 1, CHILD_FLOOR_MS)
    request = pb.HopRequest(absolute_deadline_ns=deadline_in(AMPLE_BUDGET_MS))
    outcome = call_child(caller, request, target, "", 2, None)
    assert outcome is not None and caller.calls == 3


def test_should_run_nine_configurations_in_each_of_ten_blocks_for_phase_4b():
    runs = phase4b_runs()
    assert len(runs) == PHASE4B_RUNS_PER_BLOCK * PHASE4B_BLOCKS_PER_LOAD * PHASE4B_LOADS
    blocks = group_blocks(runs)
    assert len(blocks) == PHASE4B_BLOCKS_PER_LOAD * PHASE4B_LOADS
    assert all(len(block) == PHASE4B_RUNS_PER_BLOCK for block in blocks.values())


def test_should_give_every_phase_4b_run_a_unique_identifier():
    identifiers = [run.run_id for run in phase4b_runs()]
    assert len(set(identifiers)) == len(identifiers)


def test_should_keep_the_same_block_composition_across_repeated_builds_for_phase_4b():
    first = [run.run_id for run in phase4b_runs()]
    second = [run.run_id for run in phase4b_runs()]
    assert first == second


def test_should_include_the_untraced_baseline_once_per_block_for_phase_4b():
    counts = Counter(run.run_id for run in phase4b_runs() if run.tracing_rate == 0.0)
    assert all(count == 1 for count in counts.values())
    assert len(counts) == PHASE4B_BLOCKS_PER_LOAD * PHASE4B_LOADS


def test_should_build_the_phase_5b_chain_matrix():
    runs = phase5b_chain_runs()
    assert len(runs) == PHASE5B_CHAIN_RUNS
    assert all(run.controller == "dacshed_retry" and run.max_retries == 1 for run in runs)


def test_should_build_the_phase_5b_diamond_matrix_with_a_shared_leaf_target():
    runs = phase5b_diamond_runs()
    assert len(runs) == PHASE5B_DIAMOND_RUNS
    assert {run.topology for run in runs} == {"diamond"}
    assert len({run.fault_target for run in runs}) == 1
