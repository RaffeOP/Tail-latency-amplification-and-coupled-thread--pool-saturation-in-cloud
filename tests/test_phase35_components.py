"""Tests for the Phase 3 to 5 components: wire timing, fault windows, tracing, controls, and specs."""

import random
from pathlib import Path

import pytest

import mesh_service_pb2 as pb
from mesh_config import (
    CONTROLLER_DACSHED,
    CONTROLLER_NONE,
    CONTROLLER_STATIC_CB,
    EXPORTER_SYNC,
    FAULT_ERRORS,
    FAULT_HANG,
    FAULT_NONE,
    FAULT_SLOWDOWN,
    NodeSpec,
)
from node_controls import build_overload
from node_faults import FaultPlan, fails_under_fault, hang_ms_under_fault, service_ms_under_fault
from node_tracing import build_tracer
from node_wire import decode_timed, edge_overhead_us, encode_timed

WINDOW_START_NS = 1_000
WINDOW_END_NS = 2_000
INSIDE_NS = 1_500
BEFORE_NS = 500


def make_plan(mode: str) -> FaultPlan:
    return FaultPlan(mode=mode, start_ns=WINDOW_START_NS, end_ns=WINDOW_END_NS, factor=4.0,
                     error_prob=0.3, hang_ms=1000.0)


def test_should_round_trip_a_request_through_timed_encoding():
    request = pb.HopRequest(request_id="r1", current_vertex_id=3, payload_blob=b"abc")
    raw, encode_us = encode_timed(request)
    decoded, decode_us = decode_timed(raw, pb.HopRequest)
    assert decoded == request
    assert encode_us >= 0 and decode_us >= 0


def test_should_attribute_caller_time_beyond_the_child_processing_to_the_edge():
    assert edge_overhead_us(call_wall_us=1500, child_total_us=1000) == 500


def test_should_never_report_a_negative_edge_overhead():
    assert edge_overhead_us(call_wall_us=900, child_total_us=1000) == 0


def test_should_leave_the_edge_overhead_unmeasured_without_a_child_timeline():
    assert edge_overhead_us(call_wall_us=900, child_total_us=-1) == -1


def test_should_slow_service_only_inside_the_slowdown_window():
    plan = make_plan(FAULT_SLOWDOWN)
    assert service_ms_under_fault(plan, 10.0, INSIDE_NS) == pytest.approx(40.0)
    assert service_ms_under_fault(plan, 10.0, BEFORE_NS) == pytest.approx(10.0)


def test_should_hang_only_inside_the_hang_window():
    plan = make_plan(FAULT_HANG)
    assert hang_ms_under_fault(plan, INSIDE_NS) == pytest.approx(1000.0)
    assert hang_ms_under_fault(plan, WINDOW_END_NS) == 0.0


def test_should_fail_requests_at_the_configured_rate_inside_the_error_window():
    plan = make_plan(FAULT_ERRORS)
    rng = random.Random(7)
    failures = sum(fails_under_fault(plan, INSIDE_NS, rng) for _ in range(20000))
    assert failures / 20000 == pytest.approx(0.3, abs=0.02)


def test_should_never_fail_requests_outside_the_error_window():
    plan = make_plan(FAULT_ERRORS)
    assert not any(fails_under_fault(plan, BEFORE_NS, random.Random(i)) for i in range(50))


def test_should_inject_nothing_when_no_fault_is_configured():
    plan = make_plan(FAULT_NONE)
    assert service_ms_under_fault(plan, 10.0, INSIDE_NS) == pytest.approx(10.0)
    assert not fails_under_fault(plan, INSIDE_NS, random.Random(1))


def test_should_build_no_tracer_when_the_sampling_rate_is_zero(tmp_path: Path):
    assert build_tracer(0.0, EXPORTER_SYNC, str(tmp_path / "spans.jsonl")) is None


def test_should_export_only_sampled_spans_as_json_lines(tmp_path: Path):
    path = tmp_path / "spans.jsonl"
    tracer = build_tracer(1.0, EXPORTER_SYNC, str(path))
    span = tracer.begin("")
    tracer.finish(span)
    tracer.shutdown()
    lines = path.read_text().splitlines()
    assert len(lines) == 1 and '"name": "ExecuteHop"' in lines[0]


def test_should_export_no_spans_when_the_root_is_not_sampled(tmp_path: Path):
    path = tmp_path / "spans.jsonl"
    tracer = build_tracer(1e-9, EXPORTER_SYNC, str(path))
    assert tracer is not None
    for _ in range(50):
        tracer.finish(tracer.begin(""))
    tracer.shutdown()
    assert path.read_text() == ""


def test_should_pass_the_trace_context_to_children_so_that_they_share_a_trace(tmp_path: Path):
    tracer = build_tracer(1.0, EXPORTER_SYNC, str(tmp_path / "spans.jsonl"))
    parent = tracer.begin("")
    traceparent = tracer.child_traceparent(parent)
    child = tracer.begin(traceparent)
    assert format(child.get_span_context().trace_id, "032x") == format(parent.get_span_context().trace_id, "032x")
    tracer.finish(child)
    tracer.finish(parent)
    tracer.shutdown()


def test_should_build_no_interceptors_for_the_unprotected_runtime():
    assert build_overload(CONTROLLER_NONE, 0.0)[0] == []


def test_should_build_one_interceptor_for_each_overload_controller():
    assert len(build_overload(CONTROLLER_STATIC_CB, 0.0)[0]) == 1
    assert len(build_overload(CONTROLLER_DACSHED, 10.0)[0]) == 1


def test_should_give_each_node_spec_a_round_trip_through_json(tmp_path: Path):
    spec = NodeSpec(vertex_id=1, port=31001, children=[], runtime="sync", pool_size=8, service_mean_ms=10.0,
                    service_sigma=0.5, cpu_burn_fraction=0.1, seed=5, controller=CONTROLLER_DACSHED,
                    min_remaining_ms=42.0, max_retries=2)
    path = tmp_path / "spec.json"
    spec.write_json(path)
    assert NodeSpec.read_json(path) == spec
