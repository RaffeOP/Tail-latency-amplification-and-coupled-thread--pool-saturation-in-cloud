"""Tests for the open-loop schedule, coordinated-omission latency, and service-time model."""

import random

import pytest

from load_generator import OUTCOME_OK, LoadPlan, OpenLoopClient, poisson_offsets_s
from service_work import ServiceTimeSampler, compensated_sleep_s

NS_PER_S = 1_000_000_000
SAMPLE_COUNT = 20_000
MEAN_TOLERANCE = 0.03


def make_client() -> OpenLoopClient:
    plan = LoadPlan(root_port=1, root_vertex=0, rate_rps=10.0, duration_s=1.0, seed=1,
                    payload_bytes=0, timeout_s=1.0)
    return OpenLoopClient(stub=None, plan=plan, origin_ns=0)


def test_should_schedule_about_rate_times_duration_arrivals():
    offsets = poisson_offsets_s(200.0, 50.0, random.Random(7))
    assert len(offsets) == pytest.approx(200.0 * 50.0, rel=0.02)


def test_should_keep_every_arrival_inside_the_run_window_in_order():
    offsets = poisson_offsets_s(100.0, 10.0, random.Random(3))
    assert all(0.0 <= value < 10.0 for value in offsets)
    assert offsets == sorted(offsets)


def test_should_measure_latency_from_intended_send_time_not_actual_send_time():
    client = make_client()
    intended_ns = 1 * NS_PER_S
    sent_ns = intended_ns + 30_000_000
    end_ns = intended_ns + 50_000_000
    client._record(0, intended_ns, sent_ns, end_ns, OUTCOME_OK, [])
    assert client.result.latency_us == [50_000]
    assert client.result.lag_ns == [30_000_000]


def test_should_keep_service_time_mean_close_to_the_configured_mean():
    sampler = ServiceTimeSampler(mean_ms=10.0, sigma=0.5, seed=11)
    mean_ms = sum(sampler.sample_ms() for _ in range(SAMPLE_COUNT)) / SAMPLE_COUNT
    assert mean_ms == pytest.approx(10.0, rel=MEAN_TOLERANCE)


def test_should_shorten_sleep_requests_by_the_measured_overshoot():
    assert compensated_sleep_s(0.010) < 0.010


def test_should_never_request_a_negative_sleep():
    assert compensated_sleep_s(0.0) == 0.0
