"""Tests for the overload controllers: admission, deadline budgets, adaptive limits, and breakers."""

import threading
import time

import pytest

import overload_controller as oc
from overload_controller import DacShedController, EnvoyStyleCircuitBreaker, NaiveRetryController

NS_PER_MS = 1_000_000


def expire_window() -> None:
    time.sleep(oc.DAC_WINDOW_S * 1.5)


def test_should_admit_requests_up_to_the_initial_limit():
    controller = DacShedController(initial_limit=3)
    decisions = [controller.admit(0, 0.0).admitted for _ in range(4)]
    assert decisions == [True, True, True, False]


def test_should_reject_when_the_remaining_deadline_budget_is_below_the_critical_path_floor():
    controller = DacShedController()
    deadline_ns = time.time_ns() + 5 * NS_PER_MS
    decision = controller.admit(deadline_ns, min_remaining_ms=50.0)
    assert (decision.admitted, decision.reason) == (False, "DEADLINE_BUDGET")


def test_should_admit_when_the_deadline_is_absent():
    assert DacShedController().admit(0, min_remaining_ms=1e9).admitted


def test_should_not_shrink_the_limit_on_downstream_failure_alone():
    controller = DacShedController(initial_limit=16)
    expire_window()
    controller.admit(0, 0.0)
    controller.complete(latency_ms=5.0, failed=True)
    assert controller.limit == pytest.approx(16.0)


def test_should_shrink_the_limit_when_the_latency_gradient_exceeds_the_threshold():
    controller = DacShedController(initial_limit=16)
    expire_window()
    controller.admit(0, 0.0)
    controller.complete(latency_ms=5.0, failed=False)
    expire_window()
    controller.admit(0, 0.0)
    controller.complete(latency_ms=50.0, failed=False)
    assert controller.limit < 16


def test_should_not_shrink_the_limit_when_only_its_own_admissions_are_rejected():
    controller = DacShedController(initial_limit=2)
    controller.admit(0, 0.0)
    controller.admit(0, 0.0)
    for _ in range(5):
        assert controller.admit(0, 0.0).admitted is False
    expire_window()
    controller.complete(latency_ms=5.0, failed=False)
    assert controller.limit == pytest.approx(2.0)


def test_should_release_capacity_when_a_request_completes():
    controller = DacShedController(initial_limit=1)
    assert controller.admit(0, 0.0).admitted
    assert not controller.admit(0, 0.0).admitted
    controller.complete(latency_ms=1.0, failed=False)
    assert controller.admit(0, 0.0).admitted


def test_should_keep_in_flight_count_consistent_under_concurrent_use():
    controller = DacShedController(initial_limit=1000)
    admitted = []

    def worker() -> None:
        for _ in range(200):
            if controller.admit(0, 0.0).admitted:
                admitted.append(1)
                controller.complete(latency_ms=1.0, failed=False)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert controller.admit(0, 0.0).admitted
    assert len(admitted) == 8 * 200


def test_should_open_the_envoy_style_breaker_after_consecutive_failures():
    breaker = EnvoyStyleCircuitBreaker(error_threshold=3, cooldown_s=10.0)
    for _ in range(3):
        assert breaker.admit(0, 0.0).admitted
        breaker.complete(latency_ms=1.0, failed=True)
    assert breaker.admit(0, 0.0).reason == "CIRCUIT_OPEN"


def test_should_cap_the_envoy_style_breaker_at_connections_plus_pending():
    breaker = EnvoyStyleCircuitBreaker(max_connections=1, max_pending=1)
    assert breaker.admit(0, 0.0).admitted
    assert breaker.admit(0, 0.0).admitted
    assert breaker.admit(0, 0.0).reason == "QUEUE_OVERFLOW"


def test_should_retry_failures_only_until_the_retry_budget_is_spent():
    policy = NaiveRetryController(max_retries=2)
    assert policy.should_retry(attempt=0, failed=True)
    assert policy.should_retry(attempt=1, failed=True)
    assert not policy.should_retry(attempt=2, failed=True)
    assert not policy.should_retry(attempt=0, failed=False)
