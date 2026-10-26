"""Overload controllers: no-op, naive retries, an Envoy-style static breaker, and DAC-Shed.

The Envoy-style breaker reproduces Envoy's static knobs (connection cap, pending cap, consecutive
error ejection) in-process. It is an emulation, not an Envoy deployment. All controllers are
thread-safe because the synchronous runtime calls them from several handler threads.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque

NS_PER_MS = 1_000_000
DEFAULT_MAX_RETRIES = 3
DEFAULT_CONNECTIONS = 64
DEFAULT_PENDING = 128
DEFAULT_ERROR_THRESHOLD = 5
DEFAULT_COOLDOWN_S = 2.0
DAC_INITIAL_LIMIT = 16
DAC_MIN_LIMIT = 4
DAC_MAX_LIMIT = 128
DAC_ALPHA_INCREASE = 1.0
DAC_BETA_DECREASE = 0.85
DAC_THETA_LOW = 1.5
DAC_THETA_HIGH = 3.0
DAC_WINDOW_S = 0.05
DAC_BASE_WINDOWS = 20
DAC_UTILIZATION_FLOOR = 0.8
DAC_NEUTRAL_GRADIENT = 1.0


@dataclass(frozen=True)
class Admission:
    admitted: bool
    reason: str


ADMITTED = Admission(True, "OK")


class OverloadController:
    def admit(self, deadline_ns: int, min_remaining_ms: float) -> Admission:
        return ADMITTED

    def complete(self, latency_ms: float, failed: bool) -> None:
        return None


class NoOpController(OverloadController):
    """Unconstrained baseline."""


class NaiveRetryController:
    """Client-side policy: retry any failure immediately, up to a fixed count, with no budget."""

    def __init__(self, max_retries: int = DEFAULT_MAX_RETRIES):
        self._max_retries = max_retries

    def should_retry(self, attempt: int, failed: bool) -> bool:
        return failed and attempt < self._max_retries


class EnvoyStyleCircuitBreaker(OverloadController):
    def __init__(self, max_connections: int = DEFAULT_CONNECTIONS, max_pending: int = DEFAULT_PENDING,
                 error_threshold: int = DEFAULT_ERROR_THRESHOLD, cooldown_s: float = DEFAULT_COOLDOWN_S):
        self._max_in_flight = max_connections + max_pending
        self._error_threshold = error_threshold
        self._cooldown_s = cooldown_s
        self._lock = threading.Lock()
        self._in_flight = 0
        self._consecutive_errors = 0
        self._open_until = 0.0

    def admit(self, deadline_ns: int, min_remaining_ms: float) -> Admission:
        with self._lock:
            if time.monotonic() < self._open_until:
                return Admission(False, "CIRCUIT_OPEN")
            if self._in_flight >= self._max_in_flight:
                return Admission(False, "QUEUE_OVERFLOW")
            self._in_flight += 1
            return ADMITTED

    def complete(self, latency_ms: float, failed: bool) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            if not failed:
                self._consecutive_errors = 0
                return
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._error_threshold:
                self._open_until = time.monotonic() + self._cooldown_s
                self._consecutive_errors = 0


class DacShedController(OverloadController):
    """Deadline-budget shedding plus a Vegas-style concurrency limit driven by a latency gradient.

    The gradient is the window mean latency over the base latency, where the base is the smallest
    window mean among the last DAC_BASE_WINDOWS windows. A gradient above DAC_THETA_HIGH shrinks the
    limit, and a gradient below DAC_THETA_LOW grows it when the limit is in use. Downstream failures do
    not move the limit: a failing dependency is not evidence of local queueing, and shrinking on it would
    shed healthy traffic. Our own rejections never shrink the limit either, so shedding cannot feed on itself.
    """

    def __init__(self, min_limit: int = DAC_MIN_LIMIT, max_limit: int = DAC_MAX_LIMIT,
                 initial_limit: int = DAC_INITIAL_LIMIT):
        self._min_limit = min_limit
        self._max_limit = max_limit
        self._limit = float(initial_limit)
        self._lock = threading.Lock()
        self._in_flight = 0
        self._window_latencies: list = []
        self._window_started = time.monotonic()
        self._base_windows: Deque[float] = deque(maxlen=DAC_BASE_WINDOWS)
        self._gradient = DAC_NEUTRAL_GRADIENT

    @property
    def limit(self) -> float:
        with self._lock:
            return self._limit

    @property
    def gradient(self) -> float:
        """Latency gradient of the most recent completed window; read by the retry gate."""
        with self._lock:
            return self._gradient

    def admit(self, deadline_ns: int, min_remaining_ms: float) -> Admission:
        with self._lock:
            if deadline_ns > 0:
                remaining_ms = (deadline_ns - time.time_ns()) / NS_PER_MS
                if remaining_ms < min_remaining_ms:
                    return Admission(False, "DEADLINE_BUDGET")
            if self._in_flight >= int(self._limit):
                return Admission(False, "CONCURRENCY_LIMIT")
            self._in_flight += 1
            return ADMITTED

    def complete(self, latency_ms: float, failed: bool) -> None:
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)
            self._window_latencies.append(latency_ms)
            if time.monotonic() - self._window_started >= DAC_WINDOW_S:
                self._adjust_locked()

    def _adjust_locked(self) -> None:
        if self._window_latencies:
            window_mean = sum(self._window_latencies) / len(self._window_latencies)
            self._base_windows.append(window_mean)
            gradient = window_mean / min(self._base_windows)
            self._gradient = gradient
            if gradient > DAC_THETA_HIGH:
                self._limit = max(self._min_limit, self._limit * DAC_BETA_DECREASE)
            elif gradient < DAC_THETA_LOW and self._in_flight >= DAC_UTILIZATION_FLOOR * self._limit:
                self._limit = min(self._max_limit, self._limit + DAC_ALPHA_INCREASE)
        self._window_latencies = []
        self._window_started = time.monotonic()
