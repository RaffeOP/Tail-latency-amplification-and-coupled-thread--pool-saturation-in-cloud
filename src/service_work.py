"""Service-time sampling and the blocking work model shared by every node runtime."""

import math
import random
import threading
import time

SECONDS_PER_MS = 1e-3
SLEEP_OVERSHOOT_BASE_MS = 0.3
SLEEP_OVERSHOOT_SLOPE = 0.44


class ServiceTimeSampler:
    """Lognormal service times whose mean equals mean_ms for any sigma."""

    def __init__(self, mean_ms: float, sigma: float, seed: int):
        self._sigma = sigma
        self._mu = math.log(mean_ms) - 0.5 * sigma * sigma
        self._rng = random.Random(seed)
        self._lock = threading.Lock()

    def sample_ms(self) -> float:
        with self._lock:
            return self._rng.lognormvariate(self._mu, self._sigma)


def burn_cpu(seconds: float) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        pass


def compensated_sleep_s(target_s: float) -> float:
    """Shortens a sleep so that the host timer's overshoot lands on the target.

    On the test Mac, sleep(t) returns after about 0.3 ms + 0.44 * t extra (fit to 1-12 ms sleeps).
    """
    target_ms = target_s / SECONDS_PER_MS
    corrected_ms = (target_ms - SLEEP_OVERSHOOT_BASE_MS) / (1.0 + SLEEP_OVERSHOOT_SLOPE)
    return max(0.0, corrected_ms) * SECONDS_PER_MS


def blocking_work(service_ms: float, burn_fraction: float) -> None:
    """Models I/O-bound business logic: most of the time is a wait, a small share is CPU."""
    total_s = service_ms * SECONDS_PER_MS
    burn_s = total_s * burn_fraction
    time.sleep(compensated_sleep_s(total_s - burn_s))
    burn_cpu(burn_s)
