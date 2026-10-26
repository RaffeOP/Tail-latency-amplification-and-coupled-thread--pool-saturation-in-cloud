"""Quiet-host gate: starts a run only when background load is low, and detects disturbed runs.

Other workloads on the same machine add CPU contention and scheduling noise. The gate waits
for a quiet window before each run. A run whose background load exceeds the limit while it
ran is marked contaminated and is retried later rather than counted.
"""

import time
from typing import Callable, Dict, List

import psutil

from run_telemetry import host_noise_probe_p99_ms

QUIET_BUSY_PERCENT = 25.0
CONTAMINATION_CORES_LIMIT = 1.0
QUIET_WINDOW_S = 1.0
QUIET_WINDOWS_REQUIRED = 3
QUIET_NOISE_MS = 3.0
WAIT_REPORT_EVERY_S = 60.0
BUSY_FIELDS = ("user", "nice", "system")


def host_busy_percent(window_s: float) -> float:
    before = psutil.cpu_times()
    time.sleep(window_s)
    after = psutil.cpu_times()
    busy = sum(getattr(after, name) - getattr(before, name) for name in BUSY_FIELDS)
    idle = after.idle - before.idle
    total = busy + idle
    return 100.0 * busy / total if total > 0 else 100.0


def wait_for_quiet_host(should_stop: Callable[[], bool], log: Callable[[str], None]) -> bool:
    quiet_windows = 0
    last_report_s = float("-inf")
    while not should_stop():
        busy = host_busy_percent(QUIET_WINDOW_S)
        noise_ms = host_noise_probe_p99_ms()
        is_quiet = busy <= QUIET_BUSY_PERCENT and noise_ms <= QUIET_NOISE_MS
        quiet_windows = quiet_windows + 1 if is_quiet else 0
        if quiet_windows >= QUIET_WINDOWS_REQUIRED:
            return True
        if time.monotonic() - last_report_s >= WAIT_REPORT_EVERY_S:
            log(f"waiting for a quiet host: busy {busy:.0f}%, noise p99 {noise_ms:.1f} ms")
            last_report_s = time.monotonic()
    return False


def background_cores(rows: List[Dict[str, float]]) -> float:
    if len(rows) < 2:
        return float("inf")
    first, last = rows[0], rows[-1]
    elapsed_s = (last["t_ns"] - first["t_ns"]) / 1e9
    if elapsed_s <= 0:
        return float("inf")
    host_cores = (last["host_busy_s"] - first["host_busy_s"]) / elapsed_s
    runner_key = "runner_cpu_s"
    own_cores = (
        (last["node_cpu_s"] - first["node_cpu_s"])
        + (last.get(runner_key, 0.0) - first.get(runner_key, 0.0))
        + (last.get("proxy_cpu_s", 0.0) - first.get("proxy_cpu_s", 0.0))
    ) / elapsed_s
    return max(0.0, host_cores - own_cores)


def is_contaminated(rows: List[Dict[str, float]]) -> bool:
    return background_cores(rows) > CONTAMINATION_CORES_LIMIT
