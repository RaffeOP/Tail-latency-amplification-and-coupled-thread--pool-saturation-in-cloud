"""Background sampler for CPU time and resident memory of the node processes and the host."""

import os
import threading
import time
from typing import Dict, List, Sequence

import psutil

BUSY_CPU_FIELDS = ("user", "nice", "system")
NOISE_PROBE_SAMPLES = 300
NOISE_PROBE_TARGET_S = 0.001
PERCENT_99 = 0.99


def cpu_seconds(process: psutil.Process) -> float:
    try:
        times = process.cpu_times()
    except psutil.Error:
        return 0.0
    return times.user + times.system


def host_noise_probe_p99_ms() -> float:
    """p99 overshoot of 1 ms busy-waits: a direct measure of host scheduling noise."""
    lateness_ms: List[float] = []
    for _ in range(NOISE_PROBE_SAMPLES):
        start = time.perf_counter()
        deadline = start + NOISE_PROBE_TARGET_S
        while time.perf_counter() < deadline:
            pass
        lateness_ms.append((time.perf_counter() - start - NOISE_PROBE_TARGET_S) * 1000.0)
    lateness_ms.sort()
    return lateness_ms[int(PERCENT_99 * len(lateness_ms)) - 1]


class ProcessSampler:
    def __init__(self, pids: List[int], interval_s: float, proxy_pids: Sequence[int] = ()):
        self._processes = [psutil.Process(pid) for pid in pids]
        self._proxies = [psutil.Process(pid) for pid in proxy_pids]
        self._runner = psutil.Process(os.getpid())
        self._interval_s = interval_s
        self._rows: List[Dict[str, float]] = []
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="telemetry", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join()

    def rows(self) -> List[Dict[str, float]]:
        return list(self._rows)

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            self._rows.append(self._sample())
            self._stop_event.wait(self._interval_s)

    def _sample(self) -> Dict[str, float]:
        node_cpu_s = 0.0
        node_rss_bytes = 0
        alive = 0
        for process in self._processes:
            try:
                times = process.cpu_times()
                node_cpu_s += times.user + times.system
                node_rss_bytes += process.memory_info().rss
                alive += 1
            except psutil.Error:
                continue
        proxy_cpu_s = sum(cpu_seconds(process) for process in self._proxies)
        host = psutil.cpu_times()
        runner = self._runner.cpu_times()
        host_busy_s = sum(getattr(host, name) for name in BUSY_CPU_FIELDS)
        host_total_s = host_busy_s + host.idle
        return {
            "t_ns": time.time_ns(),
            "node_cpu_s": node_cpu_s,
            "runner_cpu_s": runner.user + runner.system,
            "proxy_cpu_s": proxy_cpu_s,
            "node_rss_bytes": float(node_rss_bytes),
            "nodes_alive": float(alive),
            "host_busy_s": host_busy_s,
            "host_total_s": host_total_s,
            "loadavg_1m": os.getloadavg()[0],
            "mem_percent": float(psutil.virtual_memory().percent),
            "swap_used_bytes": float(psutil.swap_memory().used),
        }
