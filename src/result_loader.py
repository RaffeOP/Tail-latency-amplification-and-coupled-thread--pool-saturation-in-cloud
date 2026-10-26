"""Loads completed run folders into per-run metrics for the analysis, tables, and figures."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pyarrow.parquet as pq

from host_gate import background_cores
from storage_manifest import completed_run_ids

US_PER_MS = 1000.0
NS_PER_S = 1e9
PERCENTILE_KEYS = {"p50": "p50.0_us", "p90": "p90.0_us", "p99": "p99.0_us", "p999": "p99.9_us", "p9999": "p99.99_us"}
ROOT_VERTEX = 0


@dataclass
class RunData:
    run: Dict[str, object]
    summary: Dict[str, float]
    latency_us: np.ndarray
    outcome: np.ndarray
    lag_ns: np.ndarray
    request_index: np.ndarray
    measured: np.ndarray
    hop_vertex: np.ndarray
    hop_request: np.ndarray
    hop_pool_wait_us: np.ndarray
    hop_business_us: np.ndarray
    hop_total_us: np.ndarray
    telemetry_rows: List[Dict[str, float]] = field(default_factory=list)
    hop_deser_us: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hop_ser_us: np.ndarray = field(default_factory=lambda: np.zeros(0))
    hop_proxy_us: np.ndarray = field(default_factory=lambda: np.zeros(0))

    @property
    def run_id(self) -> str:
        return str(self.run["run_id"])

    @property
    def rate_rps(self) -> float:
        return float(self.run["arrival_rate_rps"])

    @property
    def measured_span_s(self) -> float:
        return float(self.run["duration_s"]) - float(self.run["warmup_s"])

    def measured_latency_ms(self) -> np.ndarray:
        return self.latency_us[self.measured] / US_PER_MS

    def percentile_ms(self, key: str) -> float:
        return float(self.summary[PERCENTILE_KEYS[key]]) / US_PER_MS

    def throughput_rps(self) -> float:
        return float(self.summary["measured"]) / self.measured_span_s

    def timeout_fraction(self) -> float:
        return float(self.summary["timeout"]) / max(1.0, float(self.summary["measured"]))

    def vertex_business_ms(self, vertex: int) -> np.ndarray:
        return self.hop_business_us[self.hop_vertex == vertex] / US_PER_MS

    def root_holding_ms_measured(self) -> np.ndarray:
        measured_ids = self.request_index[self.measured]
        in_window = np.isin(self.hop_request, measured_ids)
        return self.hop_total_us[(self.hop_vertex == ROOT_VERTEX) & in_window] / US_PER_MS

    def measured_root_pool_utilization(self) -> float:
        pool = float(self.run["pool_size"])
        holds_ms = self.root_holding_ms_measured()
        if holds_ms.size == 0:
            return float("nan")
        return self.throughput_rps() * float(np.mean(holds_ms)) / US_PER_MS / pool

    def node_cores(self) -> float:
        rows = self.telemetry_rows
        span_s = (rows[-1]["t_ns"] - rows[0]["t_ns"]) / NS_PER_S
        return (rows[-1]["node_cpu_s"] - rows[0]["node_cpu_s"]) / span_s

    def background_cores(self) -> float:
        return background_cores(self.telemetry_rows)

    def host_noise_p99_ms(self) -> float:
        return float(self.summary["host_noise_p99_ms"])


def _read_columns(path: Path) -> Dict[str, np.ndarray]:
    return {name: np.asarray(values) for name, values in pq.read_table(path).to_pydict().items()}


def _telemetry_rows(path: Path) -> List[Dict[str, float]]:
    columns = _read_columns(path)
    keys = list(columns.keys())
    count = len(columns[keys[0]]) if keys else 0
    return [{key: float(columns[key][index]) for key in keys} for index in range(count)]


def _optional_hop_column(hops: Dict[str, np.ndarray], name: str) -> np.ndarray:
    if name in hops:
        return hops[name].astype(np.float64)
    return np.full(hops["vertex_id"].shape, -1.0)


def load_run(run_dir: Path) -> RunData:
    meta = json.loads((run_dir / "meta.json").read_text())
    requests = _read_columns(run_dir / "requests.parquet")
    hops = _read_columns(run_dir / "hops.parquet")
    run = RunData(
        run=meta["run"],
        summary=meta["summary"],
        latency_us=requests["latency_us"].astype(np.float64),
        outcome=requests["outcome"],
        lag_ns=requests["lag_ns"].astype(np.float64),
        request_index=requests["request_index"],
        measured=requests["measured"].astype(bool),
        hop_vertex=hops["vertex_id"],
        hop_request=hops["request_index"],
        hop_pool_wait_us=hops["pool_wait_us"].astype(np.float64),
        hop_business_us=hops["business_proc_us"].astype(np.float64),
        hop_total_us=hops["total_hop_us"].astype(np.float64),
        telemetry_rows=_telemetry_rows(run_dir / "telemetry.parquet"),
    )
    run.hop_deser_us = _optional_hop_column(hops, "deser_cpu_us")
    run.hop_ser_us = _optional_hop_column(hops, "ser_cpu_us")
    run.hop_proxy_us = _optional_hop_column(hops, "proxy_hop_us")
    return run


def load_runs(phase_root: Path) -> List[RunData]:
    """Only runs listed in the manifest count; contaminated and partial folders never enter the analysis."""
    done = completed_run_ids(phase_root / "manifest.jsonl")
    folders = sorted(path for path in phase_root.iterdir() if path.name in done and (path / "meta.json").exists())
    return [load_run(folder) for folder in folders]
