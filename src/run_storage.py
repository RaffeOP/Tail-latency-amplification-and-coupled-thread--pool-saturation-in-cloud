"""Writes one completed run to its result folder: raw traces, a latency summary, and metadata."""

import json
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List

import grpc
import pyarrow as pa
import pyarrow.parquet as pq
from hdrh.histogram import HdrHistogram

from load_generator import OUTCOME_ERROR, OUTCOME_OK, OUTCOME_TIMEOUT, LoadResult
from mesh_config import RunSpec

HDR_LOWEST_US = 1
HDR_HIGHEST_US = 600_000_000
HDR_SIG_FIGS = 3
REPORTED_PERCENTILES = (50.0, 90.0, 99.0, 99.9, 99.99)


def summarize(load: LoadResult, warmup_ns: int) -> Dict[str, float]:
    histogram = HdrHistogram(HDR_LOWEST_US, HDR_HIGHEST_US, HDR_SIG_FIGS)
    counts = {OUTCOME_OK: 0, OUTCOME_ERROR: 0, OUTCOME_TIMEOUT: 0}
    for offset_ns, latency_us, outcome in zip(load.intended_offset_ns, load.latency_us, load.outcome):
        if offset_ns < warmup_ns:
            continue
        histogram.record_value(max(latency_us, HDR_LOWEST_US))
        counts[outcome] += 1
    summary: Dict[str, float] = {
        "measured": float(sum(counts.values())),
        "ok": float(counts[OUTCOME_OK]),
        "error": float(counts[OUTCOME_ERROR]),
        "timeout": float(counts[OUTCOME_TIMEOUT]),
        "mean_us": float(histogram.get_mean_value()),
        "max_us": float(histogram.get_max_value()),
    }
    for percentile in REPORTED_PERCENTILES:
        summary[f"p{percentile}_us"] = float(histogram.get_value_at_percentile(percentile))
    return summary


def write_parquet(path: Path, columns: Dict[str, list]) -> None:
    pq.write_table(pa.table(columns), path)


def write_request_trace(run_dir: Path, load: LoadResult, warmup_ns: int) -> None:
    write_parquet(
        run_dir / "requests.parquet",
        {
            "request_index": load.request_index,
            "intended_offset_ns": load.intended_offset_ns,
            "lag_ns": load.lag_ns,
            "latency_us": load.latency_us,
            "outcome": load.outcome,
            "measured": [offset >= warmup_ns for offset in load.intended_offset_ns],
        },
    )


def write_hop_trace(run_dir: Path, load: LoadResult) -> None:
    rows = load.hop_rows
    write_parquet(
        run_dir / "hops.parquet",
        {
            "request_index": [row[0] for row in rows],
            "vertex_id": [row[1] for row in rows],
            "pool_wait_us": [row[2] for row in rows],
            "business_proc_us": [row[3] for row in rows],
            "total_hop_us": [row[4] for row in rows],
            "deser_cpu_us": [row[5] for row in rows],
            "ser_cpu_us": [row[6] for row in rows],
            "proxy_hop_us": [row[7] for row in rows],
        },
    )


def write_telemetry(run_dir: Path, rows: List[Dict[str, float]]) -> None:
    columns = {key: [row[key] for row in rows] for key in rows[0]} if rows else {"t_ns": []}
    write_parquet(run_dir / "telemetry.parquet", columns)


def machine_info() -> Dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "grpc": grpc.__version__,
        "cpu_count": os.cpu_count(),
    }


def write_run_folder(
    run_dir: Path,
    run: RunSpec,
    load: LoadResult,
    telemetry_rows: List[Dict[str, float]],
    summary: Dict[str, float],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    warmup_ns = int(run.warmup_s * 1_000_000_000)
    write_request_trace(run_dir, load, warmup_ns)
    write_hop_trace(run_dir, load)
    write_telemetry(run_dir, telemetry_rows)
    meta = {
        "run": asdict(run),
        "summary": summary,
        "scheduled": load.scheduled,
        "generator_aborted": load.generator_aborted,
        "machine": machine_info(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
