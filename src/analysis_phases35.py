"""Phase 3 to 5 analysis: serialization and sidecar tax, tracing overhead, and overload-control goodput.

Every number is computed from the stored run folders, read through each phase's manifest.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from result_loader import RunData, load_runs

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
METRICS_DIR = RESULTS_ROOT / "metrics"
US_PER_S = 1_000_000.0
MS_PER_S = 1000.0
LEVEL_PATTERN = re.compile(r"-L(\d+)-")
DECIMALS = 4


def request_ids(run: RunData) -> np.ndarray:
    return run.request_index[run.measured]


def sampled_hop_mask(run: RunData) -> np.ndarray:
    return np.isin(run.hop_request, request_ids(run))


def cpu_seconds(run: RunData) -> Tuple[float, float]:
    rows = run.telemetry_rows
    node = rows[-1]["node_cpu_s"] - rows[0]["node_cpu_s"]
    proxy = rows[-1].get("proxy_cpu_s", 0.0) - rows[0].get("proxy_cpu_s", 0.0)
    return node, proxy


def serialization_us_per_request(run: RunData) -> float:
    mask = sampled_hop_mask(run)
    sampled = len(set(run.hop_request[mask].tolist()))
    if sampled == 0:
        return float("nan")
    wire = np.clip(run.hop_deser_us[mask], 0, None).sum() + np.clip(run.hop_ser_us[mask], 0, None).sum()
    return float(wire) / sampled


def median_edge_overhead_us(run: RunData) -> float:
    values = run.hop_proxy_us[sampled_hop_mask(run)]
    values = values[values >= 0]
    return float(np.median(values)) if values.size else float("nan")


def phase3_rows(runs: List[RunData]) -> List[List[object]]:
    groups: Dict[Tuple[int, int], List[List[float]]] = defaultdict(list)
    for run in runs:
        node_s, proxy_s = cpu_seconds(run)
        requests = max(1, len(run.request_index))
        node_us = node_s * US_PER_S / requests
        proxy_us = proxy_s * US_PER_S / requests
        total_us = node_us + proxy_us
        serial_us = serialization_us_per_request(run)
        groups[(int(run.run["topology_args"]["depth"]), int(run.run["payload_bytes"]))].append([
            serial_us / total_us if total_us > 0 else float("nan"),
            proxy_us / total_us if total_us > 0 else float("nan"),
            median_edge_overhead_us(run),
            run.percentile_ms("p99"),
        ])
    rows: List[List[object]] = []
    for (depth, payload), values in sorted(groups.items()):
        medians = np.nanmedian(np.array(values), axis=0)
        rows.append([depth, payload, len(values)] + [round(float(v), DECIMALS) for v in medians])
    return rows


def median_pair(runs: List[RunData]) -> Tuple[float, float]:
    p99 = float(np.median([run.percentile_ms("p99") for run in runs]))
    p999 = float(np.median([run.percentile_ms("p999") for run in runs]))
    return p99, p999


def phase4_rows(runs: List[RunData]) -> List[List[object]]:
    """Medians across replicates, with the matched rate-zero baseline and each configuration's host background."""
    baseline: Dict[float, List[RunData]] = defaultdict(list)
    cases: Dict[Tuple[float, str, float], List[RunData]] = defaultdict(list)
    for run in runs:
        rate = round(run.rate_rps, 3)
        if float(run.run["tracing_rate"]) == 0.0:
            baseline[rate].append(run)
        else:
            cases[(float(run.run["tracing_rate"]), str(run.run["trace_exporter"]), rate)].append(run)
    rows: List[List[object]] = []
    for rate, base_runs in sorted(baseline.items()):
        base99, base999 = median_pair(base_runs)
        background = float(np.median([run.background_cores() for run in base_runs]))
        rows.append([0.0, "none", rate, len(base_runs), round(base99, DECIMALS), round(base999, DECIMALS),
                     0.0, 0.0, round(background, DECIMALS)])
    for (tracing, exporter, rate), values in sorted(cases.items()):
        if rate not in baseline:
            continue
        base99, base999 = median_pair(baseline[rate])
        p99, p999 = median_pair(values)
        background = float(np.median([run.background_cores() for run in values]))
        rows.append([tracing, exporter, rate, len(values), round(p99, DECIMALS), round(p999, DECIMALS),
                     round(p99 - base99, DECIMALS), round(p999 - base999, DECIMALS), round(background, DECIMALS)])
    return rows


def goodput_for_run(run: RunData) -> Tuple[float, float, float]:
    span_s = run.measured_span_s
    measured = run.measured
    outcome_ok = run.outcome == 0
    sla_us = float(run.run["sla_ms"]) * 1000.0
    within = measured & outcome_ok & (run.latency_us <= sla_us)
    goodput = float(within.sum()) / span_s
    success = float((measured & outcome_ok).sum()) / max(1, int(measured.sum()))
    ok_latency = run.latency_us[measured & outcome_ok] / 1000.0
    p999_ok = float(np.percentile(ok_latency, 99.9)) if ok_latency.size else float("nan")
    return goodput, success, p999_ok


def phase5_rows(runs: List[RunData]) -> List[List[object]]:
    groups: Dict[Tuple[str, str, int], List[Tuple[float, float, float, float]]] = defaultdict(list)
    for run in runs:
        match = LEVEL_PATTERN.search(str(run.run["run_id"]))
        level = int(match.group(1)) if match else 0
        goodput, success, p999 = goodput_for_run(run)
        groups[(str(run.run["fault"]), str(run.run["controller"]), level)].append(
            (goodput, success, p999, run.rate_rps))
    rows: List[List[object]] = []
    for (fault, controller, level), values in sorted(groups.items()):
        goodput, success, p999, rate = np.nanmean(np.array(values), axis=0)
        rows.append([fault, controller, level, round(float(rate), 3), len(values),
                     round(float(goodput), DECIMALS), round(float(success), DECIMALS), round(float(p999), DECIMALS)])
    return rows


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def main(argv: List[str]) -> int:
    summary_written = []
    for phase, builder, header, name in (
        (3, phase3_rows, ["depth", "payload_bytes", "runs", "serialization_share", "proxy_share",
                          "median_edge_overhead_us", "p99_ms"], "phase3_tax.csv"),
        (4, phase4_rows, ["tracing_rate", "exporter", "rate_rps", "runs", "p99_ms", "p999_ms",
                          "delta_p99_ms", "delta_p999_ms", "background_median_cores"], "phase4_tracing.csv"),
        (5, phase5_rows, ["fault", "controller", "level", "rate_rps", "runs", "goodput_rps",
                          "success_fraction", "p999_ok_ms"], "phase5_summary.csv"),
    ):
        root = RESULTS_ROOT / "raw_traces" / f"phase{phase}"
        if not root.exists():
            continue
        runs = load_runs(root)
        if not runs:
            continue
        write_csv(METRICS_DIR / name, header, builder(runs))
        summary_written.append(name)
    print("wrote " + (", ".join(summary_written) if summary_written else "nothing (no completed Phase 3 to 5 runs)"))
    return 0 if summary_written else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
