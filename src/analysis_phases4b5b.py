"""Analysis of the revision phases, following the rules in results/revision/plan.json.

Phase 4b: within-block paired differences against the untraced baseline of the same block, reported
as median, minimum and maximum across the five blocks, with the count of positive blocks and the
time spread of each block. Phase 5b: replicate medians with fixed-seed bootstrap intervals, shown
beside the Phase 5 dacshed and naive_retry rows at the same chain level. No hypothesis tests.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analysis_phases35 import LEVEL_PATTERN, goodput_for_run
from extension_analysis import median_interval
from result_loader import RunData, load_runs
from run_interleaved import block_key
from tail_cascade_analyzer import BOOTSTRAP_SEED

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
PHASE4B_DIR = RESULTS_ROOT / "raw_traces" / "phase6"
PHASE5B_DIR = RESULTS_ROOT / "raw_traces" / "phase7"
PHASE5_DIR = RESULTS_ROOT / "raw_traces" / "phase5"
METRICS_DIR = RESULTS_ROOT / "metrics"
COMPARATOR_CONTROLLERS = ("dacshed", "naive_retry")
BASELINE_TRACING_RATE = 0.0
DECIMALS = 3
MS_PER_MINUTE = 60_000.0
PERCENTILES_CHECKED = ("p99", "p999")


def finish_times(manifest: Path) -> Dict[str, datetime]:
    times: Dict[str, datetime] = {}
    for line in manifest.read_text().splitlines():
        if line.strip():
            record = json.loads(line)
            times[record["run_id"]] = datetime.fromisoformat(record["finished"])
    return times


def phase4b_paired(runs: List[RunData]) -> Dict[Tuple[str, str, float, int], Dict[str, float]]:
    """Paired differences keyed by load, exporter, rate, and block."""
    baseline: Dict[Tuple[str, int], RunData] = {}
    traced: Dict[Tuple[str, str, float, int], RunData] = {}
    for run in runs:
        load, block = block_key(run)
        if float(run.run["tracing_rate"]) == BASELINE_TRACING_RATE:
            baseline[(load, block)] = run
        else:
            key = (load, str(run.run["trace_exporter"]), round(float(run.run["tracing_rate"]) * 100), block)
            traced[key] = run
    paired: Dict[Tuple[str, str, float, int], Dict[str, float]] = {}
    for (load, exporter, percent, block), run in traced.items():
        control = baseline.get((load, block))
        if control is None:
            continue
        paired[(load, exporter, percent, block)] = {
            name: run.percentile_ms(name) - control.percentile_ms(name) for name in PERCENTILES_CHECKED
        }
    return paired


def block_spread_minutes(runs: List[RunData], times: Dict[str, datetime]) -> Dict[Tuple[str, int], float]:
    grouped: Dict[Tuple[str, int], List[datetime]] = defaultdict(list)
    for run in runs:
        grouped[block_key(run)].append(times[run.run["run_id"]])
    return {key: (max(values) - min(values)).total_seconds() / 60.0 for key, values in grouped.items()}


def phase4b_rows(runs: List[RunData], times: Dict[str, datetime]) -> List[Dict[str, object]]:
    paired = phase4b_paired(runs)
    spreads = block_spread_minutes(runs, times)
    configs: Dict[Tuple[str, str, float], List[Tuple[int, Dict[str, float]]]] = defaultdict(list)
    for (load, exporter, percent, block), deltas in paired.items():
        configs[(load, exporter, percent)].append((block, deltas))
    rows: List[Dict[str, object]] = []
    for (load, exporter, percent), entries in sorted(configs.items()):
        row: Dict[str, object] = {"load": load, "exporter": exporter, "rate_percent": percent,
                                  "blocks": len(entries)}
        for name in PERCENTILES_CHECKED:
            values = np.array([deltas[name] for _, deltas in entries], dtype=np.float64)
            row[f"delta_{name}_median_ms"] = round(float(np.median(values)), DECIMALS)
            row[f"delta_{name}_min_ms"] = round(float(values.min()), DECIMALS)
            row[f"delta_{name}_max_ms"] = round(float(values.max()), DECIMALS)
            row[f"delta_{name}_positive_blocks"] = int((values > 0).sum())
        row["block_spread_max_min"] = round(max(spreads[(load, block)] for block, _ in entries), DECIMALS)
        rows.append(row)
    return rows


def phase5b_groups(runs: List[RunData]) -> Dict[Tuple[str, str, str, int], List[RunData]]:
    groups: Dict[Tuple[str, str, str, int], List[RunData]] = defaultdict(list)
    for run in runs:
        match = LEVEL_PATTERN.search(str(run.run["run_id"]))
        key = (str(run.run["shape"]), str(run.run["fault"]), str(run.run["controller"]), int(match.group(1)))
        groups[key].append(run)
    return groups


def summarize_phase5(groups: Dict[Tuple[str, str, str, int], List[RunData]], source: str,
                     rng: np.random.Generator, keep=None) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for (shape, fault, controller, level), group in sorted(groups.items()):
        if keep is not None and not keep(controller):
            continue
        metrics = [goodput_for_run(run) for run in group]
        row: Dict[str, object] = {
            "source": source, "shape": shape, "fault": fault, "controller": controller, "level": level,
            "rate_rps": round(float(np.median([run.rate_rps for run in group])), DECIMALS),
            "replicates": len(group),
        }
        for index, name in enumerate(("goodput", "success", "p999_ok")):
            point, lower, upper = median_interval([m[index] for m in metrics], rng)
            row[f"{name}_median"] = round(point, DECIMALS)
            row[f"{name}_lo"] = round(lower, DECIMALS)
            row[f"{name}_hi"] = round(upper, DECIMALS)
        rows.append(row)
    return rows


def write_rows(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    phase4b_rows_out = phase4b_rows(load_runs(PHASE4B_DIR), finish_times(PHASE4B_DIR / "manifest.jsonl"))
    write_rows(METRICS_DIR / "phase4b_interleaved.csv", phase4b_rows_out)
    phase5b = summarize_phase5(phase5b_groups(load_runs(PHASE5B_DIR)), "5b", rng)
    comparators = summarize_phase5(phase5b_groups(load_runs(PHASE5_DIR)), "5", rng,
                                   keep=lambda controller: controller in COMPARATOR_CONTROLLERS)
    write_rows(METRICS_DIR / "phase5b_summary.csv", phase5b + comparators)
    print(f"phase 4b configurations: {len(phase4b_rows_out)}; phase 5b rows: {len(phase5b)}; "
          f"comparator rows: {len(comparators)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
