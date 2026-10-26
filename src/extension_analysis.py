"""Five-replicate analysis of Phases 4 and 5, following the rules pre-registered in results/extension/plan.json.

Each configuration is summarised by the median across its replicates. Intervals are 95% percentile
bootstrap intervals over replicates (BOOTSTRAP_RESAMPLES resamples, fixed seed). The values from the
original replicates (Phase 4 replicates 1 to 3, Phase 5 replicates 1 and 2) are written beside them.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from analysis_phases35 import LEVEL_PATTERN, goodput_for_run
from result_loader import RunData, load_runs
from tail_cascade_analyzer import (
    BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED, CI_LOWER_PERCENT, CI_UPPER_PERCENT,
)

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
ORIGINAL_MAX_REPLICATE = {4: 3, 5: 2}
DECIMALS = 3


def replicate_of(run: RunData) -> int:
    return int(run.run["replicate"])


def median_interval(values: Sequence[float], rng: np.random.Generator) -> Tuple[float, float, float]:
    data = np.asarray(values, dtype=np.float64)
    picks = rng.integers(0, data.size, size=(BOOTSTRAP_RESAMPLES, data.size))
    draws = np.nanmedian(data[picks], axis=1)
    lower, upper = np.percentile(draws, [CI_LOWER_PERCENT, CI_UPPER_PERCENT])
    return float(np.nanmedian(data)), float(lower), float(upper)


def delta_interval(traced: Sequence[float], baseline: Sequence[float],
                   rng: np.random.Generator) -> Tuple[float, float, float]:
    treated = np.asarray(traced, dtype=np.float64)
    control = np.asarray(baseline, dtype=np.float64)
    point = float(np.nanmedian(treated) - np.nanmedian(control))
    treated_draws = np.nanmedian(treated[rng.integers(0, treated.size, size=(BOOTSTRAP_RESAMPLES, treated.size))], axis=1)
    control_draws = np.nanmedian(control[rng.integers(0, control.size, size=(BOOTSTRAP_RESAMPLES, control.size))], axis=1)
    lower, upper = np.percentile(treated_draws - control_draws, [CI_LOWER_PERCENT, CI_UPPER_PERCENT])
    return point, float(lower), float(upper)


def phase4_rows(runs: List[RunData], rng: np.random.Generator) -> List[Dict[str, object]]:
    baseline: Dict[float, List[RunData]] = defaultdict(list)
    traced: Dict[Tuple[float, str, float], List[RunData]] = defaultdict(list)
    for run in runs:
        rate = round(run.rate_rps, 3)
        if float(run.run["tracing_rate"]) == 0.0:
            baseline[rate].append(run)
        else:
            traced[(float(run.run["tracing_rate"]), str(run.run["trace_exporter"]), rate)].append(run)
    rows: List[Dict[str, object]] = []
    for (tracing, exporter, rate), group in sorted(traced.items()):
        base = baseline.get(rate, [])
        if not base:
            continue
        original = [r for r in group if replicate_of(r) <= ORIGINAL_MAX_REPLICATE[4]]
        base99 = [r.percentile_ms("p99") for r in base]
        base999 = [r.percentile_ms("p999") for r in base]
        d99, lo99, hi99 = delta_interval([r.percentile_ms("p99") for r in group], base99, rng)
        d999, lo999, hi999 = delta_interval([r.percentile_ms("p999") for r in group], base999, rng)
        rows.append({
            "tracing_rate": tracing, "exporter": exporter, "rate_rps": rate,
            "replicates": len(group), "baseline_replicates": len(base),
            "delta_p99_ms": round(d99, DECIMALS), "delta_p99_lo_ms": round(lo99, DECIMALS),
            "delta_p99_hi_ms": round(hi99, DECIMALS),
            "delta_p999_ms": round(d999, DECIMALS), "delta_p999_lo_ms": round(lo999, DECIMALS),
            "delta_p999_hi_ms": round(hi999, DECIMALS),
            "background_median_cores": round(float(np.nanmedian([r.background_cores() for r in group])), DECIMALS),
            "original_replicates": len(original),
            "original_delta_p99_ms": round(float(np.nanmedian([r.percentile_ms("p99") for r in original])
                                                  - np.nanmedian(base99)), DECIMALS) if original else "",
            "original_delta_p999_ms": round(float(np.nanmedian([r.percentile_ms("p999") for r in original])
                                                   - np.nanmedian(base999)), DECIMALS) if original else "",
        })
    return rows


def phase5_rows(runs: List[RunData], rng: np.random.Generator) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, str, int], List[RunData]] = defaultdict(list)
    for run in runs:
        match = LEVEL_PATTERN.search(str(run.run["run_id"]))
        groups[(str(run.run["fault"]), str(run.run["controller"]), int(match.group(1)))].append(run)
    rows: List[Dict[str, object]] = []
    for (fault, controller, level), group in sorted(groups.items()):
        metrics = [goodput_for_run(r) for r in group]
        original = [goodput_for_run(r) for r in group if replicate_of(r) <= ORIGINAL_MAX_REPLICATE[5]]
        row: Dict[str, object] = {
            "fault": fault, "controller": controller, "level": level,
            "rate_rps": round(float(np.nanmedian([r.rate_rps for r in group])), DECIMALS),
            "replicates": len(group),
        }
        for index, name in enumerate(("goodput", "success", "p999_ok")):
            point, lower, upper = median_interval([m[index] for m in metrics], rng)
            row[f"{name}_median"] = round(point, DECIMALS)
            row[f"{name}_lo"] = round(lower, DECIMALS)
            row[f"{name}_hi"] = round(upper, DECIMALS)
            row[f"{name}_original"] = round(float(np.nanmedian([m[index] for m in original])), DECIMALS) if original else ""
        rows.append(row)
    return rows


def write_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    phase4 = phase4_rows(load_runs(RESULTS_ROOT / "raw_traces" / "phase4"), rng)
    phase5 = phase5_rows(load_runs(RESULTS_ROOT / "raw_traces" / "phase5"), rng)
    write_rows(RESULTS_ROOT / "metrics" / "phase4_tracing_extended.csv", phase4)
    write_rows(RESULTS_ROOT / "metrics" / "phase5_summary_extended.csv", phase5)
    print(f"phase 4: {len(phase4)} configurations; phase 5: {len(phase5)} cells")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
