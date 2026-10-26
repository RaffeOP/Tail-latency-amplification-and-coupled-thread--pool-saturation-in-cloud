#!/usr/bin/env python3
"""Re-runs the Phase 1 shape analysis with the quiet-host gate relaxed.

The strict gate admits only manifest-listed runs. This script adds every
contaminated attempt whose background load lies in (1.0, upper], re-derives the
per-shape metrics with the same analyze_shape code path, and compares them with
the strict-gate table. The strict configuration (upper = 1.0) is re-derived first
as a reproduction check against results/metrics/phase1_shapes.csv.
"""
import csv
import sys
from pathlib import Path
from typing import Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis_phase1 import analyze_shape, group_by_shape  # noqa: E402
from result_loader import RunData, load_run, load_runs  # noqa: E402

PHASE_ROOT = PROJECT_ROOT / "results" / "raw_traces" / "phase1"
OUTPUT_CSV = PROJECT_ROOT / "results" / "metrics" / "sensitivity_gate_phase1.csv"
STRICT_CSV = PROJECT_ROOT / "results" / "metrics" / "phase1_shapes.csv"
GATE_UPPER_LIMITS = (1.0, 1.2, 1.3)
CONTAMINATED_SUFFIX = ".contaminated"
HEADER = [
    "gate_upper_cores", "shape", "vertices", "runs", "gamma_crit_p999", "gamma_load_p999",
    "theta_star_rps", "theta_ci_lower_rps", "theta_ci_upper_rps", "theta_stability",
    "gpd_xi", "gpd_excesses",
]


def contaminated_runs_within(upper_cores: float) -> List[RunData]:
    admitted: List[RunData] = []
    for folder in sorted(PHASE_ROOT.glob(f"*{CONTAMINATED_SUFFIX}")):
        run = load_run(folder)
        background = run.background_cores()
        if 1.0 < background <= upper_cores:
            admitted.append(run)
    return admitted


def analyze_gate(upper_cores: float) -> List[Dict[str, object]]:
    runs = load_runs(PHASE_ROOT)
    if upper_cores > 1.0:
        runs = runs + contaminated_runs_within(upper_cores)
    rows = []
    for shape, shape_runs in sorted(group_by_shape(runs).items()):
        result = analyze_shape(shape, shape_runs)
        rows.append({
            "gate_upper_cores": upper_cores,
            "shape": shape,
            "vertices": result.vertices,
            "runs": len(shape_runs),
            "gamma_crit_p999": round(result.gamma_crit_p999, 4),
            "gamma_load_p999": round(result.gamma_load_p999, 4),
            "theta_star_rps": round(result.theta_star_rps, 3),
            "theta_ci_lower_rps": round(result.theta_ci_lower_rps, 3),
            "theta_ci_upper_rps": round(result.theta_ci_upper_rps, 3),
            "theta_stability": round(result.theta_stability, 3),
            "gpd_xi": round(result.gpd_xi, 4),
            "gpd_excesses": result.gpd_excesses,
        })
    return rows


def strict_reference() -> Dict[str, Dict[str, str]]:
    with STRICT_CSV.open() as handle:
        return {row["shape"]: row for row in csv.DictReader(handle)}


def main() -> int:
    all_rows: List[Dict[str, object]] = []
    for upper in GATE_UPPER_LIMITS:
        print(f"analyzing gate upper limit {upper:.2f} cores", flush=True)
        all_rows.extend(analyze_gate(upper))
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER)
        writer.writeheader()
        writer.writerows(all_rows)
    reference = strict_reference()
    strict_rows = [row for row in all_rows if row["gate_upper_cores"] == 1.0]
    worst_theta_gap = 0.0
    for row in strict_rows:
        reference_row = reference.get(str(row["shape"]))
        if reference_row is None:
            print(f"reproduction: no reference row for {row['shape']}")
            continue
        gap = abs(float(row["theta_star_rps"]) - float(reference_row["theta_star_rps"]))
        worst_theta_gap = max(worst_theta_gap, gap)
    print(f"strict-gate reproduction: largest |Theta* difference| vs phase1_shapes.csv = {worst_theta_gap:.3f} req/s")
    print(f"wrote {OUTPUT_CSV} ({len(all_rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
