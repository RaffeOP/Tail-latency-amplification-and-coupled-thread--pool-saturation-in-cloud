"""Raw values for the Phase 5 full sweep and the Phase 1 utilization at each saturation point.

The bracket level of a shape is the upper load level that encloses its saturation horizon. Its root
utilization, host CPU share, and timeout fraction are averaged over the replicates at that level.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
HOST_CORES = 8
PERCENT = 100.0
LEVEL_DECIMALS = 3
MAX_TIMEOUT_FRACTION = 0.01
RUNTIME_SHAPES = ("chain_D8", "sg_D3_F4", "diamond_D5_W8")
RUNTIMES = ("sync", "aio", "hybrid")

SweepRow = Tuple[str, str, int, float, float, float, float]
KneeRow = Tuple[str, int, float, float, float, float]


def read_csv(path: Path) -> List[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def phase5_sweep() -> List[SweepRow]:
    rows = sorted(read_csv(METRICS_DIR / "phase5_summary.csv"),
                  key=lambda r: (r["fault"], r["controller"], int(r["level"])))
    return [
        (row["fault"], row["controller"], int(row["level"]), float(row["rate_rps"]),
         float(row["goodput_rps"]), float(row["success_fraction"]), float(row["p999_ok_ms"]))
        for row in rows
    ]


def knee_utilization() -> List[KneeRow]:
    shapes = read_csv(METRICS_DIR / "phase1_shapes.csv")
    runs = read_csv(METRICS_DIR / "phase1_runs.csv")
    result: List[KneeRow] = []
    for shape_row in shapes:
        shape = shape_row["shape"].split(" / ")[0]
        bracket_top = float(shape_row["theta_ci_upper_rps"])
        level_runs = [r for r in runs if r["shape"] == shape]
        levels = {round(float(r["rate_rps"]), LEVEL_DECIMALS) for r in level_runs}
        level = min(levels, key=lambda value: abs(value - bracket_top))
        at_level = [r for r in level_runs if round(float(r["rate_rps"]), LEVEL_DECIMALS) == level]

        def mean(key: str) -> float:
            return sum(float(r[key]) for r in at_level) / len(at_level)

        result.append((shape, int(shape_row["vertices"]), level, mean("rho_pool_root_measured"),
                       PERCENT * mean("node_cores") / HOST_CORES, PERCENT * mean("timeout_fraction")))
    return result


RuntimeCell = Tuple[str, str, List[float], List[float], List[bool]]


def runtime_matrix() -> List[RuntimeCell]:
    """Per shape and runtime: the shared loads, the median P99.9 at each load, and whether timeouts exceed 1%."""
    grouped: Dict[Tuple[str, str], Dict[float, List[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in read_csv(METRICS_DIR / "phase2_runs.csv"):
        grouped[(row["shape"], row["runtime"])][round(float(row["rate_rps"]), LEVEL_DECIMALS)].append(row)
    cells: List[RuntimeCell] = []
    for shape in RUNTIME_SHAPES:
        for runtime in RUNTIMES:
            levels = grouped[(shape, runtime)]
            loads = sorted(levels)
            p999 = [float(np.median([float(r["p999_ms"]) for r in levels[load]])) for load in loads]
            timeouts = [float(np.mean([float(r["timeout_fraction"]) for r in levels[load]])) for load in loads]
            flags = [value > MAX_TIMEOUT_FRACTION for value in timeouts]
            cells.append((shape, runtime, loads, p999, flags))
    return cells

