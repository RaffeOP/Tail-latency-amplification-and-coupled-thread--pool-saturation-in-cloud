"""Compares the analytical root P99.9 with the measured P99.9 at every stable Phase 1 load level.

A level is stable under the same rule as the tail analysis: mean root pool utilization below 0.9 and
mean timeout fraction at most 1%, both averaged over the replicates at that level.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from analysis_phase1 import MODEL_PARAMS
from analytical_dag_model import predict
from dag_topology_builder import build_topology

STABLE_RHO_POOL = 0.9
MAX_TIMEOUT_FRACTION = 0.01
PREDICTION_PROB = 0.999
LEVEL_DECIMALS = 3
PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
PHASE1_RAW = PROJECT_ROOT / "results" / "raw_traces" / "phase1"


def shape_graph(run_id: str):
    meta = json.loads((PHASE1_RAW / run_id / "meta.json").read_text())
    return build_topology(str(meta["run"]["topology"]), dict(meta["run"]["topology_args"]))


def stable_level_samples(rows: List[Dict[str, str]]) -> Dict[Tuple[str, float], List[float]]:
    grouped: Dict[Tuple[str, float], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["shape"], round(float(row["rate_rps"]), LEVEL_DECIMALS))].append(row)
    stable: Dict[Tuple[str, float], List[float]] = {}
    for key, level_rows in grouped.items():
        utilization = np.mean([float(r["rho_pool_root_measured"]) for r in level_rows])
        timeouts = np.mean([float(r["timeout_fraction"]) for r in level_rows])
        if utilization < STABLE_RHO_POOL and timeouts <= MAX_TIMEOUT_FRACTION:
            stable[key] = [float(r["p999_ms"]) for r in level_rows]
    return stable


def model_agreement_summary() -> Dict[str, Tuple[int, int, float, float, float]]:
    """Per shape: vertices, stable levels, and the median, minimum, and maximum of measured over predicted P99.9."""
    with (METRICS_DIR / "phase1_runs.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    first_run: Dict[str, str] = {}
    for row in rows:
        first_run.setdefault(row["shape"], row["run_id"])
    graphs = {shape: shape_graph(run_id) for shape, run_id in first_run.items()}
    ratios: Dict[str, List[float]] = defaultdict(list)
    for (shape, rate), samples in sorted(stable_level_samples(rows).items()):
        predicted = predict(graphs[shape], rate, MODEL_PARAMS).root_quantiles_ms[PREDICTION_PROB]
        ratios[shape].append(float(np.mean(samples)) / predicted)
    return {
        shape: (graphs[shape].number_of_nodes(), len(values), float(np.median(values)), min(values), max(values))
        for shape, values in ratios.items()
    }
