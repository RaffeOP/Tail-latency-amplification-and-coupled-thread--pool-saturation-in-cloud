"""Phase 1 analysis: per-level tails, amplification factors, saturation horizons, and GPD tails.

Writes CSV summaries to results/metrics/. Every number is computed from the stored run folders.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import networkx as nx
import numpy as np

from analytical_dag_model import ModelParams, predict
from dag_topology_builder import build_topology
from result_loader import RunData, load_runs
from tail_cascade_analyzer import (
    bootstrap_gpd_xi,
    bootstrap_horizon,
    fit_gpd_tail,
    gamma_crit,
    gamma_load,
)

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
MIN_LEVELS_FOR_HORIZON = 5  # the penalized spline needs five points; the design has exactly five levels
STABLE_RHO_POOL = 0.9
MAX_TIMEOUT_FRACTION = 0.01
LEVEL_DECIMALS = 3
ROOT_VERTEX = 0
MODEL_PARAMS = ModelParams(
    pool_size=8,
    service_mean_ms=10.0,
    service_sigma=0.5,
    net_hop_ms=0.5,
    cpu_per_visit_ms=1.5,
    host_cores=8,
)


@dataclass
class ShapeResult:
    shape: str
    topology: str
    vertices: int
    levels_rps: List[float] = field(default_factory=list)
    level_p999_ms: List[List[float]] = field(default_factory=list)
    gamma_crit_p999: float = float("nan")
    gamma_load_p999: float = float("nan")
    theta_star_rps: float = float("nan")
    theta_ci_lower_rps: float = float("nan")
    theta_ci_upper_rps: float = float("nan")
    theta_stability: float = float("nan")
    gpd_xi: float = float("nan")
    gpd_xi_lower: float = float("nan")
    gpd_xi_upper: float = float("nan")
    gpd_sigma_ms: float = float("nan")
    gpd_excesses: int = 0


def group_key(run: RunData) -> str:
    """Shape and runtime model together, so that Phase 2 runtimes are never pooled."""
    return f"{run.run['shape']} / {run.run['runtime']}"


def group_by_shape(runs: List[RunData]) -> Dict[str, List[RunData]]:
    groups: Dict[str, List[RunData]] = defaultdict(list)
    for run in runs:
        groups[group_key(run)].append(run)
    return groups


def group_by_level(shape_runs: List[RunData]) -> Dict[float, List[RunData]]:
    levels: Dict[float, List[RunData]] = defaultdict(list)
    for run in shape_runs:
        levels[round(run.rate_rps, LEVEL_DECIMALS)].append(run)
    return dict(sorted(levels.items()))


def critical_path_median_ms(graph: nx.DiGraph, medians_ms: Dict[int, float]) -> float:
    """Longest root-to-leaf path where each vertex contributes its median service time."""
    longest: Dict[int, float] = {}
    for node in reversed(list(nx.topological_sort(graph))):
        children = list(graph.successors(node))
        longest[node] = medians_ms[node] + max((longest[child] for child in children), default=0.0)
    return longest[ROOT_VERTEX]


def baseline_medians_ms(graph: nx.DiGraph, base_runs: List[RunData]) -> Dict[int, float]:
    medians: Dict[int, float] = {}
    for vertex in graph.nodes():
        samples = np.concatenate([run.vertex_business_ms(vertex) for run in base_runs])
        medians[vertex] = float(np.median(samples))
    return medians


def pooled_excesses_ms(top_runs: List[RunData]) -> np.ndarray:
    return np.concatenate([run.measured_latency_ms() for run in top_runs])


def stable_top_level(levels: Dict[float, List[RunData]]) -> float:
    """Highest load level whose runs stay unsaturated; overloaded tails are not stationary."""
    stable = list(levels.keys())[0]
    for level, level_runs in levels.items():
        utilization = np.mean([run.measured_root_pool_utilization() for run in level_runs])
        timeouts = np.mean([run.timeout_fraction() for run in level_runs])
        if utilization < STABLE_RHO_POOL and timeouts <= MAX_TIMEOUT_FRACTION:
            stable = level
    return stable


def analyze_shape(shape: str, shape_runs: List[RunData]) -> ShapeResult:
    first = shape_runs[0]
    graph = build_topology(str(first.run["topology"]), dict(first.run["topology_args"]))
    result = ShapeResult(shape=shape, topology=str(first.run["topology"]), vertices=graph.number_of_nodes())
    levels = group_by_level(shape_runs)
    result.levels_rps = list(levels.keys())
    result.level_p999_ms = [[run.percentile_ms("p999") for run in level_runs] for level_runs in levels.values()]
    base_runs = levels[result.levels_rps[0]]
    top_runs = levels[stable_top_level(levels)]
    medians = baseline_medians_ms(graph, base_runs)
    critical_ms = critical_path_median_ms(graph, medians)
    top_p999 = float(np.mean([run.percentile_ms("p999") for run in top_runs]))
    base_p999 = float(np.mean([run.percentile_ms("p999") for run in base_runs]))
    result.gamma_crit_p999 = gamma_crit(top_p999, critical_ms)
    result.gamma_load_p999 = gamma_load(top_p999, base_p999)
    if len(result.levels_rps) >= MIN_LEVELS_FOR_HORIZON:
        replicate_samples = [[run.measured_latency_ms() for run in level_runs] for level_runs in levels.values()]
        horizon = bootstrap_horizon(np.array(result.levels_rps), replicate_samples)
        result.theta_star_rps = horizon["theta_star_rps"]
        result.theta_ci_lower_rps = horizon["bracket_lower_rps"]
        result.theta_ci_upper_rps = horizon["bracket_upper_rps"]
        result.theta_stability = horizon["stability"]
    pooled = pooled_excesses_ms(top_runs)
    fit = fit_gpd_tail(pooled)
    result.gpd_excesses = int(fit["n_excess"])
    result.gpd_xi = fit["xi"]
    result.gpd_sigma_ms = fit["sigma_ms"]
    if result.gpd_excesses >= 30:
        result.gpd_xi_lower, result.gpd_xi_upper = bootstrap_gpd_xi(pooled)
    return result


def model_prediction_ms(graph: nx.DiGraph, rate_rps: float, prob: float) -> float:
    prediction = predict(graph, rate_rps, MODEL_PARAMS)
    return prediction.root_quantiles_ms[prob]


def write_csv(path: Path, header: List[str], rows: List[List[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)


def write_run_table(path: Path, runs: List[RunData]) -> None:
    rows = []
    for run in runs:
        rows.append([
            run.run_id, run.run["shape"], run.run["runtime"], round(run.rate_rps, LEVEL_DECIMALS), run.run["replicate"],
            round(run.throughput_rps(), 3), round(run.percentile_ms("p50"), 3), round(run.percentile_ms("p99"), 3),
            round(run.percentile_ms("p999"), 3), round(run.timeout_fraction(), 5),
            round(run.measured_root_pool_utilization(), 4), round(run.node_cores(), 3),
            round(run.background_cores(), 3), round(run.host_noise_p99_ms(), 3),
        ])
    write_csv(path, ["run_id", "shape", "runtime", "rate_rps", "replicate", "throughput_rps", "p50_ms", "p99_ms",
                     "p999_ms", "timeout_fraction", "rho_pool_root_measured", "node_cores", "background_cores",
                     "host_noise_p99_ms"], rows)


def write_shape_table(path: Path, results: List[ShapeResult]) -> None:
    rows = [[
        r.shape, r.topology, r.vertices, round(r.gamma_crit_p999, 4), round(r.gamma_load_p999, 4),
        round(r.theta_star_rps, 3), round(r.theta_ci_lower_rps, 3), round(r.theta_ci_upper_rps, 3),
        round(r.gpd_xi, 4), round(r.gpd_xi_lower, 4), round(r.gpd_xi_upper, 4), round(r.gpd_sigma_ms, 4),
        r.gpd_excesses, round(r.theta_stability, 4),
    ] for r in results]
    write_csv(path, ["shape", "topology", "vertices", "gamma_crit_p999", "gamma_load_p999", "theta_star_rps",
                     "theta_ci_lower_rps", "theta_ci_upper_rps", "gpd_xi", "gpd_xi_lower", "gpd_xi_upper",
                     "gpd_sigma_ms", "gpd_excesses", "theta_stability"], rows)


def main(argv: List[str]) -> int:
    phase_root = RESULTS_ROOT / (argv[0] if argv else "raw_traces/phase1")
    runs = load_runs(phase_root)
    if not runs:
        print(f"no completed runs under {phase_root}")
        return 1
    results = [analyze_shape(shape, shape_runs) for shape, shape_runs in group_by_shape(runs).items()]
    metrics = RESULTS_ROOT / "metrics"
    tag = phase_root.name
    write_run_table(metrics / f"{tag}_runs.csv", runs)
    write_shape_table(metrics / f"{tag}_shapes.csv", results)
    print(f"analyzed {len(runs)} runs across {len(results)} shapes into {metrics}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
