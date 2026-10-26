"""Paper figures built from stored run data: vector PDF, Type 42 fonts, journal column widths.

Figures 7 and 8 need Phase 3 (serialization and sidecar) and Phase 5 (overload control) data.
They are reported as not generated until those runs exist; no placeholder figures are drawn.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from analysis_phase1 import MODEL_PARAMS, ShapeResult, analyze_shape, group_by_level, group_by_shape, model_prediction_ms  # noqa: E402
from analytical_dag_model import predict  # noqa: E402
from dag_topology_builder import build_topology  # noqa: E402
from plot_style import (  # noqa: E402
    COLUMN_WIDTH_IN, FULL_WIDTH_IN, INK_SECONDARY, OTHER_COLOR, SERIES_COLORS, SURFACE_COLOR,
    apply_journal_style, series_style,
)
from result_loader import RunData, load_runs  # noqa: E402

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
FAMILY_PREFIXES = (("chain", "chain"), ("sg_", "scatter_gather"), ("diamond", "diamond"), ("deathstar", "deathstar"))
FAMILY_ORDER = ("chain", "scatter_gather", "diamond", "deathstar")
HOST_CORES = 8
PROB_P999 = 0.999
# One representative shape per call-graph family, for the tail-versus-load overlay.
REPRESENTATIVE_SHAPES_FIG2 = ("chain_D8", "sg_D3_F4", "diamond_D5_W8", "deathstar_social_12")
MIN_EXCESSES_FOR_QQ = 30
MODEL_MAX_UTILIZATION = 0.95
HORIZON_FIG_HEIGHT_IN = 3.4


def family_of(shape: str) -> str:
    for prefix, family in FAMILY_PREFIXES:
        if shape.startswith(prefix):
            return family
    return "other"


def finish_axes(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_facecolor(SURFACE_COLOR)


def fig01_chain_amplification(results: List[ShapeResult], out: Path) -> bool:
    chains = sorted((r for r in results if r.topology == "chain"), key=lambda r: r.vertices)
    if len(chains) < 2:
        return False
    depths = [r.vertices for r in chains]
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ax.plot(depths, [r.gamma_crit_p999 for r in chains], label="Γ_crit (P99.9)", **series_style(0))
    ax.plot(depths, [r.gamma_load_p999 for r in chains], label="Γ_load (P99.9)", **series_style(1))
    ax.legend(loc="upper right")
    finish_axes(ax, "Chain depth D (vertices)", "Amplification factor")
    ax.set_xticks(depths)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def stable_model_curve(graph, rates: List[float]) -> tuple:
    """Model values only where the bottleneck pool utilization stays below MODEL_MAX_UTILIZATION."""
    kept = [rate for rate in rates if predict(graph, rate, MODEL_PARAMS).max_rho_pool < MODEL_MAX_UTILIZATION]
    return kept, [model_prediction_ms(graph, rate, PROB_P999) for rate in kept]


def fig02_tail_vs_load(runs: List[RunData], out: Path) -> bool:
    shapes = [s for s in REPRESENTATIVE_SHAPES_FIG2 if any(r.run["shape"] == s for r in runs)]
    if not shapes:
        return False
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.8))
    for index, shape in enumerate(shapes):
        shape_runs = [r for r in runs if r.run["shape"] == shape]
        graph = build_topology(str(shape_runs[0].run["topology"]), dict(shape_runs[0].run["topology_args"]))
        levels = group_by_level(shape_runs)
        rates = np.array(list(levels.keys()))
        per_level = list(levels.values())
        means = np.array([np.mean([r.percentile_ms("p999") for r in lvl]) for lvl in per_level])
        lows = np.array([np.min([r.percentile_ms("p999") for r in lvl]) for lvl in per_level])
        highs = np.array([np.max([r.percentile_ms("p999") for r in lvl]) for lvl in per_level])
        style = series_style(index, solid=True)
        ax.errorbar(rates, means, yerr=[means - lows, highs - means], label=shape, capsize=2, **style)
        model_rates, model_ms = stable_model_curve(graph, list(rates))
        if model_rates:
            ax.plot(model_rates, model_ms, color=style["color"], linestyle=":", linewidth=1.0)
    ax.set_yscale("log")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, title="measured (solid), model (dotted)")
    finish_axes(ax, "Arrival rate λ (req/s)", "P99.9 latency (ms)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def fig03_pool_vs_cpu(runs: List[RunData], out: Path) -> bool:
    if not runs:
        return False
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    for index, family in enumerate(FAMILY_ORDER):
        subset = [r for r in runs if family_of(str(r.run["shape"])) == family]
        if subset:
            ax.scatter([r.node_cores() / HOST_CORES for r in subset],
                       [r.measured_root_pool_utilization() for r in subset],
                       s=14, color=SERIES_COLORS[index], label=family, edgecolor=SURFACE_COLOR, linewidth=0.4)
    ax.axhline(1.0, color=INK_SECONDARY, linewidth=0.6, linestyle="--")
    ax.legend(loc="lower right")
    finish_axes(ax, "Measured CPU utilization of mesh processes", "Root pool utilization ρ_pool")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def fig04_saturation_horizon(results: List[ShapeResult], out: Path) -> bool:
    estimated = [r for r in results if np.isfinite(r.theta_star_rps)]
    if not estimated:
        return False
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, HORIZON_FIG_HEIGHT_IN))
    y = np.arange(len(estimated))
    theta = np.array([r.theta_star_rps for r in estimated])
    lower = theta - np.array([r.theta_ci_lower_rps for r in estimated])
    upper = np.array([r.theta_ci_upper_rps for r in estimated]) - theta
    ax.errorbar(theta, y, xerr=[lower, upper], fmt="o", color=SERIES_COLORS[0], capsize=2,
                markeredgecolor=SURFACE_COLOR)
    ax.set_yticks(y)
    ax.set_yticklabels([r.shape for r in estimated])
    ax.invert_yaxis()
    finish_axes(ax, "Saturation horizon Θ* (req/s)", "Shape")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def fig05_gpd_diagnostic(runs: List[RunData], out: Path) -> bool:
    candidates = [r for r in runs if r.measured_latency_ms().size > 0]
    if not candidates:
        return False
    run = max(candidates, key=lambda r: r.measured_latency_ms().size)
    values = run.measured_latency_ms()
    threshold = np.quantile(values, 0.95)
    excess = np.sort(values[values > threshold] - threshold)
    if excess.size < MIN_EXCESSES_FOR_QQ:
        return False
    fit = stats.genpareto.fit(excess, floc=0.0)
    probs = (np.arange(1, excess.size + 1) - 0.5) / excess.size
    theoretical = stats.genpareto.ppf(probs, fit[0], loc=0.0, scale=fit[2])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.6))
    ax.scatter(theoretical, excess, s=6, color=SERIES_COLORS[0], edgecolor="none")
    limit = max(theoretical.max(), excess.max())
    ax.plot([0, limit], [0, limit], color=INK_SECONDARY, linewidth=0.6, linestyle="--")
    finish_axes(ax, "GPD quantile (ms)", "Empirical excess (ms)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def fig06_generator_lag(runs: List[RunData], out: Path) -> bool:
    if not runs:
        return False
    rates = np.array([r.rate_rps for r in runs])
    p50 = np.array([np.percentile(r.lag_ns[r.measured], 50) / 1e6 for r in runs])
    p99 = np.array([np.percentile(r.lag_ns[r.measured], 99) / 1e6 for r in runs])
    fig, ax = plt.subplots(figsize=(COLUMN_WIDTH_IN, 2.4))
    ax.scatter(rates, p50, s=10, color=SERIES_COLORS[0], label="median lag")
    ax.scatter(rates, p99, s=10, color=SERIES_COLORS[1], marker="s", label="P99 lag")
    ax.legend(loc="upper left")
    finish_axes(ax, "Arrival rate λ (req/s)", "Generator lag (ms)")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def main(argv: List[str]) -> int:
    apply_journal_style()
    phase_root = RESULTS_ROOT / (argv[0] if argv else "raw_traces/phase1")
    out_dir = Path(argv[1]) if len(argv) > 1 else RESULTS_ROOT / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(phase_root)
    shape_runs = group_by_shape(runs)
    results = [analyze_shape(shape, members) for shape, members in shape_runs.items()]
    produced = {
        "fig01.pdf": fig01_chain_amplification(results, out_dir / "fig01.pdf"),
        "fig02.pdf": fig02_tail_vs_load(runs, out_dir / "fig02.pdf"),
        "fig03.pdf": fig03_pool_vs_cpu(runs, out_dir / "fig03.pdf"),
        "fig04.pdf": fig04_saturation_horizon(results, out_dir / "fig04.pdf"),
        "fig05.pdf": fig05_gpd_diagnostic(runs, out_dir / "fig05.pdf"),
        "fig06.pdf": fig06_generator_lag(runs, out_dir / "fig06.pdf"),
    }
    for name, done in produced.items():
        print(f"{name}: {'written' if done else 'not generated (insufficient data)'}")
    print("fig07.pdf: not generated (requires Phase 3 serialization and sidecar runs)")
    print("fig08.pdf: not generated (requires Phase 5 overload-control runs)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
