"""Builds the LaTeX result tables (booktabs) from the analysis CSVs and the campaign manifests.

tab01: campaign inventory (planned and completed runs per shape and runtime).
tab02: amplification factors and saturation horizon brackets (Phases 1 and 2).
tab03: generalized Pareto tail parameters with 95% bootstrap intervals (Phases 1 and 2).
tab04: overload-control comparison (Phase 5).
tab05: tracing overhead against matched baselines (Phase 4).
tab06: serialization and sidecar shares and per-hop edge overhead (Phase 3).
tab07: runtime comparison on three shapes (Phase 2).
tab08: analytical model against measured P99.9 at stable load levels (Phase 1).
tab09: background load of kept and discarded runs by phase.
tab10: full Phase 5 sweep. tab11: utilization at each Phase 1 saturation point.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

from campaign_discards import discard_summaries
from campaign_matrix import phase_runs
from model_agreement import model_agreement_summary
from storage_manifest import completed_run_ids
from table_extras import knee_utilization, phase5_sweep, runtime_matrix
from table_format import (
    fmt, interval, latex_escape, read_rows, signed, split_shape_runtime, write_table,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
RAW_ROOT = PROJECT_ROOT / "results" / "raw_traces"
TABLES_DIR = PROJECT_ROOT / "tables"
DIGITS_RATE = 1
DIGITS_RATIO = 2
DIGITS_TAIL = 3
DIGITS_MS = 1
PHASES_WITH_DATA = (1, 2)
OVERLOAD_MID_LEVEL = 3
OVERLOAD_HIGH_LEVEL = 5
US_PER_MS = 1000.0


def inventory_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for phase in PHASES_WITH_DATA:
        done = completed_run_ids(RAW_ROOT / f"phase{phase}" / "manifest.jsonl")
        planned = phase_runs(phase)
        counts: Counter = Counter()
        planned_by_key: Counter = Counter()
        for run in planned:
            key = (str(run.shape), run.runtime)
            planned_by_key[key] += 1
            if run.run_id in done:
                counts[key] += 1
        for (shape, runtime), total in sorted(planned_by_key.items()):
            rows.append([str(phase), latex_escape(shape), runtime, f"{counts[(shape, runtime)]}/{total}"])
    return rows


def amplification_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for tag in ("phase1",):
        for row in read_rows(METRICS_DIR / f"{tag}_shapes.csv"):
            shape, runtime = split_shape_runtime(row["shape"])
            theta = interval(float(row["theta_star_rps"]), float(row["theta_ci_lower_rps"]),
                             float(row["theta_ci_upper_rps"]), DIGITS_RATE)
            rows.append([
                shape, runtime, row["vertices"],
                fmt(float(row["gamma_crit_p999"]), DIGITS_RATIO),
                fmt(float(row["gamma_load_p999"]), DIGITS_RATIO),
                theta,
            ])
    return rows


def tail_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for tag in ("phase1",):
        for row in read_rows(METRICS_DIR / f"{tag}_shapes.csv"):
            shape, runtime = split_shape_runtime(row["shape"])
            xi = interval(float(row["gpd_xi"]), float(row["gpd_xi_lower"]), float(row["gpd_xi_upper"]), DIGITS_RATIO)
            rows.append([shape, runtime, row["gpd_excesses"], xi, fmt(float(row["gpd_sigma_ms"]), DIGITS_TAIL)])
    return rows


def overload_rows() -> List[List[str]]:
    by_key: Dict[Tuple[str, str], Dict[int, Dict[str, str]]] = {}
    for row in read_rows(METRICS_DIR / "phase5_summary.csv"):
        by_key.setdefault((row["fault"], row["controller"]), {})[int(row["level"])] = row
    rows: List[List[str]] = []
    for (fault, controller), levels in sorted(by_key.items()):
        mid = levels.get(OVERLOAD_MID_LEVEL)
        top = levels.get(OVERLOAD_HIGH_LEVEL)
        if mid is None:
            continue
        rows.append([
            latex_escape(fault), latex_escape(controller),
            fmt(float(mid["goodput_rps"]), DIGITS_RATE),
            fmt(float(mid["success_fraction"]), DIGITS_RATIO),
            fmt(float(mid["p999_ok_ms"]), DIGITS_RATE),
            fmt(float(top["goodput_rps"]), DIGITS_RATE) if top is not None else "--",
        ])
    return rows


def tracing_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for row in read_rows(METRICS_DIR / "phase4_tracing.csv"):
        rows.append([
            latex_escape(row["exporter"]),
            fmt(float(row["tracing_rate"]), DIGITS_RATIO),
            fmt(float(row["rate_rps"]), DIGITS_RATE),
            fmt(float(row["p99_ms"]), DIGITS_MS),
            signed(float(row["delta_p99_ms"]), DIGITS_MS),
            fmt(float(row["p999_ms"]), DIGITS_MS),
            signed(float(row["delta_p999_ms"]), DIGITS_MS),
            fmt(float(row["background_median_cores"]), DIGITS_RATIO),
        ])
    return rows


def tax_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    for row in read_rows(METRICS_DIR / "phase3_tax.csv"):
        rows.append([
            row["depth"],
            row["payload_bytes"],
            fmt(100.0 * float(row["serialization_share"]), DIGITS_RATIO) + r"\%",
            fmt(100.0 * float(row["proxy_share"]), DIGITS_RATE) + r"\%",
            fmt(float(row["median_edge_overhead_us"]) / US_PER_MS, DIGITS_RATIO),
            fmt(float(row["p99_ms"]), DIGITS_MS),
        ])
    return rows


def runtime_rows() -> List[List[str]]:
    rows: List[List[str]] = []
    current = ""
    for shape, runtime, loads, p999, flags in runtime_matrix():
        if shape != current:
            current = shape
            rows.append([latex_escape(shape), "load (req/s)"] + [fmt(load, DIGITS_RATE) for load in loads])
        cells = [f"{fmt(value, DIGITS_MS)}$^{{*}}$" if flag else fmt(value, DIGITS_MS)
                 for value, flag in zip(p999, flags)]
        rows.append(["", runtime] + cells)
    return rows


def model_agreement_rows() -> List[List[str]]:
    summary = model_agreement_summary()
    return [
        [latex_escape(shape), str(vertices), str(levels), fmt(median, DIGITS_RATIO),
         f"{fmt(low, DIGITS_RATIO)}--{fmt(high, DIGITS_RATIO)}"]
        for shape, (vertices, levels, median, low, high) in sorted(summary.items())
    ]


def discard_rows() -> List[List[str]]:
    return [
        [str(item.phase), str(item.kept), str(item.discarded_attempts),
         fmt(item.kept_background_median, DIGITS_RATIO), fmt(item.kept_background_max, DIGITS_RATIO),
         fmt(item.discarded_background_median, DIGITS_RATIO), fmt(item.discarded_background_min, DIGITS_RATIO)]
        for item in discard_summaries()
    ]


def sweep_rows() -> List[List[str]]:
    return [
        [latex_escape(fault), latex_escape(controller), str(level), fmt(rate, DIGITS_RATE),
         fmt(goodput, DIGITS_RATE), fmt(success, DIGITS_RATIO), fmt(p999, DIGITS_RATE)]
        for fault, controller, level, rate, goodput, success, p999 in phase5_sweep()
    ]


def knee_rows() -> List[List[str]]:
    return [
        [latex_escape(shape), str(vertices), fmt(level, DIGITS_RATE), fmt(utilization, DIGITS_RATIO),
         fmt(cpu, DIGITS_RATE), fmt(timeouts, DIGITS_RATE)]
        for shape, vertices, level, utilization, cpu, timeouts in knee_utilization()
    ]


def build_all() -> List[Path]:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    outputs = [
        (TABLES_DIR / "tab01.tex", "Campaign inventory: completed and planned runs per shape and runtime model.",
         "tab:inventory", ["Phase", "Shape", "Runtime", "Completed/planned"], inventory_rows()),
        (TABLES_DIR / "tab02.tex",
         r"Phase 1 shapes (synchronous runtime): amplification factors at the highest stable load, and saturation horizons $\Theta^*$ (req/s), each with the two load levels that enclose it in brackets.",
         "tab:amplification",
         ["Shape", "Runtime", "$|V|$", r"$\Gamma_{\mathrm{crit}}$ (P99.9)", r"$\Gamma_{\mathrm{load}}$ (P99.9)", r"$\Theta^*$"],
         amplification_rows()),
        (TABLES_DIR / "tab03.tex",
         r"Phase 1 shapes: generalized Pareto tail parameters above the 95th percentile at the highest stable load, with 95\% bootstrap intervals for $\xi$.",
         "tab:tails", ["Shape", "Runtime", "Excesses", r"$\xi$ [95\% CI]", r"$\sigma$ (ms)"], tail_rows()),
        (TABLES_DIR / "tab04.tex",
         r"Overload control under injected faults on a four-vertex chain: goodput within a 250\,ms SLA at load levels L3 and L5 (req/s), success fraction and P99.9 of successful requests at L3.",
         "tab:overload", ["Fault", "Controller", "Goodput L3", "Success L3", "P99.9 OK L3 (ms)", "Goodput L5"],
         overload_rows()),
        (TABLES_DIR / "tab05.tex",
         r"Tracing overhead on a four-vertex chain at two loads, as medians across replicates (three per traced configuration, six for the rate-zero baseline). Deltas are traced minus the baseline at the same load. Background is the median host background of the runs.",
         "tab:tracing", ["Exporter", "Sampling", "Load (req/s)", "P99 (ms)", r"$\Delta$P99 (ms)", "P99.9 (ms)", r"$\Delta$P99.9 (ms)", "Background (cores)"],
         tracing_rows()),
        (TABLES_DIR / "tab06.tex",
         r"Medians across five replicates per cell at 30\,req/s: serialization and sidecar shares of CPU per request, median per-hop edge overhead, and P99 latency.",
         "tab:tax", ["Depth", "Payload (B)", "Serialization", "Sidecar", "Edge overhead (ms)", "P99 (ms)"],
         tax_rows()),
        (TABLES_DIR / "tab07.tex",
         r"Runtime comparison at the same five loads for three shapes: median P99.9 (ms) across three replicates. Load rows give the offered rate of each level. Asterisks mark levels where more than 1\% of requests timed out.",
         "tab:runtimes", ["Shape", "Runtime / load", "L1", "L2", "L3", "L4", "L5"],
         runtime_rows()),
        (TABLES_DIR / "tab08.tex",
         r"Measured P99.9 divided by the analytical root quantile at each stable Phase 1 load level, per shape. Values below one mean that the model over-predicts.",
         "tab:model", ["Shape", "$|V|$", "Stable levels", "Median ratio", "Range"],
         model_agreement_rows()),
        (TABLES_DIR / "tab09.tex",
         r"Background load of kept and discarded runs by phase, in cores, from each run's host telemetry. A run is discarded when its background load averages more than one core. Discarded attempts are counted from the campaign log.",
         "tab:discards", ["Phase", "Kept runs", "Discarded attempts", "Kept median", "Kept max", "Discarded median", "Discarded min"],
         discard_rows()),
        (TABLES_DIR / "tab10.tex",
         r"Full Phase 5 sweep: goodput within the 250\,ms SLA, success fraction, and P99.9 of successful requests for each fault, controller, and load level. Each cell averages two runs.",
         "tab:sweep", ["Fault", "Controller", "Level", "Load (req/s)", "Goodput (req/s)", "Success", "P99.9 OK (ms)"],
         sweep_rows()),
        (TABLES_DIR / "tab11.tex",
         r"Root pool utilization, host CPU, and timeouts at the upper bracket level of each Phase 1 shape. Root utilization above 0.9 lies beyond the stability cutoff.",
         "tab:knee", ["Shape", "$|V|$", "Level (req/s)", "Root utilization", r"Host CPU (\%)", r"Timeouts (\%)"],
         knee_rows()),
    ]
    written = []
    for path, caption, label, header, rows in outputs:
        if rows:
            write_table(path, caption, label, header, rows)
            written.append(path)
    return written


def main(argv: List[str]) -> int:
    written = build_all()
    for path in written:
        print(f"wrote {path}")
    if not written:
        print("no analysis outputs yet; run analysis_phase1.py for phase1 and phase2 first")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
