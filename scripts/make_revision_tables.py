#!/usr/bin/env python3
"""Writes the tables added in the Springer revision directly from the metric CSVs.

Outputs (under tables/):
  tab12_gate_sensitivity.tex  strict versus relaxed quiet-host gate, Phase 1 shapes
  tab13_m1_summary.tex        M0, path-SCV and M1 ratios, by family and at the top level
  tab14_gate_folders.tex      contaminated folders retained on disk, by directory
"""
import csv
import statistics
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from host_gate import background_cores  # noqa: E402
from result_loader import load_run  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

METRICS = PROJECT_ROOT / "results" / "metrics"
TABLES = PROJECT_ROOT / "tables"
RAW = PROJECT_ROOT / "results" / "raw_traces"
CAMPAIGN_PHASES = ("phase1", "phase2", "phase3", "phase4", "phase5")
BORDER_LOW = 1.0
BORDER_HIGH = 1.2


def read_csv(path: Path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def shape_label(name: str) -> str:
    return name.replace("_", r"\_")


def write_gate_sensitivity() -> None:
    rows = read_csv(METRICS / "sensitivity_gate_phase1.csv")
    by_gate = defaultdict(dict)
    for row in rows:
        by_gate[float(row["gate_upper_cores"])][row["shape"]] = row
    strict, relaxed = by_gate[1.0], by_gate[1.2]
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Sensitivity of the Phase 1 results to the quiet-host gate. The strict gate admits manifest runs with background load of at most 1.0 core. The relaxed gate also admits contaminated attempts with background load in (1.0, 1.2] cores and re-runs the same analysis code. $\Theta^*$ brackets are unchanged for all 18 shapes. $\Gamma_{\mathrm{crit}}$ and stability are shown for the strict and relaxed gates.}",
        r"\label{tab:gate_sensitivity}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Shape & Runs (strict) & Runs (1.2) & $\Gamma_{\mathrm{crit}}$ (strict) & $\Gamma_{\mathrm{crit}}$ (1.2) & Stability (strict) & Stability (1.2) \\",
        r"\midrule",
    ]
    for shape in sorted(strict):
        s, r = strict[shape], relaxed[shape]
        lines.append(
            f"{shape_label(shape.split(' / ')[0])} & {s['runs']} & {r['runs']} & "
            f"{float(s['gamma_crit_p999']):.2f} & {float(r['gamma_crit_p999']):.2f} & "
            f"{float(s['theta_stability']):.3f} & {float(r['theta_stability']):.3f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "tab12_gate_sensitivity.tex").write_text("\n".join(lines))


def write_m1_summary() -> None:
    rows = read_csv(METRICS / "model_m1_ablation.csv")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Analytical diagnostics against measured P99.9 at the stable load levels. Each entry is the ratio of measured to predicted P99.9 (values below one mean over-prediction). Upper rows summarize all 60 shape-level pairs by call-graph family; lower rows give the highest stable level of the three shapes discussed in the text.}",
        r"\label{tab:m1_summary}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Group & Pairs & M0 median & Path-SCV median & M1 median \\",
        r"\midrule",
    ]
    families = sorted({row["family"] for row in rows})
    for family in families:
        sub = [row for row in rows if row["family"] == family]
        lines.append(
            f"{family} & {len(sub)} & {statistics.median(float(r['ratio_m0']) for r in sub):.2f} & "
            f"{statistics.median(float(r['ratio_path_scv']) for r in sub):.2f} & "
            f"{statistics.median(float(r['ratio_m1']) for r in sub):.2f} \\\\"
        )
    all_pairs = rows
    lines.append(
        f"All pairs & {len(all_pairs)} & {statistics.median(float(r['ratio_m0']) for r in all_pairs):.2f} & "
        f"{statistics.median(float(r['ratio_path_scv']) for r in all_pairs):.2f} & "
        f"{statistics.median(float(r['ratio_m1']) for r in all_pairs):.2f} \\\\"
    )
    lines += [r"\midrule", r"Shape (highest stable level) & Rate (req/s) & M0 & Path-SCV & M1 \\", r"\midrule"]
    for shape in ("diamond_D3_W8", "sg_D3_F4", "sg_D2_F2"):
        sub = [row for row in rows if row["shape"] == shape]
        if not sub:
            continue
        top = max(sub, key=lambda row: float(row["rate_rps"]))
        lines.append(
            f"{shape_label(shape)} & {float(top['rate_rps']):.1f} & {float(top['ratio_m0']):.2f} & "
            f"{float(top['ratio_path_scv']):.2f} & {float(top['ratio_m1']):.2f} \\\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "tab13_m1_summary.tex").write_text("\n".join(lines))


def contaminated_stats(folder: Path):
    backgrounds = []
    for run_dir in sorted(folder.glob("*.contaminated")):
        telemetry = run_dir / "telemetry.parquet"
        if telemetry.exists():
            backgrounds.append(background_cores(pq.read_table(telemetry).to_pylist()))
    return backgrounds


def write_gate_folders() -> None:
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\footnotesize",
        r"\caption{Contaminated run folders retained on disk, by campaign phase, with the number whose background load lies in (1.0, 1.2] cores. The campaign log records 472 discarded attempts in total (Table~\ref{tab:discards}); the folders below are those still present. The abandoned Phase 4 extension is listed separately and is not part of the campaign.}",
        r"\label{tab:gate_folders}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Directory & Folders & With telemetry & Background $\le 1.0$ core & In $(1.0, 1.2]$ cores \\",
        r"\midrule",
    ]
    totals = [0, 0, 0]
    for name in CAMPAIGN_PHASES + ("phase4_extension_abandoned",):
        folder = RAW / name
        if not folder.is_dir():
            continue
        backgrounds = contaminated_stats(folder)
        folders = len(list(folder.glob("*.contaminated")))
        border = sum(1 for value in backgrounds if BORDER_LOW < value <= BORDER_HIGH)
        below = sum(1 for value in backgrounds if value <= BORDER_LOW)
        if name in CAMPAIGN_PHASES:
            totals = [totals[0] + folders, totals[1] + len(backgrounds), totals[2] + border]
        label = name.replace("_", r"\_")
        lines.append(f"{label} & {folders} & {len(backgrounds)} & {below} & {border} \\\\")
    lines += [
        r"\midrule",
        f"Campaign phases & {totals[0]} & {totals[1]} & -- & {totals[2]} \\\\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
        "",
    ]
    (TABLES / "tab14_gate_folders.tex").write_text("\n".join(lines))


def main() -> int:
    write_gate_sensitivity()
    write_m1_summary()
    write_gate_folders()
    print("wrote tab12_gate_sensitivity.tex, tab13_m1_summary.tex, tab14_gate_folders.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
