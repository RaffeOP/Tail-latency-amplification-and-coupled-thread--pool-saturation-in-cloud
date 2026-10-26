"""Figures 7 and 8, built from the Phase 3 and Phase 5 summary tables.

Figure 7: serialization and sidecar share of CPU across payload sizes, one line per chain depth.
Figure 8: goodput within the SLA against arrival rate, one panel per fault, one line per controller.
"""

import csv
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from plot_style import FULL_WIDTH_IN, apply_journal_style, series_style  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METRICS_DIR = PROJECT_ROOT / "results" / "metrics"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
PERCENT = 100.0
FAULTS = ("slowdown", "errors", "hang")
CONTROLLERS = ("static_cb", "naive_retry", "dacshed")


def read_rows(metrics_dir: Path, name: str) -> List[Dict[str, str]]:
    path = metrics_dir / name
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def fig07_tax(metrics_dir: Path, out: Path) -> bool:
    rows = read_rows(metrics_dir, "phase3_tax.csv")
    if not rows:
        return False
    fig, (left, right) = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 2.6))
    for index, depth in enumerate(sorted({int(row["depth"]) for row in rows})):
        subset = sorted((row for row in rows if int(row["depth"]) == depth), key=lambda row: int(row["payload_bytes"]))
        payload = [int(row["payload_bytes"]) for row in subset]
        style = series_style(index)
        left.plot(payload, [PERCENT * float(row["serialization_share"]) for row in subset], label=f"depth {depth}", **style)
        right.plot(payload, [PERCENT * float(row["proxy_share"]) for row in subset], label=f"depth {depth}", **style)
    for axis, ylabel in ((left, "Serialization share of CPU (%)"), (right, "Sidecar share of CPU (%)")):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Payload size (bytes)")
        axis.set_ylabel(ylabel)
        axis.margins(y=0.2)
        axis.legend(loc="upper center", ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def fig08_goodput(metrics_dir: Path, out: Path) -> bool:
    rows = read_rows(metrics_dir, "phase5_summary.csv")
    if not rows:
        return False
    fig, axes = plt.subplots(1, len(FAULTS), figsize=(FULL_WIDTH_IN, 2.5), sharey=True)
    for axis, fault in zip(axes, FAULTS):
        for index, controller in enumerate(CONTROLLERS):
            subset = sorted(
                (row for row in rows if row["fault"] == fault and row["controller"] == controller),
                key=lambda row: float(row["rate_rps"]),
            )
            if subset:
                axis.plot([float(row["rate_rps"]) for row in subset],
                          [float(row["goodput_rps"]) for row in subset], label=controller, **series_style(index))
        axis.set_title(fault)
        axis.set_xlabel("Arrival rate (req/s)")
    axes[0].set_ylabel("Goodput within SLA (req/s)")
    axes[0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    return True


def main(argv: List[str]) -> int:
    apply_journal_style()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    produced = {
        "fig07.pdf": fig07_tax(METRICS_DIR, FIGURES_DIR / "fig07.pdf"),
        "fig08.pdf": fig08_goodput(METRICS_DIR, FIGURES_DIR / "fig08.pdf"),
    }
    for name, done in produced.items():
        print(f"{name}: {'written' if done else 'not generated (no Phase 3 or Phase 5 summary yet)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
