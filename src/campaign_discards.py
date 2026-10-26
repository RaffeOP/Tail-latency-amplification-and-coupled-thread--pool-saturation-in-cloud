"""Background load of kept and discarded runs in every phase, read from each run's host telemetry.

A run is discarded when its background load, the host busy time minus the mesh's own CPU, averages
more than one core over the run. Kept runs are listed in each phase's manifest. Discarded attempts
are counted from the campaign log, because a repeated attempt replaces the earlier discarded folder.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pyarrow.parquet as pq

from host_gate import background_cores

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = PROJECT_ROOT / "results" / "raw_traces"
PHASES = (1, 2, 3, 4, 5)
CONTAMINATED_SUFFIX = ".contaminated"
DISCARD_MARKER = " contaminated "
TELEMETRY_FILE = "telemetry.parquet"
NO_VALUE = float("nan")


@dataclass(frozen=True)
class PhaseDiscards:
    phase: int
    kept: int
    discarded_attempts: int
    kept_background_median: float
    kept_background_max: float
    discarded_background_median: float
    discarded_background_min: float


def run_background(folder: Path) -> float:
    rows = pq.read_table(folder / TELEMETRY_FILE).to_pylist()
    return background_cores(rows)


def kept_run_ids(root: Path) -> List[str]:
    lines = (root / "manifest.jsonl").read_text().splitlines()
    return [json.loads(line)["run_id"] for line in lines if line.strip()]


def discarded_folders(root: Path) -> List[Path]:
    folders = sorted(root.glob(f"*{CONTAMINATED_SUFFIX}"))
    return [folder for folder in folders if (folder / TELEMETRY_FILE).exists()]


def phase_summary(phase: int) -> PhaseDiscards:
    root = RAW_ROOT / f"phase{phase}"
    kept = [run_background(root / run_id) for run_id in kept_run_ids(root)]
    discarded = [run_background(folder) for folder in discarded_folders(root)]
    log_lines = (root / "campaign.log").read_text().splitlines()
    attempts = sum(1 for line in log_lines if DISCARD_MARKER in line)
    return PhaseDiscards(
        phase=phase,
        kept=len(kept),
        discarded_attempts=attempts,
        kept_background_median=float(np.median(kept)) if kept else NO_VALUE,
        kept_background_max=float(np.max(kept)) if kept else NO_VALUE,
        discarded_background_median=float(np.median(discarded)) if discarded else NO_VALUE,
        discarded_background_min=float(np.min(discarded)) if discarded else NO_VALUE,
    )


def discard_summaries() -> List[PhaseDiscards]:
    return [phase_summary(phase) for phase in PHASES if (RAW_ROOT / f"phase{phase}" / "manifest.jsonl").exists()]
