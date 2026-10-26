"""Runs the five-replicate extension of Phases 4 and 5 under the existing campaign runner.

The extension raises the Phase 4 traced configurations from three to five replicates and every
Phase 5 cell from two to five. The runner executes only run IDs that are not yet in the phase
manifest, so the original runs are reused unchanged. Seeds and protocol are those of the campaign.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import sys
from typing import List

import campaign_matrix
import run_campaign
from mesh_config import RunSpec

EXTENDED_REPLICATES = {4: "PHASE4_REPLICATES", 5: "PHASE5_REPLICATES"}
EXTENDED_VALUE = 5
DEADLINE = "2026-11-01T23:59:00+05:30"


def extended_phase_runs(phase: int) -> List[RunSpec]:
    constant = EXTENDED_REPLICATES[phase]
    original = getattr(campaign_matrix, constant)
    setattr(campaign_matrix, constant, EXTENDED_VALUE)
    try:
        return campaign_matrix.phase_runs(phase)
    finally:
        setattr(campaign_matrix, constant, original)


def main(argv: List[str]) -> int:
    phase = int(argv[0])
    run_campaign.phase_runs = extended_phase_runs
    return run_campaign.main(["--phase", str(phase), "--mode", "full", "--deadline", DEADLINE])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
