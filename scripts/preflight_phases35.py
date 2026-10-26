"""Preflight for Phases 3 to 5: one short run per feature path, on the project source.

Exit status 0 means every run finished with ok or contaminated (the host may be busy; the feature
still ran). A failed run exits 1, which stops the campaign chain before the 225 full runs start.
"""

import sys
from dataclasses import replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PROJECT_SRC = PROJECT / "src"
OUTPUT_ROOT = PROJECT / "results" / "preflight"
PREFLIGHT_DURATION_S = 6.0
PREFLIGHT_WARMUP_S = 1.0
ACCEPTED = ("ok", "contaminated")

sys.path.insert(0, str(PROJECT_SRC))
import campaign_matrix as matrix  # noqa: E402
from run_single import execute_run  # noqa: E402


def cases():
    p3 = next(r for r in matrix.phase3_runs() if r.payload_bytes == 1024 and r.topology_args["depth"] == 2)
    p4 = next(r for r in matrix.phase4_runs() if r.tracing_rate == 1.0 and r.trace_exporter == "sync")
    p5 = next(r for r in matrix.phase5_runs() if r.fault == "errors" and r.controller == "dacshed")
    return [("phase3-sidecar", p3), ("phase4-tracing", p4), ("phase5-fault-controller", p5)]


def main() -> int:
    failures = []
    for label, run in cases():
        short = replace(run, duration_s=PREFLIGHT_DURATION_S, warmup_s=PREFLIGHT_WARMUP_S)
        status = execute_run(short, OUTPUT_ROOT / f"{label}-{short.run_id}", 0, lambda: False)
        print(label, short.run_id, status, flush=True)
        if status not in ACCEPTED:
            failures.append(label)
    if failures:
        print("preflight failed for: " + ", ".join(failures))
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
