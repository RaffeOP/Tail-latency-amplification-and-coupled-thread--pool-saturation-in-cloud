"""Runs one campaign phase with checkpointing, a quiet-host gate, and a hard stop time.

Completed runs are recorded in a manifest and skipped on restart. A run starts only on a quiet
host and only if it can finish before the deadline. Runs disturbed by other load are kept aside
as contaminated and retried later.
"""

import argparse
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List

from campaign_extra import EXTRA_PHASES, extra_phase_runs
from campaign_matrix import phase_runs, pilot_subset
from host_gate import wait_for_quiet_host
from mesh_config import RunSpec
from run_single import STATUS_ABORTED, STATUS_OK, execute_run
from storage_manifest import append_record, completed_run_ids

RESULTS_ROOT = Path(__file__).resolve().parent.parent / "results"
RUN_OVERHEAD_S = 20.0
RETRY_PAUSE_S = 60.0
LOG_NAME = "campaign.log"


def all_phase_runs(phase: int) -> List[RunSpec]:
    if phase in EXTRA_PHASES:
        return extra_phase_runs(phase)
    return phase_runs(phase)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a campaign phase with checkpointing.")
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--mode", choices=("full", "pilot"), default="full")
    parser.add_argument("--deadline", required=True, help="ISO 8601 timestamp with offset")
    return parser.parse_args(argv)


def make_logger(path: Path) -> Callable[[str], None]:
    def log(message: str) -> None:
        line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {message}"
        print(line, flush=True)
        with path.open("a") as handle:
            handle.write(line + "\n")
    return log


def run_one(run: RunSpec, root: Path, slot: int, past_deadline: Callable[[], bool],
            manifest: Path, log: Callable[[str], None]) -> str:
    started = time.monotonic()
    status = execute_run(run, root / run.run_id, slot, past_deadline)
    log(f"{run.run_id} {status} {time.monotonic() - started:.1f}s")
    if status == STATUS_OK:
        append_record(manifest, {"run_id": run.run_id, "status": status,
                                 "finished": datetime.now().astimezone().isoformat(timespec="seconds")})
    return status


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    deadline = datetime.fromisoformat(args.deadline)
    runs = pilot_subset(all_phase_runs(args.phase)) if args.mode == "pilot" else all_phase_runs(args.phase)
    root = RESULTS_ROOT / ("pilot" if args.mode == "pilot" else "raw_traces") / f"phase{args.phase}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    log = make_logger(root / LOG_NAME)
    log(f"phase={args.phase} mode={args.mode} planned={len(runs)} deadline={deadline.isoformat()}")

    def past_deadline() -> bool:
        return datetime.now(deadline.tzinfo) >= deadline

    while True:
        done = completed_run_ids(manifest)
        pending = [run for run in runs if run.run_id not in done]
        if not pending:
            log("all planned runs are complete")
            return 0
        progressed = False
        for slot, run in enumerate(pending):
            if past_deadline():
                log("deadline reached; stopping")
                return 0
            finish_by = datetime.now(deadline.tzinfo) + timedelta(seconds=run.duration_s + RUN_OVERHEAD_S)
            if finish_by > deadline:
                log("the next run cannot finish before the deadline; stopping")
                return 0
            if not wait_for_quiet_host(past_deadline, log):
                log("deadline reached while waiting for a quiet host; stopping")
                return 0
            status = run_one(run, root, slot, past_deadline, manifest, log)
            if status == STATUS_ABORTED:
                log("deadline reached during a run; the partial run was discarded")
                return 0
            progressed = progressed or status == STATUS_OK
        if not progressed:
            time.sleep(RETRY_PAUSE_S)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
