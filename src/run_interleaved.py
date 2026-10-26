"""Runs Phase 4b block by block: all nine configurations of a block finish before the next block starts.

Blocks are keyed by load and block number, parsed from the run identifier. A contaminated run is
retried inside its own block, so no later block starts while an earlier one is incomplete.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import argparse
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import List, Tuple

from campaign_extra import PHASE_4B, extra_phase_runs
from host_gate import wait_for_quiet_host
from mesh_config import RunSpec
from run_campaign import RESULTS_ROOT, RETRY_PAUSE_S, RUN_OVERHEAD_S, make_logger, run_one
from run_single import STATUS_ABORTED, STATUS_OK
from storage_manifest import completed_run_ids

RUN_ID_LOAD_FIELD = 1
RUN_ID_BLOCK_FIELD = 2
BLOCK_PREFIX = "b"


def block_key(run: RunSpec) -> Tuple[str, int]:
    fields = run.run_id.split("-")
    return fields[RUN_ID_LOAD_FIELD], int(fields[RUN_ID_BLOCK_FIELD][len(BLOCK_PREFIX):])


def group_blocks(runs: List[RunSpec]) -> "OrderedDict[Tuple[str, int], List[RunSpec]]":
    blocks: "OrderedDict[Tuple[str, int], List[RunSpec]]" = OrderedDict()
    for run in runs:
        blocks.setdefault(block_key(run), []).append(run)
    return blocks


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Phase 4b in randomised blocks.")
    parser.add_argument("--deadline", required=True, help="ISO 8601 timestamp with offset")
    return parser.parse_args(argv)


def run_block(block: List[RunSpec], root, manifest, deadline, log) -> bool:
    """Returns False when the deadline stops the phase, True when the block is complete."""

    def past_deadline() -> bool:
        return datetime.now(deadline.tzinfo) >= deadline

    while True:
        done = completed_run_ids(manifest)
        pending = [run for run in block if run.run_id not in done]
        if not pending:
            return True
        progressed = False
        for slot, run in enumerate(pending):
            finish_by = datetime.now(deadline.tzinfo) + timedelta(seconds=run.duration_s + RUN_OVERHEAD_S)
            if past_deadline() or finish_by > deadline:
                log("deadline would be crossed; stopping")
                return False
            if not wait_for_quiet_host(past_deadline, log):
                return False
            status = run_one(run, root, slot, past_deadline, manifest, log)
            if status == STATUS_ABORTED:
                return False
            progressed = progressed or status == STATUS_OK
        if not progressed:
            time.sleep(RETRY_PAUSE_S)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    deadline = datetime.fromisoformat(args.deadline)
    root = RESULTS_ROOT / "raw_traces" / f"phase{PHASE_4B}"
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    log = make_logger(root / "campaign.log")
    blocks = group_blocks(extra_phase_runs(PHASE_4B))
    log(f"phase={PHASE_4B} blocks={len(blocks)} planned={sum(len(b) for b in blocks.values())} deadline={deadline.isoformat()}")
    for key, block in blocks.items():
        if not run_block(block, root, manifest, deadline, log):
            return 0
        log(f"block {key[0]} b{key[1]} complete")
    log("all planned runs are complete")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
