"""Append-only manifest of completed runs, used to resume a campaign after a stop."""

import json
from pathlib import Path
from typing import Dict, Set


def completed_run_ids(manifest: Path) -> Set[str]:
    if not manifest.exists():
        return set()
    return {json.loads(line)["run_id"] for line in manifest.read_text().splitlines() if line.strip()}


def append_record(manifest: Path, record: Dict[str, object]) -> None:
    with manifest.open("a") as handle:
        handle.write(json.dumps(record) + "\n")
