#!/usr/bin/env python3
"""Background monitor for the R4 campaign. It runs without the assistant, so it uses no tokens.

Every CHECK_INTERVAL_S it appends one status line to results/monitor.log. It starts the campaign
chain only when no campaign process is alive, the deadline has not passed, and the last start
made progress. Phases 3 to 5 start only after Phases 1 and 2 are complete. When every planned run
is complete it runs the analysis pipeline once and exits.

Automatic restarts stop and an attention marker is written if a chain exits with an error, or if
three restarts in a row make no progress. Remove results/monitor_state/attention to resume.

Usage: python3 scripts/monitor_campaign.py          (run continuously)
       python3 scripts/monitor_campaign.py --once   (print one status line and exit; starts nothing)
"""

import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import psutil

PROJECT = Path(__file__).resolve().parent.parent
RESULTS = PROJECT / "results"
RAW = RESULTS / "raw_traces"
STATE = RESULTS / "monitor_state"
MONITOR_LOG = RESULTS / "monitor.log"
PIPELINE_LOG = RESULTS / "monitor_pipeline.log"
CHAIN_LOG = RESULTS / "monitor_chain.log"
ATTENTION = STATE / "attention"
PIPELINE_DONE = STATE / "pipeline_done"
DEADLINE_TEXT = "2026-11-01T23:59:00+05:30"
DEADLINE = datetime.fromisoformat(DEADLINE_TEXT)
CHECK_INTERVAL_S = 600.0
NO_PROGRESS_LIMIT = 3
STATUS_TEXT_LIMIT = 110
PLANNED = {1: 180, 2: 135, 3: 75, 4: 60, 5: 90}
PHASE12 = "phase12"
PHASE35 = "phase35"
PHASE12_COMMAND = (
    f"python3 src/run_campaign.py --phase 1 --mode full --deadline {DEADLINE_TEXT} && "
    f"python3 src/run_campaign.py --phase 2 --mode full --deadline {DEADLINE_TEXT}"
)
PHASE35_COMMAND = (
    f"python3 scripts/preflight_phases35.py && "
    f"python3 src/run_campaign.py --phase 3 --mode full --deadline {DEADLINE_TEXT} && "
    f"python3 src/run_campaign.py --phase 4 --mode full --deadline {DEADLINE_TEXT} && "
    f"python3 src/run_campaign.py --phase 5 --mode full --deadline {DEADLINE_TEXT}"
)
PIPELINE_STEPS = (
    ("analysis phase 1", ["src/analysis_phase1.py", "raw_traces/phase1"]),
    ("analysis phase 2", ["src/analysis_phase1.py", "raw_traces/phase2"]),
    ("analysis phases 3-5", ["src/analysis_phases35.py"]),
    ("tables", ["src/make_tables.py"]),
    ("figures 1-6", ["src/plot_journal_figures.py", "raw_traces/phase1"]),
    ("figures 7-8", ["src/plot_phases35.py"]),
)


def stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    with MONITOR_LOG.open("a") as handle:
        handle.write(f"{stamp()} {message}\n")


def manifest_count(phase: int) -> int:
    path = RAW / f"phase{phase}" / "manifest.jsonl"
    if not path.exists():
        return 0
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def contaminated_count(phase: int) -> int:
    folder = RAW / f"phase{phase}"
    return sum(1 for _ in folder.glob("*.contaminated")) if folder.exists() else 0


def cmdline_of(process: psutil.Process) -> str:
    try:
        return " ".join(process.cmdline())
    except psutil.Error:
        return ""


CAMPAIGN_MARKER = "run_campaign.py --phase"
SHELL_NAMES = ("zsh", "bash", "fish", "tcsh")


def is_interactive_shell(process: psutil.Process) -> bool:
    try:
        return Path(process.cmdline()[0]).name in SHELL_NAMES
    except (psutil.Error, IndexError):
        return False


def campaign_processes() -> List[int]:
    """Campaign runners and their chain shells. Interactive shells are skipped, because a command
    that quotes the marker would otherwise look like a live runner."""
    found = []
    for process in psutil.process_iter():
        if is_interactive_shell(process):
            continue
        if CAMPAIGN_MARKER in cmdline_of(process):
            found.append(process.pid)
    return found


def orphan_node_processes() -> List[psutil.Process]:
    node_script = str(PROJECT / "src" / "node_server.py")
    orphans = []
    for process in psutil.process_iter():
        try:
            if process.ppid() == 1 and node_script in cmdline_of(process):
                orphans.append(process)
        except psutil.Error:
            continue
    return orphans


def latest_campaign_line() -> str:
    logs = [RAW / f"phase{phase}" / "campaign.log" for phase in PLANNED]
    existing = [path for path in logs if path.exists()]
    if not existing:
        return "no campaign log yet"
    newest = max(existing, key=lambda path: path.stat().st_mtime)
    lines = newest.read_text().splitlines()
    return lines[-1] if lines else "empty campaign log"


def status_line(counts: Dict[int, int], runners: List[int]) -> str:
    parts = [f"load1={os.getloadavg()[0]:.1f}", f"runners={len(runners)}"]
    parts += [f"p{phase}={counts[phase]}/{PLANNED[phase]}" for phase in PLANNED]
    parts.append(f"contaminated={sum(contaminated_count(phase) for phase in PLANNED)}")
    parts.append("state=" + ("attention" if ATTENTION.exists() else "ok"))
    last = latest_campaign_line()[:STATUS_TEXT_LIMIT]
    return " ".join(parts) + f' last="{last}"'


def raise_attention(reason: str) -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    ATTENTION.write_text(f"{stamp()} {reason}\n")
    log(f"ATTENTION: {reason}. Automatic restarts are stopped until {ATTENTION} is removed.")


def launch(command: str) -> subprocess.Popen:
    handle = CHAIN_LOG.open("a")
    return subprocess.Popen(["sh", "-c", command], cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT,
                            start_new_session=True)


def check_exit_codes(chains: Dict[str, Optional[subprocess.Popen]]) -> None:
    for name, process in chains.items():
        if process is not None and process.poll() not in (None, 0):
            raise_attention(f"{name} chain exited with status {process.returncode}")
            chains[name] = None


def guard_restart(name: str, total_done: int, last_total: Dict[str, Optional[int]],
                  no_progress: Dict[str, int]) -> bool:
    previous = last_total[name]
    if previous is not None and total_done <= previous:
        no_progress[name] += 1
    else:
        no_progress[name] = 0
    last_total[name] = total_done
    if no_progress[name] >= NO_PROGRESS_LIMIT:
        raise_attention(f"{name} restarted {no_progress[name]} times with no new completed runs")
        return False
    return True


def kill_orphans() -> None:
    for process in orphan_node_processes():
        log(f"stopping orphaned node process {process.pid} left by a finished runner")
        try:
            process.kill()
        except psutil.Error:
            continue


def run_pipeline() -> None:
    if PIPELINE_DONE.exists():
        return
    for label, arguments in PIPELINE_STEPS:
        log(f"pipeline: {label}")
        result = subprocess.run([sys.executable] + arguments, cwd=PROJECT, capture_output=True, text=True)
        with PIPELINE_LOG.open("a") as handle:
            handle.write(f"== {stamp()} {label} exit {result.returncode}\n")
            handle.write(result.stdout[-2000:] + "\n" + result.stderr[-2000:] + "\n")
        if result.returncode != 0:
            raise_attention(f"pipeline step failed: {label}")
            return
    PIPELINE_DONE.write_text(stamp() + "\n")
    log("pipeline complete; results are in results/metrics, results/figures, and tables/")


def monitor_loop() -> None:
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "monitor.pid").write_text(f"{os.getpid()}\n")
    chains: Dict[str, Optional[subprocess.Popen]] = {PHASE12: None, PHASE35: None}
    last_total: Dict[str, Optional[int]] = {PHASE12: None, PHASE35: None}
    no_progress: Dict[str, int] = {PHASE12: 0, PHASE35: 0}
    log(f"monitor started (pid {os.getpid()}), deadline {DEADLINE_TEXT}")
    while datetime.now(DEADLINE.tzinfo) < DEADLINE:
        counts = {phase: manifest_count(phase) for phase in PLANNED}
        runners = campaign_processes()
        log(status_line(counts, runners))
        if all(counts[phase] >= PLANNED[phase] for phase in PLANNED):
            run_pipeline()
            log("monitor finished")
            return
        check_exit_codes(chains)
        if not ATTENTION.exists() and not runners:
            kill_orphans()
            name = PHASE12 if counts[1] < PLANNED[1] or counts[2] < PLANNED[2] else PHASE35
            command = PHASE12_COMMAND if name == PHASE12 else PHASE35_COMMAND
            if guard_restart(name, sum(counts.values()), last_total, no_progress):
                chains[name] = launch(command)
                log(f"started {name} chain (pid {chains[name].pid})")
        time.sleep(CHECK_INTERVAL_S)
    log("deadline reached; monitor exiting")


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor the R4 campaign.")
    parser.add_argument("--once", action="store_true", help="print one status line and exit")
    arguments = parser.parse_args()
    if arguments.once:
        counts = {phase: manifest_count(phase) for phase in PLANNED}
        print(status_line(counts, campaign_processes()))
        return 0
    monitor_loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
