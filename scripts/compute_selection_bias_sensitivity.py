#!/usr/bin/env python3
"""
Selection-Bias Sensitivity Analysis on the 472 Stored Discarded Runs.
Analyzes background load distribution across kept (540) and discarded (472) runs,
evaluating borderline runs with mean background load in (1.00, 1.20] cores.
Zero live runs required.
"""
from pathlib import Path
import sys
import numpy as np
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from host_gate import background_cores

def main():
    raw_root = PROJECT_ROOT / "results" / "raw_traces"
    campaign_phases = {"phase1", "phase2", "phase3", "phase4", "phase5"}
    contam_folders = [f for f in raw_root.glob("*/*.contaminated") if f.parent.name in campaign_phases]
    
    bgs_discarded = []
    for f in contam_folders:
        t_file = f / "telemetry.parquet"
        if t_file.exists():
            rows = pq.read_table(t_file).to_pylist()
            bgs_discarded.append(background_cores(rows))
            
    bgs_discarded = np.array(bgs_discarded)
    borderline_120 = ((bgs_discarded > 1.00) & (bgs_discarded <= 1.20)).sum()
    borderline_130 = ((bgs_discarded > 1.00) & (bgs_discarded <= 1.30)).sum()
    
    print("=" * 70)
    print("Selection-Bias Sensitivity Analysis (472 Discarded Runs):")
    print("-" * 70)
    print(f"Total discarded directories with telemetry: {len(bgs_discarded)}")
    print(f"Discarded background range: {bgs_discarded.min():.2f} to {bgs_discarded.max():.2f} cores")
    print(f"Discarded median: {np.median(bgs_discarded):.2f} cores, mean: {np.mean(bgs_discarded):.2f} cores")
    print(f"Borderline runs in (1.00, 1.20] cores: {borderline_120}")
    print(f"Borderline runs in (1.00, 1.30] cores: {borderline_130}")
    print("This script counts borderline discards only. It does not re-run the analysis;")
    print("see scripts/relaxed_gate_sensitivity.py for the re-analysis of saturation brackets.")
    print("=" * 70)

if __name__ == "__main__":
    main()
