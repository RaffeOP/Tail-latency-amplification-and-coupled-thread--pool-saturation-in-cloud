#!/usr/bin/env python3
"""
Offline Trace-Replay Counterfactual for Budget-Gated Single Retry (dacshed_retry).
Evaluates existing Phase 5 run directories on disk (results/raw_traces/phase5).
Zero live runs required.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PHASE5_DIR = PROJECT_ROOT / "results" / "raw_traces" / "phase5"
SLA_US = 250_000.0        # 250 ms SLA
FLOOR_US = 20_000.0       # 20 ms floor for child 2 -> child 3
MEASURED_WINDOW_S = 80.0  # 90s duration - 10s warmup

def evaluate_run(run_dir: Path, error_prob: float = 0.30) -> dict:
    req_file = run_dir / "requests.parquet"
    if not req_file.exists():
        raise FileNotFoundError(f"Missing {req_file}")
    df = pq.read_table(req_file).to_pandas()
    df_meas = df[df["measured"]].copy()
    
    total_reqs = len(df_meas)
    succ_mask = (df_meas["outcome"] == 0) & (df_meas["latency_us"] <= SLA_US)
    err_mask = (df_meas["outcome"] != 0) & (df_meas["latency_us"] <= SLA_US - 2 * FLOOR_US)
    
    base_goodput = succ_mask.sum() / MEASURED_WINDOW_S
    base_succ_rate = succ_mask.sum() / total_reqs
    
    succ_latencies = df_meas.loc[succ_mask, "latency_us"].to_numpy()
    err_latencies = df_meas.loc[err_mask, "latency_us"].to_numpy()
    n_err = len(err_latencies)
    
    # Trace-replay: each eligible error issues at most 1 retry
    # Retry succeeds with probability (1 - error_prob)
    rng = np.random.default_rng(42)
    retry_latencies = rng.choice(succ_latencies, size=n_err, replace=True)
    retry_success = rng.random(size=n_err) < (1.0 - error_prob)
    recovered = retry_success & ((err_latencies + retry_latencies) <= SLA_US)
    
    new_succ_count = succ_mask.sum() + recovered.sum()
    retry_goodput = new_succ_count / MEASURED_WINDOW_S
    retry_succ_rate = new_succ_count / total_reqs
    
    # Overhead in extra hop calls across 4-hop chain
    extra_invocations_pct = (n_err / (total_reqs * 4)) * 100.0
    
    return {
        "run_id": run_dir.name,
        "total_reqs": total_reqs,
        "base_goodput": base_goodput,
        "retry_goodput": retry_goodput,
        "base_succ": base_succ_rate,
        "retry_succ": retry_succ_rate,
        "n_err": n_err,
        "recovered": int(recovered.sum()),
        "extra_invocations_pct": extra_invocations_pct,
    }

def main():
    print("=" * 70)
    print("Offline Trace-Replay Counterfactual Bound (Phase 5 Level 5, 154.2 req/s):")
    print("=" * 70)
    
    runs = ["p5-errors-dacshed-L5-r1", "p5-errors-dacshed-L5-r2"]
    results = []
    for r in runs:
        res = evaluate_run(PHASE5_DIR / r)
        results.append(res)
        print(f"Run {r}:")
        print(f"  Base goodput: {res['base_goodput']:.2f} req/s (succ = {res['base_succ']:.4f})")
        print(f"  Trace-replay goodput: {res['retry_goodput']:.2f} req/s (succ = {res['retry_succ']:.4f})")
        print(f"  Recovered: {res['recovered']}/{res['n_err']} ({res['recovered']/res['n_err']*100:.1f}%)")
        print(f"  Overhead: +{res['extra_invocations_pct']:.2f}% leaf/hop invocations")
    
    mean_base_gp = np.mean([r["base_goodput"] for r in results])
    mean_retry_gp = np.mean([r["retry_goodput"] for r in results])
    mean_base_succ = np.mean([r["base_succ"] for r in results])
    mean_retry_succ = np.mean([r["retry_succ"] for r in results])
    mean_overhead = np.mean([r["extra_invocations_pct"] for r in results])
    
    print("-" * 70)
    print(f"Mean Base dacshed: goodput = {mean_base_gp:.1f} req/s (succ = {mean_base_succ:.2f})")
    print(f"Mean Trace-replay dacshed_retry: goodput = {mean_retry_gp:.1f} req/s (succ = {mean_retry_succ:.2f})")
    print(f"Overhead: +{mean_overhead:.1f}% invocations")
    print("Only injected-error traces are evaluated here. Slowdown and hang runs contain no")
    print("injected errors, so the retry gate permits no retry under those faults by construction.")

if __name__ == "__main__":
    main()
