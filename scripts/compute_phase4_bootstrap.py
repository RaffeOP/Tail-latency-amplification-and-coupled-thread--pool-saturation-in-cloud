#!/usr/bin/env python3
"""
Phase 4 Pooled Bootstrap & Directional Lower Bound on Tracing Overhead.
Evaluates high-load traced runs (S >= 0.25, 18 runs across 6 configurations)
against the rate-zero baseline (6 runs) from results/raw_traces/phase4.
Zero live runs required.
"""
from pathlib import Path
import sys
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from result_loader import load_runs

def main():
    root = PROJECT_ROOT / "results" / "raw_traces" / "phase4"
    runs = load_runs(root)
    
    base_high = [r for r in runs if float(r.run["tracing_rate"]) == 0.0 and round(r.rate_rps, 1) == 115.7]
    traced_high = [r for r in runs if float(r.run["tracing_rate"]) >= 0.25 and round(r.rate_rps, 1) == 115.7]
    
    base_p99 = np.median([r.percentile_ms("p99") for r in base_high])
    base_p999 = np.median([r.percentile_ms("p999") for r in base_high])
    base_bg = np.median([r.background_cores() for r in base_high])
    
    p99_traced = np.array([r.percentile_ms("p99") for r in traced_high])
    p999_traced = np.array([r.percentile_ms("p999") for r in traced_high])
    bg_traced = np.array([r.background_cores() for r in traced_high])
    
    # Configuration-level medians (6 configurations)
    from collections import defaultdict
    configs = defaultdict(list)
    for r in traced_high:
        key = (float(r.run["tracing_rate"]), str(r.run["trace_exporter"]))
        configs[key].append(r)
        
    delta_p99_cells = []
    delta_p999_cells = []
    bg_cells = []
    
    print("=" * 70)
    print("Phase 4 High-Load Cell Medians (115.7 req/s):")
    print(f"Baseline (S=0.0): P99 = {base_p99:.2f} ms, P99.9 = {base_p999:.2f} ms, background = {base_bg:.2f} cores")
    print("-" * 70)
    for (rate, exporter), cell_runs in sorted(configs.items()):
        cell_p99 = np.median([r.percentile_ms("p99") for r in cell_runs])
        cell_p999 = np.median([r.percentile_ms("p999") for r in cell_runs])
        cell_bg = np.median([r.background_cores() for r in cell_runs])
        d99 = cell_p99 - base_p99
        d999 = cell_p999 - base_p999
        delta_p99_cells.append(d99)
        delta_p999_cells.append(d999)
        bg_cells.append(cell_bg)
        print(f"S={rate:<4} {exporter:<5}: P99 = {cell_p99:.2f} ms (Δ = +{d99:.2f} ms), P99.9 = {cell_p999:.2f} ms (Δ = +{d999:.2f} ms), bg = {cell_bg:.2f} cores")
    
    # Exact sign test across 6 cells
    n_pos_99 = sum(1 for d in delta_p99_cells if d > 0)
    n_pos_999 = sum(1 for d in delta_p999_cells if d > 0)
    p_val_sign = 2.0 * (0.5 ** len(delta_p99_cells))
    
    # Bootstrap over the 18 high-load runs
    rng = np.random.default_rng(42)
    b_diffs_99 = []
    b_diffs_999 = []
    for _ in range(10000):
        s_base_99 = rng.choice([r.percentile_ms("p99") for r in base_high], size=len(base_high), replace=True)
        s_traced_99 = rng.choice(p99_traced, size=len(p99_traced), replace=True)
        b_diffs_99.append(np.median(s_traced_99) - np.median(s_base_99))
        
        s_base_999 = rng.choice([r.percentile_ms("p999") for r in base_high], size=len(base_high), replace=True)
        s_traced_999 = rng.choice(p999_traced, size=len(p999_traced), replace=True)
        b_diffs_999.append(np.median(s_traced_999) - np.median(s_base_999))
        
    ci_99 = np.percentile(b_diffs_99, [2.5, 97.5])
    ci_999 = np.percentile(b_diffs_999, [2.5, 97.5])
    
    print("=" * 70)
    print("Pooled 18-Run Analysis:")
    print(f"P99 delta range: [{min(delta_p99_cells):.2f}, {max(delta_p99_cells):.2f}] ms")
    print(f"P99.9 delta range: [{min(delta_p999_cells):.2f}, {max(delta_p999_cells):.2f}] ms")
    print(f"Exact two-sided sign test: p = {p_val_sign:.5f} ({n_pos_99}/6 positive at P99, {n_pos_999}/6 positive at P99.9)")
    print(f"Pooled bootstrap median overhead at P99: +{np.median(b_diffs_99):.2f} ms (95% CI [{ci_99[0]:.2f}, {ci_99[1]:.2f}] ms, +{np.median(b_diffs_99)/base_p99*100:.1f}%)")
    print(f"Pooled median overhead at P99.9: +{np.median(p999_traced) - base_p999:.2f} ms (+{(np.median(p999_traced) - base_p999)/base_p999*100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
