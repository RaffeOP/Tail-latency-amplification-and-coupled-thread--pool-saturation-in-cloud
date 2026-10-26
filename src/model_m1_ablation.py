"""Three-model ablation on the Phase 1 stable pairs: primary M0, path-SCV diagnostic, and exploratory M1.

Uses the same stable (shape, level) pairs as the model-agreement table, so the M0 medians must
reproduce the published values. Writes results/metrics/model_m1_ablation.csv and prints summaries.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import csv
from typing import Dict, List

import numpy as np

from analysis_phase1 import MODEL_PARAMS
from analytical_dag_model import predict
from model_agreement import METRICS_DIR, PREDICTION_PROB, shape_graph, stable_level_samples
from model_m1_diagnostic import m1_root_quantile_ms, path_scv_root_quantile_ms

OUTPUT_CSV = METRICS_DIR / "model_m1_ablation.csv"
MODEL_KEYS = ["m0", "path_scv", "m1"]
DIGITS = 3
CSV_DECIMALS = 4
QUARTILE_PERCENTILES = [25, 50, 75]


def load_phase1_rows() -> List[Dict[str, str]]:
    with (METRICS_DIR / "phase1_runs.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def pair_predictions(shape_graphs: Dict[str, object], shape: str, rate: float) -> Dict[str, float]:
    graph = shape_graphs[shape]
    return {
        "m0": predict(graph, rate, MODEL_PARAMS).root_quantiles_ms[PREDICTION_PROB],
        "path_scv": path_scv_root_quantile_ms(graph, rate, MODEL_PARAMS),
        "m1": m1_root_quantile_ms(graph, rate, MODEL_PARAMS),
    }


def build_pairs(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    first_run: Dict[str, str] = {}
    for row in rows:
        first_run.setdefault(row["shape"], row["run_id"])
    shape_graphs = {shape: shape_graph(run_id) for shape, run_id in first_run.items()}
    pairs = []
    for (shape, rate), samples in sorted(stable_level_samples(rows).items()):
        measured = float(np.mean(samples))
        predicted = pair_predictions(shape_graphs, shape, rate)
        record: Dict[str, object] = {"shape": shape, "family": shape.split("_")[0], "rate_rps": rate,
                                     "measured_p999_ms": measured}
        for key in MODEL_KEYS:
            record[f"predicted_{key}_ms"] = predicted[key]
            record[f"ratio_{key}"] = measured / predicted[key]
        pairs.append(record)
    return pairs


def summarize(values: List[float]) -> str:
    q1, median, q3 = np.percentile(values, QUARTILE_PERCENTILES)
    return f"median {median:.{DIGITS}f} (IQR {q1:.{DIGITS}f} to {q3:.{DIGITS}f}, n={len(values)})"


def write_pairs(pairs: List[Dict[str, object]]) -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    header = list(pairs[0].keys())
    with OUTPUT_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for record in pairs:
            writer.writerow({key: round(value, CSV_DECIMALS) if isinstance(value, float) else value
                             for key, value in record.items()})


def print_summary(pairs: List[Dict[str, object]]) -> None:
    for key in MODEL_KEYS:
        values = [float(p[f"ratio_{key}"]) for p in pairs]
        print(f"ALL  {key:9s} {summarize(values)}")
    for family in sorted({str(p["family"]) for p in pairs}):
        subset = [p for p in pairs if p["family"] == family]
        for key in MODEL_KEYS:
            print(f"{family:8s} {key:9s} {summarize([float(p[f'ratio_{key}']) for p in subset])}")


def main() -> int:
    pairs = build_pairs(load_phase1_rows())
    write_pairs(pairs)
    print_summary(pairs)
    print(f"wrote {OUTPUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
