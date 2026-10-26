"""Shared LaTeX formatting helpers for the booktabs result tables."""

import csv
import math
from pathlib import Path
from typing import Dict, List, Sequence

SHAPE_RUNTIME_SEPARATOR = " / "
WIDE_TABLES = frozenset({
    "tab02.tex", "tab03.tex", "tab04.tex", "tab05.tex", "tab06.tex", "tab07.tex",
    "tab09.tex", "tab10.tex", "tab11.tex",
})


def latex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def fmt(value: float, digits: int) -> str:
    if value is None or not math.isfinite(value):
        return "--"
    return f"{value:.{digits}f}"


def signed(value: float, digits: int) -> str:
    if not math.isfinite(value):
        return "--"
    return f"{value:+.{digits}f}"


def interval(point: float, lower: float, upper: float, digits: int) -> str:
    if not all(math.isfinite(v) for v in (point, lower, upper)):
        return "--"
    return f"{fmt(point, digits)} [{fmt(lower, digits)}, {fmt(upper, digits)}]"


def split_shape_runtime(label: str) -> List[str]:
    if SHAPE_RUNTIME_SEPARATOR in label:
        shape, runtime = label.split(SHAPE_RUNTIME_SEPARATOR, 1)
        return [latex_escape(shape), runtime]
    return [latex_escape(label), "sync"]


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_table(path: Path, caption: str, label: str, header: Sequence[str], rows: List[List[str]]) -> None:
    environment = "table*" if path.name in WIDE_TABLES else "table"
    columns = "l" * len(header)
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(rf"\textbf{{{column}}}" for column in header) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{environment}}}", ""]
    path.write_text("\n".join(lines))
