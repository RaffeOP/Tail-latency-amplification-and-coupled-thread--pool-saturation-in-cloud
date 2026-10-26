"""Figure style for the paper: reference-palette slots, journal column widths, and recessive chrome.

Colors follow the validated reference palette (light mode) on a white surface. Series are also
separated by marker shape and dash pattern, because two slots sit below 3:1 contrast on white.
"""

import matplotlib as mpl

SERIES_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
OTHER_COLOR = "#898781"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
BASELINE_COLOR = "#c3c2b7"
SURFACE_COLOR = "#ffffff"
COLUMN_WIDTH_IN = 3.5
FULL_WIDTH_IN = 7.16
LINE_WIDTH_PT = 1.5
MARKER_SIZE_PT = 5.0
SERIES_MARKERS = ("o", "s", "^", "D")
SERIES_DASHES = ((None, None), (4.0, 2.0), (1.0, 1.5), (6.0, 2.0, 1.0, 2.0))
SAVE_DPI = 300


def apply_journal_style() -> None:
    mpl.rcParams.update({
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": BASELINE_COLOR,
        "axes.labelcolor": INK_SECONDARY,
        "axes.facecolor": SURFACE_COLOR,
        "figure.facecolor": SURFACE_COLOR,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "text.color": INK_PRIMARY,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": LINE_WIDTH_PT,
        "lines.markersize": MARKER_SIZE_PT,
        "legend.frameon": False,
        "savefig.dpi": SAVE_DPI,
    })


def series_style(index: int, solid: bool = False) -> dict:
    dash = SERIES_DASHES[index % len(SERIES_DASHES)]
    style = {
        "color": SERIES_COLORS[index % len(SERIES_COLORS)],
        "marker": SERIES_MARKERS[index % len(SERIES_MARKERS)],
        "markeredgecolor": SURFACE_COLOR,
        "markeredgewidth": 0.8,
    }
    if dash[0] is not None and not solid:
        style["dashes"] = dash
    return style
