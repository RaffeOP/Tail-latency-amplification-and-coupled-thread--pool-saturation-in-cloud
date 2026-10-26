"""Tail-cascade estimators: amplification factors, saturation horizon, and extreme-value tails.

The estimators are parametric or resampling-based. No hypothesis tests are used.

Saturation horizon: with the five load levels of this design, the knee can only be located to
one level spacing. The horizon is therefore reported as the point estimate, the bracket of the
two load levels that enclose it, and the stability of that bracket across two-stage bootstrap
resamples. A percentile interval on the knee would be degenerate at this resolution.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
from scipy.interpolate import make_smoothing_spline

BOOTSTRAP_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261023
CI_LOWER_PERCENT = 2.5
CI_UPPER_PERCENT = 97.5
KNEEDLE_GRID_POINTS = 2000
GPD_THRESHOLD_QUANTILE = 0.95
GPD_MIN_EXCESSES = 30
TAIL_PERCENT = 99.9


def gamma_load(quantile_at_load_ms: float, quantile_at_base_ms: float) -> float:
    return quantile_at_load_ms / quantile_at_base_ms


def gamma_crit(quantile_request_ms: float, critical_path_median_sum_ms: float) -> float:
    return quantile_request_ms / critical_path_median_sum_ms


def kneedle_horizon(levels: np.ndarray, tail_ms: np.ndarray, smoothing: Optional[float] = None) -> Tuple[float, float]:
    """Saturation horizon Theta*: argmax of the curvature of a penalized spline in log10 space.

    The spline minimizes sum (y - g)^2 + smoothing * integral g''^2. With smoothing=None the
    penalty is chosen by generalized cross-validation.
    """
    x = np.asarray(levels, dtype=np.float64)
    y = np.log10(np.maximum(np.asarray(tail_ms, dtype=np.float64), 1e-3))
    spline = make_smoothing_spline(x, y, lam=smoothing)
    grid = np.linspace(x.min(), x.max(), KNEEDLE_GRID_POINTS)
    first = spline.derivative(1)(grid)
    second = spline.derivative(2)(grid)
    curvature = np.abs(second) / np.power(1.0 + np.square(first), 1.5)
    index = int(np.argmax(curvature))
    return float(grid[index]), float(curvature[index])


def enclosing_levels(levels: np.ndarray, theta: float) -> Tuple[float, float]:
    """The two load levels that enclose theta: the highest level at or below it and the lowest at or above."""
    ordered = np.sort(np.asarray(levels, dtype=np.float64))
    below = ordered[ordered <= theta]
    above = ordered[ordered >= theta]
    lower = below[-1] if below.size else ordered[0]
    upper = above[0] if above.size else ordered[-1]
    return float(lower), float(upper)


def replicate_tail_ms(samples_ms: np.ndarray) -> float:
    return float(np.percentile(samples_ms, TAIL_PERCENT))


def resampled_level_tail(replicates: List[np.ndarray], rng: np.random.Generator) -> float:
    """Two-stage bootstrap for one load level: resample the replicate runs, then the requests in each."""
    picks = rng.integers(0, len(replicates), size=len(replicates))
    tails = []
    for pick in picks:
        samples = replicates[pick]
        resample = rng.choice(samples, size=samples.size, replace=True)
        tails.append(float(np.percentile(resample, TAIL_PERCENT)))
    return float(np.mean(tails))


def bootstrap_horizon(levels: np.ndarray, replicate_samples_ms: List[List[np.ndarray]],
                      smoothing: Optional[float] = None) -> Dict[str, float]:
    """Saturation horizon with its bracket and stability from a two-stage bootstrap (1000 resamples).

    replicate_samples_ms[i] holds the measured latency arrays of the replicate runs at levels[i].
    The bracket is the pair of load levels enclosing the point estimate. The stability is the
    fraction of resamples whose enclosing pair is the same pair as the point estimate's.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    point_means = np.array([
        np.mean([replicate_tail_ms(samples) for samples in replicates]) for replicates in replicate_samples_ms
    ])
    point, _ = kneedle_horizon(levels, point_means, smoothing)
    point_pair = enclosing_levels(levels, point)
    same_pair = 0
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled = np.array([resampled_level_tail(replicates, rng) for replicates in replicate_samples_ms])
        theta, _ = kneedle_horizon(levels, resampled, smoothing)
        same_pair += enclosing_levels(levels, theta) == point_pair
    return {
        "theta_star_rps": point,
        "bracket_lower_rps": point_pair[0],
        "bracket_upper_rps": point_pair[1],
        "stability": same_pair / BOOTSTRAP_RESAMPLES,
    }


def fit_gpd_tail(latencies_ms: np.ndarray) -> Dict[str, float]:
    """Peaks-over-threshold GPD fit above the 95th percentile, with an extreme-quantile estimate."""
    values = np.asarray(latencies_ms, dtype=np.float64)
    threshold = float(np.quantile(values, GPD_THRESHOLD_QUANTILE))
    excesses = values[values > threshold] - threshold
    if excesses.size < GPD_MIN_EXCESSES:
        return {"threshold_ms": threshold, "n_excess": float(excesses.size),
                "xi": float("nan"), "sigma_ms": float("nan")}
    xi, _, sigma = stats.genpareto.fit(excesses, floc=0.0)
    return {
        "threshold_ms": threshold,
        "n_excess": float(excesses.size),
        "xi": float(xi),
        "sigma_ms": float(sigma),
    }


def gpd_quantile_ms(fit: Dict[str, float], total_count: int, prob: float) -> float:
    """Q_p = u + sigma/xi * [((1 - p) * N / N_u)^(-xi) - 1], with the exponential limit at xi = 0."""
    threshold = fit["threshold_ms"]
    xi = fit["xi"]
    sigma = fit["sigma_ms"]
    exceed_ratio = (1.0 - prob) * total_count / fit["n_excess"]
    if abs(xi) < 1e-9:
        return float(threshold - sigma * np.log(exceed_ratio))
    return float(threshold + sigma / xi * (exceed_ratio ** (-xi) - 1.0))


def bootstrap_gpd_xi(latencies_ms: np.ndarray, resamples: int = BOOTSTRAP_RESAMPLES) -> Tuple[float, float]:
    """Percentile bootstrap interval for the GPD shape parameter xi."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.asarray(latencies_ms, dtype=np.float64)
    shapes = np.empty(resamples)
    for index in range(resamples):
        sample = rng.choice(values, size=values.size, replace=True)
        shapes[index] = fit_gpd_tail(sample)["xi"]
    lower, upper = np.nanpercentile(shapes, [CI_LOWER_PERCENT, CI_UPPER_PERCENT])
    return float(lower), float(upper)
