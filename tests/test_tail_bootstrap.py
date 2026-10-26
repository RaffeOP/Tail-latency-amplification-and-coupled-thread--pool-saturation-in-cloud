"""Tests for the saturation-horizon bootstrap: bracket, stability, and determinism."""

import numpy as np

import tail_cascade_analyzer as tca

SMALL_RESAMPLE_COUNT = 60
LEVELS = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
LEVEL_SCALES = (1.0, 1.1, 1.4, 4.0, 9.0)
REQUESTS_PER_REPLICATE = 4000


def replicate_samples(seed: int):
    rng = np.random.default_rng(seed)
    return [
        [scale * rng.lognormal(0.0, 0.5, REQUESTS_PER_REPLICATE) for _ in range(2)]
        for scale in LEVEL_SCALES
    ]


def test_should_bracket_the_horizon_between_the_knee_level_and_the_next_level(monkeypatch):
    monkeypatch.setattr(tca, "BOOTSTRAP_RESAMPLES", SMALL_RESAMPLE_COUNT)
    result = tca.bootstrap_horizon(LEVELS, replicate_samples(seed=4))
    assert result["bracket_lower_rps"] <= result["theta_star_rps"] <= result["bracket_upper_rps"]
    assert result["bracket_upper_rps"] - result["bracket_lower_rps"] in (0.0, 10.0)


def test_should_report_a_stability_fraction_between_zero_and_one(monkeypatch):
    monkeypatch.setattr(tca, "BOOTSTRAP_RESAMPLES", SMALL_RESAMPLE_COUNT)
    result = tca.bootstrap_horizon(LEVELS, replicate_samples(seed=5))
    assert 0.0 <= result["stability"] <= 1.0


def test_should_reproduce_the_same_result_for_the_same_data(monkeypatch):
    monkeypatch.setattr(tca, "BOOTSTRAP_RESAMPLES", SMALL_RESAMPLE_COUNT)
    first = tca.bootstrap_horizon(LEVELS, replicate_samples(seed=6))
    second = tca.bootstrap_horizon(LEVELS, replicate_samples(seed=6))
    assert first == second
