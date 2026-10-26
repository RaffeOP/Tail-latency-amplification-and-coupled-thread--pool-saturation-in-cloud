"""Tests for the stable-level rule used when comparing the model with measurements."""

from model_agreement import stable_level_samples

SHAPE = "chain_D2"


def level_row(rate: float, utilization: float, timeouts: float, p999: float) -> dict:
    return {
        "shape": SHAPE,
        "rate_rps": rate,
        "rho_pool_root_measured": utilization,
        "timeout_fraction": timeouts,
        "p999_ms": p999,
    }


def test_should_keep_a_level_whose_mean_utilization_is_below_the_cutoff():
    rows = [level_row(10.0, 0.5, 0.0, 40.0), level_row(10.0, 0.6, 0.0, 44.0)]
    assert (SHAPE, 10.0) in stable_level_samples(rows)


def test_should_drop_a_level_whose_mean_utilization_reaches_the_cutoff():
    rows = [level_row(20.0, 0.95, 0.0, 90.0), level_row(20.0, 0.95, 0.0, 95.0)]
    assert (SHAPE, 20.0) not in stable_level_samples(rows)


def test_should_drop_a_level_whose_mean_timeout_fraction_exceeds_one_percent():
    rows = [level_row(30.0, 0.4, 0.02, 50.0), level_row(30.0, 0.4, 0.03, 60.0)]
    assert (SHAPE, 30.0) not in stable_level_samples(rows)


def test_should_return_the_replicate_tail_values_of_a_stable_level():
    rows = [level_row(10.0, 0.5, 0.0, 40.0), level_row(10.0, 0.6, 0.0, 44.0)]
    assert stable_level_samples(rows)[(SHAPE, 10.0)] == [40.0, 44.0]
