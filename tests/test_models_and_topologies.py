"""Tests for the analytical DAG model, tail estimators, topology builders, and the run matrix."""

import numpy as np
import pytest
from scipy import stats

import campaign_matrix as matrix
from analytical_dag_model import ModelParams, predict, visit_counts
from dag_topology_builder import (
    build_bounded_scatter_gather,
    build_deathstar_hotel_reservation,
    build_diamond_incast_dag,
    build_serial_chain,
    build_topology,
)
from tail_cascade_analyzer import fit_gpd_tail, gamma_crit, gamma_load, gpd_quantile_ms, kneedle_horizon

PARAMS = ModelParams(pool_size=8, service_mean_ms=10.0, service_sigma=0.5, net_hop_ms=0.5,
                     cpu_per_visit_ms=1.5, host_cores=8)


def test_should_count_one_visit_per_vertex_in_a_chain():
    assert set(visit_counts(build_serial_chain(4)).values()) == {1}


def test_should_count_one_visit_per_branch_for_each_diamond_shared_leaf():
    counts = visit_counts(build_diamond_incast_dag(depth=3, width=4))
    assert max(counts.values()) == 4


def test_should_report_pool_utilization_as_rate_times_hold_over_pool_size():
    graph = build_serial_chain(1)
    prediction = predict(graph, root_rate_rps=40.0, params=PARAMS)
    expected = 40.0 * 10.0 / 1000.0 / 8
    assert prediction.vertices[0].rho_pool == pytest.approx(expected)


def test_should_grow_root_response_time_with_chain_depth():
    shallow = predict(build_serial_chain(2), 10.0, PARAMS).root_mean_ms
    deep = predict(build_serial_chain(6), 10.0, PARAMS).root_mean_ms
    assert deep > shallow


def test_should_grow_waiting_time_with_arrival_rate():
    graph = build_serial_chain(3)
    low = predict(graph, 20.0, PARAMS).root_mean_ms
    high = predict(graph, 70.0, PARAMS).root_mean_ms
    assert high > low


def test_should_order_root_quantiles_monotonically():
    quantiles = predict(build_serial_chain(3), 20.0, PARAMS).root_quantiles_ms
    values = [quantiles[p] for p in sorted(quantiles)]
    assert values == sorted(values)


def test_should_compute_gamma_ratios_from_their_definitions():
    assert gamma_load(300.0, 100.0) == pytest.approx(3.0)
    assert gamma_crit(300.0, 60.0) == pytest.approx(5.0)


def test_should_place_kneedle_horizon_near_the_knee_of_a_hockey_stick_curve():
    levels = np.linspace(10.0, 200.0, 20)
    knee = 120.0
    tail = np.where(levels < knee, 20.0 + 0.01 * levels, 20.0 + 0.01 * knee + 40.0 * (levels - knee) ** 2 / knee)
    theta, _ = kneedle_horizon(levels, tail)
    assert 90.0 <= theta <= 170.0


def test_should_recover_the_gpd_shape_from_a_known_tail():
    rng = np.random.default_rng(5)
    sample = stats.genpareto.rvs(c=0.2, scale=2.0, size=200_000, random_state=rng) + 10.0
    fit = fit_gpd_tail(sample)
    assert fit["xi"] == pytest.approx(0.2, abs=0.05)


def test_should_increase_extreme_gpd_quantiles_with_the_probability_level():
    rng = np.random.default_rng(9)
    sample = stats.genpareto.rvs(c=0.1, scale=3.0, size=20000, random_state=rng) + 5.0
    fit = fit_gpd_tail(sample)
    assert gpd_quantile_ms(fit, sample.size, 0.9999) > gpd_quantile_ms(fit, sample.size, 0.999)


def test_should_build_the_phase_one_matrix_with_180_runs_and_unique_ids():
    runs = matrix.phase1_runs()
    assert len(runs) == matrix.PHASE_RUN_COUNTS[1]
    assert len({run.run_id for run in runs}) == len(runs)


def test_should_keep_every_phase_one_topology_within_its_vertex_budget():
    for shape in matrix.phase1_shapes():
        graph = build_topology(str(shape["topology"]), dict(shape["args"]))
        assert graph.number_of_nodes() <= 43


def test_should_order_each_shapes_load_levels_ascending():
    for shape in matrix.phase1_shapes():
        grid = matrix.analytic_grid(build_topology(str(shape["topology"]), dict(shape["args"])))
        assert grid["levels_rps"] == sorted(grid["levels_rps"])


def test_should_derive_seeds_deterministically_from_run_ids():
    assert matrix.stable_seed("p1-x") == matrix.stable_seed("p1-x")
    assert matrix.stable_seed("p1-x") != matrix.stable_seed("p1-y")


def test_should_bound_scatter_gather_trees_by_the_vertex_budget():
    graph = build_bounded_scatter_gather(depth=4, fanout=8, max_vertices=32)
    assert graph.number_of_nodes() <= 32


def test_should_make_the_hotel_leaves_shared_between_two_paths():
    counts = visit_counts(build_deathstar_hotel_reservation())
    assert counts[5] == 2 and counts[6] == 2

