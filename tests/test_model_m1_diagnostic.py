"""Tests for the exploratory M1 order-statistic moments and path-length helpers."""

import math

import networkx as nx
import pytest

from model_m1_diagnostic import child_stage_moments, iid_max_moments, longest_root_path_vertices

MAX_OF_TWO_MEAN = 1.0 / math.sqrt(math.pi)
MAX_OF_TWO_VARIANCE = 1.0 - 1.0 / math.pi
MOMENT_TOLERANCE = 1e-6


def test_iid_max_of_two_mean_matches_closed_form():
    mean, _ = iid_max_moments(2)
    assert mean == pytest.approx(MAX_OF_TWO_MEAN, abs=MOMENT_TOLERANCE)


def test_iid_max_of_two_variance_matches_closed_form():
    _, variance = iid_max_moments(2)
    assert variance == pytest.approx(MAX_OF_TWO_VARIANCE, abs=MOMENT_TOLERANCE)


def test_iid_max_mean_increases_with_fan_out():
    means = [iid_max_moments(fan_out)[0] for fan_out in range(2, 9)]
    assert means == sorted(means)


def test_single_child_passes_its_moments_through_unchanged():
    response = {7: 12.5}
    variance = {7: 3.25}
    mean, var = child_stage_moments([7], response, variance, net_hop_ms=0.5)
    assert mean == pytest.approx(13.0)
    assert var == pytest.approx(3.25)


def test_leaf_has_no_downstream_stage():
    assert child_stage_moments([], {}, {}, net_hop_ms=0.5) == (0.0, 0.0)


def test_longest_root_path_counts_vertices_on_the_longest_branch():
    graph = nx.DiGraph([(0, 1), (1, 2), (1, 3), (3, 4)])
    assert longest_root_path_vertices(graph) == 4
