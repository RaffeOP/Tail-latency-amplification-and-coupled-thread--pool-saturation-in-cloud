"""Exploratory M1 diagnostic: order-statistic shift and holding-time variance for the coupled model.

M1 extends the primary model (analytical_dag_model.predict) with two terms it omits: the expected
shift of the slowest of F parallel children above their mean, and the variance of holding times,
which feeds a per-vertex waiting-time variance. Equal-moment fan-out (identical children) uses the
exact moments of the maximum of F standard normals, computed by numerical integration. Unequal
children use the same moments scaled by the root-mean-square child standard deviation, which is an
approximation. The waiting-variance term is heuristic.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import math
from functools import lru_cache
from typing import Dict, List, Tuple

import networkx as nx
from scipy.integrate import quad
from scipy.stats import norm

from analytical_dag_model import (
    MAX_UTILIZATION,
    MS_PER_S,
    ModelParams,
    lognormal_quantile_ms,
    predict,
    sakasegawa_wait_ms,
    service_scv,
    visit_counts,
)

INTEGRATION_LIMIT_SIGMA = 12.0
WAIT_UTILIZATION_FLOOR = 0.05
ROOT_PROBABILITY = 0.999


@lru_cache(maxsize=None)
def iid_max_moments(fan_out: int) -> Tuple[float, float]:
    """Mean and variance of the maximum of fan_out i.i.d. standard normals."""
    density_cdf = lambda x: norm.pdf(x) * norm.cdf(x) ** (fan_out - 1) * fan_out
    mean = quad(lambda x: x * density_cdf(x), -INTEGRATION_LIMIT_SIGMA, INTEGRATION_LIMIT_SIGMA)[0]
    second = quad(lambda x: x * x * density_cdf(x), -INTEGRATION_LIMIT_SIGMA, INTEGRATION_LIMIT_SIGMA)[0]
    return mean, second - mean * mean


def child_stage_moments(
    children: List[int],
    response: Dict[int, float],
    variance: Dict[int, float],
    net_hop_ms: float,
) -> Tuple[float, float]:
    """Mean and variance of the downstream stage that a parent thread waits for."""
    if not children:
        return 0.0, 0.0
    arrivals = [net_hop_ms + response[child] for child in children]
    if len(children) == 1:
        return arrivals[0], variance[children[0]]
    child_variances = [variance[child] for child in children]
    spread = math.sqrt(sum(child_variances) / len(children))
    mean_max, var_max = iid_max_moments(len(children))
    return max(arrivals) + spread * mean_max, spread * spread * var_max


def m1_root_mean_and_scv(graph: nx.DiGraph, root_rate_rps: float, params: ModelParams) -> Tuple[float, float]:
    """Root mean response and squared coefficient of variation under M1."""
    order = list(nx.topological_sort(graph))
    visits = visit_counts(graph)
    cs2_local = service_scv(params.service_sigma)
    var_local = cs2_local * params.service_mean_ms ** 2
    response: Dict[int, float] = {}
    variance: Dict[int, float] = {}
    for node in reversed(order):
        downstream_mean, downstream_var = child_stage_moments(
            list(graph.successors(node)), response, variance, params.net_hop_ms
        )
        hold_ms = params.service_mean_ms + downstream_mean
        hold_var = var_local + downstream_var
        cs2_node = hold_var / (hold_ms * hold_ms)
        rho = min(root_rate_rps * visits[node] * hold_ms / MS_PER_S / params.pool_size, MAX_UTILIZATION)
        wait_ms = sakasegawa_wait_ms(rho, params.pool_size, params.arrival_scv, cs2_node, hold_ms)
        wait_var = wait_ms * wait_ms * (1.0 + cs2_node) / max(rho, WAIT_UTILIZATION_FLOOR)
        response[node] = hold_ms + wait_ms
        variance[node] = hold_var + wait_var
    root = order[0]
    return response[root], variance[root] / (response[root] * response[root])


def m1_root_quantile_ms(graph: nx.DiGraph, root_rate_rps: float, params: ModelParams) -> float:
    mean_ms, scv = m1_root_mean_and_scv(graph, root_rate_rps, params)
    return lognormal_quantile_ms(mean_ms, scv, ROOT_PROBABILITY)


def longest_root_path_vertices(graph: nx.DiGraph) -> int:
    """Number of vertices on the longest root-to-leaf path."""
    order = list(nx.topological_sort(graph))
    length: Dict[int, int] = {}
    for node in reversed(order):
        length[node] = 1 + max((length[child] for child in graph.successors(node)), default=0)
    return length[order[0]]


def path_scv_root_quantile_ms(graph: nx.DiGraph, root_rate_rps: float, params: ModelParams) -> float:
    """Primary-model root mean with the root shape replaced by the path SCV, c_s^2 / L."""
    mean_ms = predict(graph, root_rate_rps, params).root_mean_ms
    cs2_path = service_scv(params.service_sigma) / longest_root_path_vertices(graph)
    return lognormal_quantile_ms(mean_ms, cs2_path, ROOT_PROBABILITY)
