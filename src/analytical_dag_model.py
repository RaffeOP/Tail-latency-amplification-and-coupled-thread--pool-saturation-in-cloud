"""Coupled G/G/m holding-time model for bounded microservice DAGs.

Arrivals are exogenous: vertex v receives root_rate * visits(v) requests per second. Each vertex
is a G/G/m station with m worker threads. A thread is held for the vertex's local work and for
the response of its slowest child, so downstream queueing enters the upstream holding times.
The waiting term is the Sakasegawa-type approximation for G/G/m (Whitt, 1993). Downstream
arrivals are treated with the root's arrival SCV, which is a stated simplification.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import math
from dataclasses import dataclass, field
from typing import Dict

import networkx as nx

MS_PER_S = 1000.0
MAX_UTILIZATION = 0.999
PERCENTILE_Z = {0.5: 0.0, 0.9: 1.2816, 0.99: 2.3263, 0.999: 3.0902, 0.9999: 3.7190}


@dataclass(frozen=True)
class ModelParams:
    pool_size: int
    service_mean_ms: float
    service_sigma: float
    net_hop_ms: float
    cpu_per_visit_ms: float
    host_cores: int
    arrival_scv: float = 1.0


@dataclass
class VertexPrediction:
    visits: int = 0
    hold_ms: float = 0.0
    response_ms: float = 0.0
    rho_pool: float = 0.0
    rho_cpu: float = 0.0


@dataclass
class DagPrediction:
    root_rate_rps: float
    root_mean_ms: float
    root_quantiles_ms: Dict[float, float] = field(default_factory=dict)
    vertices: Dict[int, VertexPrediction] = field(default_factory=dict)

    @property
    def max_rho_pool(self) -> float:
        return max(vertex.rho_pool for vertex in self.vertices.values())


def service_scv(sigma: float) -> float:
    """Squared coefficient of variation of a lognormal service time."""
    return math.exp(sigma * sigma) - 1.0


def visit_counts(graph: nx.DiGraph) -> Dict[int, int]:
    """Number of root-to-vertex paths, i.e. calls one root request makes to each vertex."""
    order = list(nx.topological_sort(graph))
    counts = {node: 0 for node in graph.nodes()}
    counts[order[0]] = 1
    for node in order:
        for child in graph.successors(node):
            counts[child] += counts[node]
    return counts


def sakasegawa_wait_ms(rho: float, servers: int, ca2: float, cs2: float, mean_hold_ms: float) -> float:
    capped = min(rho, MAX_UTILIZATION)
    exponent = math.sqrt(2.0 * (servers + 1.0)) - 1.0
    factor = (ca2 + cs2) / 2.0
    return factor * capped ** exponent / (servers * (1.0 - capped)) * mean_hold_ms


def lognormal_quantile_ms(mean_ms: float, scv: float, prob: float) -> float:
    sigma_sq = math.log(1.0 + scv)
    mu = math.log(mean_ms) - 0.5 * sigma_sq
    return math.exp(mu + math.sqrt(sigma_sq) * PERCENTILE_Z[prob])


def predict(graph: nx.DiGraph, root_rate_rps: float, params: ModelParams) -> DagPrediction:
    order = list(nx.topological_sort(graph))
    visits = visit_counts(graph)
    cs2 = service_scv(params.service_sigma)
    response: Dict[int, float] = {}
    prediction = DagPrediction(root_rate_rps=root_rate_rps, root_mean_ms=0.0)
    for node in reversed(order):
        children = list(graph.successors(node))
        downstream_ms = max((params.net_hop_ms + response[child] for child in children), default=0.0)
        hold_ms = params.service_mean_ms + downstream_ms
        rate_rps = root_rate_rps * visits[node]
        rho_pool = min(rate_rps * hold_ms / MS_PER_S / params.pool_size, MAX_UTILIZATION)
        wait_ms = sakasegawa_wait_ms(rho_pool, params.pool_size, params.arrival_scv, cs2, hold_ms)
        response[node] = wait_ms + hold_ms
        prediction.vertices[node] = VertexPrediction(
            visits=visits[node],
            hold_ms=hold_ms,
            response_ms=response[node],
            rho_pool=rho_pool,
            rho_cpu=rate_rps * params.cpu_per_visit_ms / (MS_PER_S * params.host_cores),
        )
    root = order[0]
    prediction.root_mean_ms = response[root]
    prediction.root_quantiles_ms = {
        prob: lognormal_quantile_ms(response[root], cs2, prob) for prob in PERCENTILE_Z
    }
    return prediction
