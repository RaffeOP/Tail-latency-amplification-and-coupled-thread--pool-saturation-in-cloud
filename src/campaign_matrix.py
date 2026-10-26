"""Builds the factorial run matrix and derives each shape's analytic load grid.

The grid is derived from the DAG: a vertex saturates its worker pool when
visits(v) * hold(v) * arrival_rate reaches pool_size, and the host saturates when the
summed per-visit CPU work reaches the usable core budget.
"""

import hashlib
from dataclasses import replace
from typing import Dict, List, Tuple

import networkx as nx

from dag_topology_builder import build_topology
from mesh_config import (
    CPU_PER_VISIT_MS,
    DEFAULT_CPU_BURN_FRACTION,
    DEFAULT_PAYLOAD_BYTES,
    DEFAULT_POOL_SIZE,
    DEFAULT_SERVICE_MEAN_MS,
    DEFAULT_SERVICE_SIGMA,
    NET_HOP_MS,
    RUNTIME_AIO,
    RUNTIME_HYBRID,
    RUNTIME_SYNC,
    RunSpec,
)
from mesh_config import (
    CONTROLLER_DACSHED,
    CONTROLLER_NAIVE_RETRY,
    CONTROLLER_STATIC_CB,
    EXPORTER_BATCH,
    EXPORTER_NONE,
    EXPORTER_SYNC,
    FAULT_ERRORS,
    FAULT_HANG,
    FAULT_SLOWDOWN,
    NAIVE_RETRY_ATTEMPTS,
)

PHASE_RUN_COUNTS = {1: 180, 2: 135, 3: 75, 4: 60, 5: 90}
PHASE3_DEPTHS = (2, 4, 8)
PHASE3_PAYLOADS = (64, 1024, 4096, 16384, 65536)
PHASE3_REPLICATES = 5
PHASE3_RATE_RPS = 30.0
PHASE4_DEPTH = 4
PHASE4_FRACTIONS = (0.2, 0.6)
PHASE4_RATES = (0.0, 0.1, 0.25, 0.5, 1.0)
PHASE4_EXPORTERS = (EXPORTER_SYNC, EXPORTER_BATCH)
PHASE4_REPLICATES = 3
PHASE4_BASELINE_REPLICATES = 6
PHASE5_DEPTH = 4
PHASE5_FAULTS = (FAULT_SLOWDOWN, FAULT_ERRORS, FAULT_HANG)
PHASE5_CONTROLLERS = (CONTROLLER_STATIC_CB, CONTROLLER_NAIVE_RETRY, CONTROLLER_DACSHED)
PHASE5_FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)
PHASE5_REPLICATES = 2
PHASE5_SLA_MS = 250.0
PHASE5_FAULT_TARGET = 2
PHASE1_REPLICATES = 2
PHASE2_REPLICATES = 3
LOAD_LEVELS = 5
DEFAULT_DURATION_S = 90.0
DEFAULT_WARMUP_S = 10.0
HOST_CORES = 8
CPU_HEADROOM = 0.9
SHARED_GRID_POOL_MULTIPLE = 2.0
MS_PER_S = 1000.0
PILOT_DURATION_S = 15.0
PILOT_WARMUP_S = 2.0
SEED_DIGITS = 8
# D7 x W8 (43 vertices, 43 OS processes) exceeds the 8 GB host's memory budget, so D6 x W4 replaces it.
DIAMOND_SHAPES = ((3, 4), (3, 8), (5, 4), (5, 8), (6, 4), (7, 4))
PHASE2_RUNTIMES = (RUNTIME_SYNC, RUNTIME_AIO, RUNTIME_HYBRID)
PHASE2_SHAPES = (
    {"topology": "chain", "label": "chain_D8", "args": {"depth": 8}},
    {"topology": "scatter_gather", "label": "sg_D3_F4", "args": {"depth": 3, "fanout": 4}},
    {"topology": "diamond", "label": "diamond_D5_W8", "args": {"depth": 5, "width": 8}},
)


def stable_seed(text: str) -> int:
    return int(hashlib.sha256(text.encode()).hexdigest()[:SEED_DIGITS], 16)


def phase1_shapes() -> List[Dict[str, object]]:
    shapes: List[Dict[str, object]] = []
    for depth in (2, 4, 6, 8):
        shapes.append({"topology": "chain", "label": f"chain_D{depth}", "args": {"depth": depth}})
    for depth, fanout in ((2, 2), (2, 4), (2, 8), (3, 2), (3, 4), (4, 2)):
        shapes.append({
            "topology": "scatter_gather",
            "label": f"sg_D{depth}_F{fanout}",
            "args": {"depth": depth, "fanout": fanout},
        })
    for depth, width in DIAMOND_SHAPES:
        shapes.append({
            "topology": "diamond",
            "label": f"diamond_D{depth}_W{width}",
            "args": {"depth": depth, "width": width},
        })
    shapes.append({"topology": "deathstar_social", "label": "deathstar_social_12", "args": {}})
    shapes.append({"topology": "deathstar_hotel", "label": "deathstar_hotel_8", "args": {}})
    return shapes


def visits_and_holds(graph: nx.DiGraph) -> Tuple[Dict[int, int], Dict[int, float]]:
    order = list(nx.topological_sort(graph))
    visits = {node: 0 for node in graph.nodes()}
    visits[order[0]] = 1
    for node in order:
        for child in graph.successors(node):
            visits[child] += visits[node]
    holds: Dict[int, float] = {}
    for node in reversed(order):
        children = list(graph.successors(node))
        downstream = max((NET_HOP_MS + holds[child] for child in children), default=0.0)
        holds[node] = DEFAULT_SERVICE_MEAN_MS + downstream
    return visits, holds


def analytic_grid(graph: nx.DiGraph) -> Dict[str, object]:
    visits, holds = visits_and_holds(graph)
    pool_limits = [
        DEFAULT_POOL_SIZE / (visits[node] * holds[node] / MS_PER_S) for node in graph.nodes()
    ]
    total_visits = sum(visits.values())
    cpu_per_root_s = total_visits * CPU_PER_VISIT_MS / MS_PER_S
    cpu_limit = CPU_HEADROOM * HOST_CORES / cpu_per_root_s
    cap = min(min(pool_limits), cpu_limit)
    levels = [round(cap * step / LOAD_LEVELS, 3) for step in range(1, LOAD_LEVELS + 1)]
    return {
        "vertices": graph.number_of_nodes(),
        "total_visits": total_visits,
        "root_hold_ms": holds[0],
        "pool_limit_rps": min(pool_limits),
        "cpu_limit_rps": cpu_limit,
        "levels_rps": levels,
    }


def shared_levels(graph: nx.DiGraph) -> List[float]:
    """Load levels shared by all runtime models of one shape, so their curves are comparable."""
    grid = analytic_grid(graph)
    cap = min(grid["cpu_limit_rps"], SHARED_GRID_POOL_MULTIPLE * grid["pool_limit_rps"])
    return [round(cap * step / LOAD_LEVELS, 3) for step in range(1, LOAD_LEVELS + 1)]


def run_identifier(phase: int, label: str, runtime: str, level: int, replicate: int) -> str:
    """Phase 1 keeps the identifiers of the first campaign, which the manifest records."""
    if phase == 1:
        return f"p1-{label}-L{level}-r{replicate}"
    return f"p{phase}-{label}-{runtime}-L{level}-r{replicate}"


def build_run(phase: int, shape: Dict[str, object], runtime: str, level: int,
              replicate: int, rate_rps: float) -> RunSpec:
    run_id = run_identifier(phase, str(shape["label"]), runtime, level, replicate)
    return RunSpec(
        run_id=run_id,
        phase=phase,
        topology=str(shape["topology"]),
        shape=str(shape["label"]),
        topology_args=dict(shape["args"]),
        runtime=runtime,
        arrival_rate_rps=rate_rps,
        duration_s=DEFAULT_DURATION_S,
        warmup_s=DEFAULT_WARMUP_S,
        payload_bytes=DEFAULT_PAYLOAD_BYTES,
        pool_size=DEFAULT_POOL_SIZE,
        service_mean_ms=DEFAULT_SERVICE_MEAN_MS,
        service_sigma=DEFAULT_SERVICE_SIGMA,
        cpu_burn_fraction=DEFAULT_CPU_BURN_FRACTION,
        replicate=replicate,
        seed=stable_seed(run_id),
    )


def phase1_runs() -> List[RunSpec]:
    runs: List[RunSpec] = []
    for shape in phase1_shapes():
        grid = analytic_grid(build_topology(str(shape["topology"]), dict(shape["args"])))
        for level, rate in enumerate(grid["levels_rps"], start=1):
            for replicate in range(1, PHASE1_REPLICATES + 1):
                runs.append(build_run(1, shape, RUNTIME_SYNC, level, replicate, rate))
    return runs


def phase2_runs() -> List[RunSpec]:
    runs: List[RunSpec] = []
    for shape in PHASE2_SHAPES:
        levels = shared_levels(build_topology(str(shape["topology"]), dict(shape["args"])))
        for runtime in PHASE2_RUNTIMES:
            for level, rate in enumerate(levels, start=1):
                for replicate in range(1, PHASE2_REPLICATES + 1):
                    runs.append(build_run(2, shape, runtime, level, replicate, rate))
    return runs


def base_run(phase: int, run_id: str, label: str, args: Dict[str, int], rate_rps: float,
             replicate: int) -> RunSpec:
    return RunSpec(
        run_id=run_id,
        phase=phase,
        topology="chain",
        shape=label,
        topology_args=dict(args),
        runtime=RUNTIME_SYNC,
        arrival_rate_rps=rate_rps,
        duration_s=DEFAULT_DURATION_S,
        warmup_s=DEFAULT_WARMUP_S,
        payload_bytes=DEFAULT_PAYLOAD_BYTES,
        pool_size=DEFAULT_POOL_SIZE,
        service_mean_ms=DEFAULT_SERVICE_MEAN_MS,
        service_sigma=DEFAULT_SERVICE_SIGMA,
        cpu_burn_fraction=DEFAULT_CPU_BURN_FRACTION,
        replicate=replicate,
        seed=stable_seed(run_id),
    )


def chain_capacity_rps(depth: int) -> float:
    grid = analytic_grid(build_topology("chain", {"depth": depth}))
    return min(grid["pool_limit_rps"], grid["cpu_limit_rps"])


def phase3_runs() -> List[RunSpec]:
    """Serialization and sidecar tax: chain depth x payload size, sidecar on, low fixed load."""
    runs: List[RunSpec] = []
    for depth in PHASE3_DEPTHS:
        for payload in PHASE3_PAYLOADS:
            for replicate in range(1, PHASE3_REPLICATES + 1):
                run_id = f"p3-chain_D{depth}-pay{payload}-r{replicate}"
                run = base_run(3, run_id, f"chain_D{depth}", {"depth": depth}, PHASE3_RATE_RPS, replicate)
                runs.append(replace(run, payload_bytes=payload, sidecar=True))
    return runs


def phase4_runs() -> List[RunSpec]:
    """Tracing observer effect: sampling rate x exporter mode at two load levels."""
    capacity = chain_capacity_rps(PHASE4_DEPTH)
    label = f"chain_D{PHASE4_DEPTH}"
    runs: List[RunSpec] = []
    for name, fraction in zip(("low", "high"), PHASE4_FRACTIONS):
        rate = round(capacity * fraction, 3)
        for rate_value in PHASE4_RATES:
            percent = round(rate_value * 100)
            baseline = rate_value == 0.0
            exporters = (EXPORTER_NONE,) if baseline else PHASE4_EXPORTERS
            replicates = PHASE4_BASELINE_REPLICATES if baseline else PHASE4_REPLICATES
            for exporter in exporters:
                for replicate in range(1, replicates + 1):
                    run_id = f"p4-{label}-{name}-rate{percent}-{exporter}-r{replicate}"
                    run = base_run(4, run_id, label, {"depth": PHASE4_DEPTH}, rate, replicate)
                    runs.append(replace(run, tracing_rate=rate_value, trace_exporter=exporter))
    return runs


def phase5_runs() -> List[RunSpec]:
    """Fault injection with three controllers: static breaker, naive retries, and DAC-Shed."""
    capacity = chain_capacity_rps(PHASE5_DEPTH)
    label = f"chain_D{PHASE5_DEPTH}"
    runs: List[RunSpec] = []
    for fault in PHASE5_FAULTS:
        for controller in PHASE5_CONTROLLERS:
            retries = NAIVE_RETRY_ATTEMPTS if controller == CONTROLLER_NAIVE_RETRY else 0
            for level, fraction in enumerate(PHASE5_FRACTIONS, start=1):
                rate = round(capacity * fraction, 3)
                for replicate in range(1, PHASE5_REPLICATES + 1):
                    run_id = f"p5-{fault}-{controller}-L{level}-r{replicate}"
                    run = base_run(5, run_id, label, {"depth": PHASE5_DEPTH}, rate, replicate)
                    runs.append(replace(
                        run, fault=fault, fault_target=PHASE5_FAULT_TARGET, controller=controller,
                        sla_ms=PHASE5_SLA_MS, max_retries=retries,
                    ))
    return runs


def phase_runs(phase: int) -> List[RunSpec]:
    builders = {1: phase1_runs, 2: phase2_runs, 3: phase3_runs, 4: phase4_runs, 5: phase5_runs}
    if phase not in builders:
        raise ValueError(f"unknown phase {phase}")
    return builders[phase]()


def pilot_subset(runs: List[RunSpec]) -> List[RunSpec]:
    keep = {"chain_D8", "sg_D3_F4", "diamond_D5_W8", "deathstar_social_12"}
    selected = [
        run for run in runs
        if run.shape in keep and run.replicate == 1 and run.run_id.endswith(("-L1-r1", "-L5-r1"))
    ]
    return [replace(run, duration_s=PILOT_DURATION_S, warmup_s=PILOT_WARMUP_S) for run in selected]
