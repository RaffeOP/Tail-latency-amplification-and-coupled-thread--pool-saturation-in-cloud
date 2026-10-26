"""Run matrices for the revision phases: Phase 4b (interleaved tracing) and Phase 5b (dacshed_retry).

Phase 4b repeats the tracing comparison as randomised blocks, so each traced run is paired with a
baseline recorded in the same block. Phase 5b adds the post hoc dacshed_retry controller on the
chain and the diamond in-cast. Both use the campaign protocol from campaign_matrix.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import random
from dataclasses import replace
from typing import List

from campaign_matrix import (
    PHASE5_DEPTH,
    PHASE5_FAULT_TARGET,
    PHASE5_FRACTIONS,
    PHASE5_SLA_MS,
    base_run,
    chain_capacity_rps,
    analytic_grid,
    stable_seed,
)
from dag_topology_builder import build_topology
from mesh_config import (
    CONTROLLER_DACSHED,
    CONTROLLER_DACSHED_RETRY,
    CONTROLLER_NAIVE_RETRY,
    CONTROLLER_STATIC_CB,
    EXPORTER_BATCH,
    EXPORTER_NONE,
    EXPORTER_SYNC,
    FAULT_ERRORS,
    FAULT_HANG,
    FAULT_SLOWDOWN,
    NAIVE_RETRY_ATTEMPTS,
    RunSpec,
)

PHASE_4B = 6
PHASE_5B = 7
EXTRA_PHASES = (PHASE_4B, PHASE_5B)
PHASE4B_DEPTH = 4
PHASE4B_LOADS = (("low", 0.2), ("high", 0.6))
PHASE4B_BLOCKS = 5
PHASE4B_RATES = (0.10, 0.25, 0.50, 1.00)
PHASE4B_EXPORTERS = (EXPORTER_BATCH, EXPORTER_SYNC)
PHASE5B_FAULTS = (FAULT_SLOWDOWN, FAULT_ERRORS, FAULT_HANG)
PHASE5B_CHAIN_REPLICATES = 5
PHASE5B_DIAMOND = {"depth": 3, "width": 4}
PHASE5B_DIAMOND_FRACTIONS = (0.5, 0.8)
PHASE5B_DIAMOND_REPLICATES = 3
PHASE5B_DIAMOND_CONTROLLERS = (
    CONTROLLER_STATIC_CB, CONTROLLER_NAIVE_RETRY, CONTROLLER_DACSHED, CONTROLLER_DACSHED_RETRY,
)
DACSHED_RETRY_ATTEMPTS = 1
PERCENT = 100


def retries_for(controller: str) -> int:
    if controller == CONTROLLER_NAIVE_RETRY:
        return NAIVE_RETRY_ATTEMPTS
    if controller == CONTROLLER_DACSHED_RETRY:
        return DACSHED_RETRY_ATTEMPTS
    return 0


def phase4b_configs() -> List[tuple]:
    configs = [(EXPORTER_NONE, 0.0)]
    for rate in PHASE4B_RATES:
        for exporter in PHASE4B_EXPORTERS:
            configs.append((exporter, rate))
    return configs


def phase4b_runs() -> List[RunSpec]:
    """Nine configurations per block, shuffled within each block with a seed from load and block."""
    capacity = chain_capacity_rps(PHASE4B_DEPTH)
    label = f"chain_D{PHASE4B_DEPTH}"
    runs: List[RunSpec] = []
    for load_name, fraction in PHASE4B_LOADS:
        rate = round(capacity * fraction, 3)
        for block in range(1, PHASE4B_BLOCKS + 1):
            order = phase4b_configs()
            random.Random(stable_seed(f"4b-{load_name}-{block}")).shuffle(order)
            for exporter, rate_value in order:
                percent = round(rate_value * PERCENT)
                run_id = f"p4b-{load_name}-b{block}-{exporter}-rate{percent}"
                run = base_run(PHASE_4B, run_id, label, {"depth": PHASE4B_DEPTH}, rate, block)
                runs.append(replace(run, tracing_rate=rate_value, trace_exporter=exporter))
    return runs


def phase5b_chain_runs() -> List[RunSpec]:
    capacity = chain_capacity_rps(PHASE5_DEPTH)
    label = f"chain_D{PHASE5_DEPTH}"
    runs: List[RunSpec] = []
    for fault in PHASE5B_FAULTS:
        for level, fraction in enumerate(PHASE5_FRACTIONS, start=1):
            rate = round(capacity * fraction, 3)
            for replicate in range(1, PHASE5B_CHAIN_REPLICATES + 1):
                run_id = f"p5b-chain-{fault}-{CONTROLLER_DACSHED_RETRY}-L{level}-r{replicate}"
                run = base_run(PHASE_5B, run_id, label, {"depth": PHASE5_DEPTH}, rate, replicate)
                runs.append(replace(
                    run, fault=fault, fault_target=PHASE5_FAULT_TARGET, controller=CONTROLLER_DACSHED_RETRY,
                    sla_ms=PHASE5_SLA_MS, max_retries=DACSHED_RETRY_ATTEMPTS,
                ))
    return runs


def diamond_shared_leaf(depth: int, width: int) -> int:
    """Lowest-numbered vertex with more than one parent, i.e. a shared in-cast leaf."""
    graph = build_topology("diamond", {"depth": depth, "width": width})
    shared = sorted(node for node in graph.nodes() if graph.in_degree(node) > 1)
    return shared[0]


def phase5b_diamond_runs() -> List[RunSpec]:
    graph = build_topology("diamond", dict(PHASE5B_DIAMOND))
    capacity = analytic_grid(graph)["pool_limit_rps"]
    target = diamond_shared_leaf(PHASE5B_DIAMOND["depth"], PHASE5B_DIAMOND["width"])
    label = f"diamond_D{PHASE5B_DIAMOND['depth']}_W{PHASE5B_DIAMOND['width']}"
    runs: List[RunSpec] = []
    for controller in PHASE5B_DIAMOND_CONTROLLERS:
        for level, fraction in enumerate(PHASE5B_DIAMOND_FRACTIONS, start=1):
            rate = round(capacity * fraction, 3)
            for replicate in range(1, PHASE5B_DIAMOND_REPLICATES + 1):
                run_id = f"p5b-diamond-{FAULT_SLOWDOWN}-{controller}-L{level}-r{replicate}"
                run = base_run(PHASE_5B, run_id, label, dict(PHASE5B_DIAMOND), rate, replicate)
                runs.append(replace(
                    run, topology="diamond", topology_args=dict(PHASE5B_DIAMOND), fault=FAULT_SLOWDOWN,
                    fault_target=target, controller=controller, sla_ms=PHASE5_SLA_MS,
                    max_retries=retries_for(controller),
                ))
    return runs


def phase5b_runs() -> List[RunSpec]:
    return phase5b_chain_runs() + phase5b_diamond_runs()


def extra_phase_runs(phase: int) -> List[RunSpec]:
    builders = {PHASE_4B: phase4b_runs, PHASE_5B: phase5b_runs}
    return builders[phase]()
