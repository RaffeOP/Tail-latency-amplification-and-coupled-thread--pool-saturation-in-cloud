"""Fault windows for the Phase 5 injections on one target vertex: slowdown, errors, and hangs."""

import random
from dataclasses import dataclass

from mesh_config import FAULT_ERRORS, FAULT_HANG, FAULT_NONE, FAULT_SLOWDOWN


@dataclass(frozen=True)
class FaultPlan:
    mode: str
    start_ns: int
    end_ns: int
    factor: float
    error_prob: float
    hang_ms: float


def fault_active(plan: FaultPlan, now_ns: int) -> bool:
    return plan.mode != FAULT_NONE and plan.start_ns <= now_ns < plan.end_ns


def service_ms_under_fault(plan: FaultPlan, base_ms: float, now_ns: int) -> float:
    if fault_active(plan, now_ns) and plan.mode == FAULT_SLOWDOWN:
        return base_ms * plan.factor
    return base_ms


def hang_ms_under_fault(plan: FaultPlan, now_ns: int) -> float:
    if fault_active(plan, now_ns) and plan.mode == FAULT_HANG:
        return plan.hang_ms
    return 0.0


def fails_under_fault(plan: FaultPlan, now_ns: int, rng: random.Random) -> bool:
    if not fault_active(plan, now_ns) or plan.mode != FAULT_ERRORS:
        return False
    return rng.random() < plan.error_prob
