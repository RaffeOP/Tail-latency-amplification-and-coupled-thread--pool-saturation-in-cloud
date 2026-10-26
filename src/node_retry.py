"""Client-side retry gate for the dacshed_retry controller (Phase 5b, labelled post hoc).

One retry is allowed only when all four conditions hold: the call is the first attempt, the
callee returned an error injected at the callee (not a shed decision, a timeout, or a failure
propagated from deeper in the graph), the remaining SLA budget covers twice the callee's
critical-path floor, and the caller's latency gradient is below the DAC-Shed high threshold.
Shed decisions and timeouts arrive as gRPC errors with no response, so they never pass the gate.
Authors: Devarsh Patel and Ranjithkumar S (SCOPE, VIT Vellore).
"""

import time
from typing import Callable

from node_common import STATUS_INJECTED
from node_wire import CallResult
from overload_controller import DAC_THETA_HIGH

NS_PER_MS = 1_000_000
FIRST_ATTEMPT = 0
RETRY_BUDGET_FLOOR_MULTIPLE = 2.0


class RetryGate:
    def __init__(self, gradient_source: Callable[[], float]):
        self._gradient_source = gradient_source

    def allows_retry(self, attempt: int, result: CallResult, deadline_ns: int, child_floor_ms: float) -> bool:
        if attempt != FIRST_ATTEMPT:
            return False
        if result.response is None or result.response.status_code != STATUS_INJECTED:
            return False
        remaining_ms = (deadline_ns - time.time_ns()) / NS_PER_MS
        if remaining_ms < RETRY_BUDGET_FLOOR_MULTIPLE * child_floor_ms:
            return False
        return self._gradient_source() < DAC_THETA_HIGH
