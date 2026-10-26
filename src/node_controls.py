"""Server-side overload controls for the Phase 5 runs, applied as gRPC interceptors.

The DAC-Shed variant with a client-side retry gate shares one controller instance between its
interceptor and the gate, so the gate reads the same latency gradient the controller computes.
"""

from typing import List, Optional, Tuple

from dac_shed_interceptor import OverloadInterceptor
from mesh_config import CONTROLLER_DACSHED, CONTROLLER_DACSHED_RETRY, CONTROLLER_STATIC_CB
from node_retry import RetryGate
from overload_controller import DacShedController, EnvoyStyleCircuitBreaker


def build_overload(controller: str, min_remaining_ms: float) -> Tuple[List[OverloadInterceptor], Optional[RetryGate]]:
    if controller == CONTROLLER_STATIC_CB:
        return [OverloadInterceptor(EnvoyStyleCircuitBreaker(), 0.0)], None
    if controller in (CONTROLLER_DACSHED, CONTROLLER_DACSHED_RETRY):
        dacshed = DacShedController()
        interceptors = [OverloadInterceptor(dacshed, min_remaining_ms)]
        if controller == CONTROLLER_DACSHED_RETRY:
            return interceptors, RetryGate(lambda: dacshed.gradient)
        return interceptors, None
    return [], None
