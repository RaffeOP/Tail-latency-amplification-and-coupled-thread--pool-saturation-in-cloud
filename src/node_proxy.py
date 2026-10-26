"""Sidecar proxy runtime: a bytes-level pass-through in front of one vertex.

The proxy forwards the raw request and response unchanged, so it adds a hop and a copy without
parsing protobuf. Its cost is measured from the caller's side (see node_wire.edge_overhead_us)
and from its own process CPU in the telemetry. This is an in-process emulation of a sidecar,
not Envoy.
"""

import concurrent.futures as futures
import signal
import threading

import grpc

from mesh_config import DEFAULT_CHILD_TIMEOUT_S, SHUTDOWN_GRACE_S, NodeSpec
from node_common import GRPC_OPTIONS
from node_wire import METHOD_NAME, SERVICE_NAME, bytes_method

TERMINATION_POLL_S = 0.5
PROXY_THREAD_PREFIX = "proxy"


def forward_behavior(target):
    def behavior(raw: bytes, context) -> bytes:
        try:
            response_raw, _ = target.with_call(raw, timeout=DEFAULT_CHILD_TIMEOUT_S)
        except grpc.RpcError as error:
            context.abort(error.code(), error.details() or "upstream failure")
        return response_raw
    return behavior


def passthrough_handler(target):
    rpc = grpc.unary_unary_rpc_method_handler(
        forward_behavior(target), request_deserializer=None, response_serializer=None,
    )
    return grpc.method_handlers_generic_handler(SERVICE_NAME, {METHOD_NAME: rpc})


def serve_proxy(spec: NodeSpec) -> None:
    channel = grpc.insecure_channel(f"127.0.0.1:{spec.children[0].port}", options=GRPC_OPTIONS)
    pool = futures.ThreadPoolExecutor(max_workers=spec.pool_size, thread_name_prefix=PROXY_THREAD_PREFIX)
    server = grpc.server(pool, options=GRPC_OPTIONS)
    server.add_generic_rpc_handlers((passthrough_handler(bytes_method(channel)),))
    server.add_insecure_port(f"127.0.0.1:{spec.port}")
    server.start()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())
    while not stop_event.is_set():
        stop_event.wait(TERMINATION_POLL_S)
    server.stop(SHUTDOWN_GRACE_S).wait()
    pool.shutdown(wait=False, cancel_futures=True)
    channel.close()
