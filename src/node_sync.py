"""Synchronous node runtime: each in-flight request holds a handler thread across its downstream calls.

Optional features are off unless the node spec enables them, so the Phase 1 and 2 behavior is
unchanged: instrumented wire encoding, OpenTelemetry tracing, a fault window on this vertex,
server-side overload control, and client-side retries.
"""

import concurrent.futures as futures
import random
import signal
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import grpc

import mesh_service_pb2 as pb
import mesh_service_pb2_grpc as pb_grpc
from mesh_config import DEFAULT_CHILD_TIMEOUT_S, SHUTDOWN_GRACE_S, NodeSpec
from node_common import GRPC_OPTIONS, STATUS_FAILED, STATUS_INJECTED, STATUS_OK, HopMarks, child_request, hop_timeline
from node_controls import build_overload
from node_faults import FaultPlan, fails_under_fault, hang_ms_under_fault, service_ms_under_fault
from node_retry import RetryGate
from node_tracing import HopTracer, build_tracer
from node_wire import UNMEASURED_INT, CallResult, bytes_method, bytes_service_handler, call_bytes
from service_work import ServiceTimeSampler, blocking_work

HANDLER_THREAD_PREFIX = "hop"
FANOUT_THREAD_PREFIX = "fanout"
TERMINATION_POLL_S = 0.5
MS_PER_S = 1000.0
NO_RESULT = CallResult(None, UNMEASURED_INT, UNMEASURED_INT, UNMEASURED_INT)


class StubCaller:
    """Calls a child through the generated stub; wire timing is not measured."""

    def __init__(self, stub):
        self._stub = stub

    def call(self, request) -> CallResult:
        try:
            response = self._stub.ExecuteHop(request, timeout=DEFAULT_CHILD_TIMEOUT_S)
        except grpc.RpcError:
            return NO_RESULT
        return CallResult(response, UNMEASURED_INT, UNMEASURED_INT, UNMEASURED_INT)


class BytesCaller:
    """Calls a child with explicit, timed protobuf encoding and decoding."""

    def __init__(self, method):
        self._method = method

    def call(self, request) -> CallResult:
        try:
            return call_bytes(self._method, request)
        except grpc.RpcError:
            return NO_RESULT


@dataclass(frozen=True)
class ChildTarget:
    caller: object
    vertex_id: int
    floor_ms: float


def make_caller(instrument: bool, channel: grpc.Channel):
    if instrument:
        return BytesCaller(bytes_method(channel))
    return StubCaller(pb_grpc.MeshWorkerStub(channel))


def first_traceparent(request) -> str:
    return request.span_context_stack[0] if len(request.span_context_stack) else ""


def hop_status(has_failed: bool, failed_by_fault: bool) -> int:
    """An error injected at this vertex is reported apart from one propagated from a child."""
    if failed_by_fault:
        return STATUS_INJECTED
    if has_failed:
        return STATUS_FAILED
    return STATUS_OK


def call_child(caller, request, target: ChildTarget, traceparent: str, max_retries: int,
               retry_gate: Optional[RetryGate]) -> Optional[CallResult]:
    for attempt in range(max_retries + 1):
        result = caller.call(child_request(request, target.vertex_id, traceparent))
        if result.response is not None and result.response.status_code == STATUS_OK:
            return result
        if retry_gate is not None and not retry_gate.allows_retry(
            attempt, result, request.absolute_deadline_ns, target.floor_ms,
        ):
            return None
    return None


class SyncMeshWorker(pb_grpc.MeshWorkerServicer):
    def __init__(self, spec: NodeSpec, targets: List[ChildTarget], fanout_pool: futures.Executor,
                 tracer: Optional[HopTracer], fault: FaultPlan, retry_gate: Optional[RetryGate]):
        self._vertex_id = spec.vertex_id
        self._targets = targets
        self._fanout_pool = fanout_pool
        self._tracer = tracer
        self._fault = fault
        self._retry_gate = retry_gate
        self._fault_rng = random.Random(spec.seed)
        self._instrument = spec.instrument
        self._max_retries = spec.max_retries
        self._burn_fraction = spec.cpu_burn_fraction
        self._sampler = ServiceTimeSampler(spec.service_mean_ms, spec.service_sigma, spec.seed)

    def ExecuteHop(self, request, context):
        return self.handle(request, UNMEASURED_INT)

    def handle(self, request, decode_us: int):
        start_ns = time.time_ns()
        span = self._tracer.begin(first_traceparent(request)) if self._tracer else None
        self._inject_hang(start_ns)
        service_ms = service_ms_under_fault(self._fault, self._sampler.sample_ms(), start_ns)
        blocking_work(service_ms, self._burn_fraction)
        failed_by_fault = fails_under_fault(self._fault, time.time_ns(), self._fault_rng)
        work_done_ns = time.time_ns()
        traceparent = self._tracer.child_traceparent(span) if span else ""
        child_timelines, has_failed, encode_us = self._call_children(request, traceparent)
        end_ns = time.time_ns()
        if span:
            self._tracer.finish(span)
        own = hop_timeline(self._vertex_id, HopMarks(request.ingress_arrival_ns, start_ns, work_done_ns, end_ns))
        if self._instrument:
            own.deser_cpu_us = decode_us
            own.ser_cpu_us = encode_us
        return pb.HopResponse(
            request_id=request.request_id,
            status_code=hop_status(has_failed, failed_by_fault),
            hop_timelines=[own] + child_timelines,
        )

    def _inject_hang(self, now_ns: int) -> None:
        hang_ms = hang_ms_under_fault(self._fault, now_ns)
        if hang_ms > 0.0:
            time.sleep(hang_ms / MS_PER_S)

    def _call_children(self, request, traceparent: str) -> Tuple[List[pb.HopTimeline], bool, int]:
        if not self._targets:
            return [], False, 0
        pending = [
            self._fanout_pool.submit(self._call_one, target, request, traceparent) for target in self._targets
        ]
        timelines: List[pb.HopTimeline] = []
        has_failed = False
        encode_us = 0
        for future in pending:
            outcome = future.result()
            if outcome is None:
                has_failed = True
                continue
            child_timelines, child_encode_us = outcome
            timelines.extend(child_timelines)
            encode_us += max(0, child_encode_us)
        return timelines, has_failed, encode_us

    def _call_one(self, target: ChildTarget, request, traceparent: str):
        result = call_child(target.caller, request, target, traceparent, self._max_retries, self._retry_gate)
        if result is None:
            return None
        timelines = list(result.response.hop_timelines)
        if self._instrument and timelines and result.proxy_us >= 0:
            timelines[0].proxy_hop_us = result.proxy_us
        return timelines, result.encode_us


def fault_plan(spec: NodeSpec) -> FaultPlan:
    return FaultPlan(spec.fault, spec.fault_start_ns, spec.fault_end_ns, spec.fault_factor,
                     spec.fault_error_prob, spec.fault_hang_ms)


def register_servicer(spec: NodeSpec, worker: SyncMeshWorker, server: grpc.Server) -> None:
    if spec.instrument:
        server.add_generic_rpc_handlers((bytes_service_handler(worker.handle),))
    else:
        pb_grpc.add_MeshWorkerServicer_to_server(worker, server)


def serve_sync(spec: NodeSpec) -> None:
    handler_pool = futures.ThreadPoolExecutor(max_workers=spec.pool_size, thread_name_prefix=HANDLER_THREAD_PREFIX)
    fanout_size = spec.pool_size * max(1, len(spec.children))
    fanout_pool = futures.ThreadPoolExecutor(max_workers=fanout_size, thread_name_prefix=FANOUT_THREAD_PREFIX)
    channels = [grpc.insecure_channel(f"127.0.0.1:{child.port}", options=GRPC_OPTIONS) for child in spec.children]
    targets = [
        ChildTarget(make_caller(spec.instrument, channel), child.vertex_id, child.floor_ms)
        for channel, child in zip(channels, spec.children)
    ]
    tracer = build_tracer(spec.tracing_rate, spec.trace_exporter, spec.trace_path)
    interceptors, retry_gate = build_overload(spec.controller, spec.min_remaining_ms)
    worker = SyncMeshWorker(spec, targets, fanout_pool, tracer, fault_plan(spec), retry_gate)
    server = grpc.server(handler_pool, options=GRPC_OPTIONS, interceptors=interceptors)
    register_servicer(spec, worker, server)
    server.add_insecure_port(f"127.0.0.1:{spec.port}")
    server.start()
    stop_event = threading.Event()
    signal.signal(signal.SIGTERM, lambda signum, frame: stop_event.set())
    while not stop_event.is_set():
        stop_event.wait(TERMINATION_POLL_S)
    server.stop(SHUTDOWN_GRACE_S).wait()
    fanout_pool.shutdown(wait=False, cancel_futures=True)
    handler_pool.shutdown(wait=False, cancel_futures=True)
    if tracer is not None:
        tracer.shutdown()
    for channel in channels:
        channel.close()
