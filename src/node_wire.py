"""Explicit wire encoding with CPU timing, plus the bytes-level service and call helpers.

The instrumented runtime exchanges raw bytes so that protobuf encoding and decoding can be timed
per hop. The edge overhead of a call is the caller's wall time minus the child's own reported
hop time. It contains the sidecar's handling and the transport on both ends, and it excludes the
child's processing.
"""

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import grpc

import mesh_service_pb2 as pb
from mesh_config import DEFAULT_CHILD_TIMEOUT_S

SERVICE_NAME = "cloudmesh.MeshWorker"
METHOD_NAME = "ExecuteHop"
METHOD_PATH = f"/{SERVICE_NAME}/{METHOD_NAME}"
NS_PER_US = 1_000
UNMEASURED_INT = -1


@dataclass(frozen=True)
class CallResult:
    response: Optional[pb.HopResponse]
    encode_us: int
    decode_us: int
    proxy_us: int


def encode_timed(message) -> Tuple[bytes, int]:
    start_ns = time.thread_time_ns()
    raw = message.SerializeToString()
    return raw, (time.thread_time_ns() - start_ns) // NS_PER_US


def decode_timed(raw: bytes, message_type) -> Tuple[object, int]:
    start_ns = time.thread_time_ns()
    message = message_type.FromString(raw)
    return message, (time.thread_time_ns() - start_ns) // NS_PER_US


def edge_overhead_us(call_wall_us: int, child_total_us: int) -> int:
    """Caller wall time not explained by the child's own processing: sidecar plus transport."""
    if child_total_us < 0:
        return UNMEASURED_INT
    return max(0, call_wall_us - child_total_us)


def bytes_method(channel: grpc.Channel):
    """Raw-bytes ExecuteHop callable: no serializer, so encoding is timed explicitly."""
    return channel.unary_unary(METHOD_PATH, request_serializer=None, response_deserializer=None)


def call_bytes(method, request: pb.HopRequest) -> CallResult:
    raw, encode_us = encode_timed(request)
    started_ns = time.perf_counter_ns()
    response_raw, _ = method.with_call(raw, timeout=DEFAULT_CHILD_TIMEOUT_S)
    wall_us = (time.perf_counter_ns() - started_ns) // NS_PER_US
    response, decode_us = decode_timed(response_raw, pb.HopResponse)
    child_total_us = response.hop_timelines[0].total_hop_us if len(response.hop_timelines) else UNMEASURED_INT
    return CallResult(response, encode_us, decode_us, edge_overhead_us(wall_us, child_total_us))


def bytes_service_handler(behavior_for_request):
    """gRPC generic handler that receives raw bytes and returns raw bytes.

    behavior_for_request(request, decode_us) must return a pb.HopResponse.
    """
    def behavior(raw: bytes, context) -> bytes:
        request, decode_us = decode_timed(raw, pb.HopRequest)
        return behavior_for_request(request, decode_us).SerializeToString()

    rpc = grpc.unary_unary_rpc_method_handler(behavior, request_deserializer=None, response_serializer=None)
    return grpc.method_handlers_generic_handler(SERVICE_NAME, {METHOD_NAME: rpc})
