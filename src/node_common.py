"""Wire helpers shared by the synchronous and asyncio node runtimes."""

import time
from dataclasses import dataclass

import mesh_service_pb2 as pb
from mesh_config import MAX_MESSAGE_BYTES

GRPC_OPTIONS = (
    ("grpc.so_reuseport", 0),
    ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
)
STATUS_OK = 200
STATUS_FAILED = 500
STATUS_INJECTED = 503
NS_PER_US = 1_000
UNMEASURED_INT = -1
UNMEASURED_FLOAT = -1.0


@dataclass(frozen=True)
class HopMarks:
    arrival_ns: int
    start_ns: int
    work_done_ns: int
    end_ns: int


def child_request(parent: pb.HopRequest, vertex_id: int, traceparent: str = "") -> pb.HopRequest:
    return pb.HopRequest(
        request_id=parent.request_id,
        current_vertex_id=vertex_id,
        ingress_arrival_ns=time.time_ns(),
        absolute_deadline_ns=parent.absolute_deadline_ns,
        trace_sampling_flag=parent.trace_sampling_flag,
        payload_blob=parent.payload_blob,
        span_context_stack=[traceparent] if traceparent else [],
    )


def hop_timeline(vertex_id: int, marks: HopMarks) -> pb.HopTimeline:
    return pb.HopTimeline(
        vertex_id=vertex_id,
        pool_wait_us=(marks.start_ns - marks.arrival_ns) // NS_PER_US,
        business_proc_us=(marks.work_done_ns - marks.start_ns) // NS_PER_US,
        total_hop_us=(marks.end_ns - marks.start_ns) // NS_PER_US,
        deser_cpu_us=UNMEASURED_INT,
        ser_cpu_us=UNMEASURED_INT,
        proxy_hop_us=UNMEASURED_INT,
        active_pool_utilization=UNMEASURED_FLOAT,
        process_cpu_percent=UNMEASURED_FLOAT,
    )
