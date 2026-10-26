"""Open-loop Poisson load generator with coordinated-omission-corrected latencies.

Every request has an intended send time drawn from the Poisson schedule before the run starts.
Latency is measured from that intended time, so a generator that falls behind inflates the
recorded latency instead of silently reducing the offered load.
"""

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import grpc
from grpc import aio as grpc_aio

import mesh_service_pb2 as pb
import mesh_service_pb2_grpc as pb_grpc
from node_common import GRPC_OPTIONS, STATUS_OK

NS_PER_S = 1_000_000_000
NS_PER_US = 1_000
NS_PER_MS = 1_000_000
OUTCOME_OK = 0
OUTCOME_ERROR = 1
OUTCOME_TIMEOUT = 2
HOP_SAMPLE_STRIDE = 10
HopRow = Tuple[int, int, int, int, int, int, int, int]


@dataclass(frozen=True)
class LoadPlan:
    root_port: int
    root_vertex: int
    rate_rps: float
    duration_s: float
    seed: int
    payload_bytes: int
    timeout_s: float
    sla_ms: float = 0.0
    origin_ns: int = 0


@dataclass
class LoadResult:
    scheduled: int = 0
    generator_aborted: bool = False
    request_index: List[int] = field(default_factory=list)
    intended_offset_ns: List[int] = field(default_factory=list)
    lag_ns: List[int] = field(default_factory=list)
    latency_us: List[int] = field(default_factory=list)
    outcome: List[int] = field(default_factory=list)
    hop_rows: List[HopRow] = field(default_factory=list)


def poisson_offsets_s(rate_rps: float, duration_s: float, rng: random.Random) -> List[float]:
    offsets: List[float] = []
    clock_s = rng.expovariate(rate_rps)
    while clock_s < duration_s:
        offsets.append(clock_s)
        clock_s += rng.expovariate(rate_rps)
    return offsets


async def sleep_until(target_ns: int) -> None:
    delay_s = (target_ns - time.time_ns()) / NS_PER_S
    if delay_s > 0:
        await asyncio.sleep(delay_s)


class OpenLoopClient:
    def __init__(self, stub, plan: LoadPlan, origin_ns: int):
        self._stub = stub
        self._plan = plan
        self._origin_ns = origin_ns
        self._payload = b"\x00" * plan.payload_bytes
        self.result = LoadResult()

    async def issue(self, index: int, intended_ns: int) -> None:
        sent_ns = time.time_ns()
        request = pb.HopRequest(
            request_id=f"r{index}",
            current_vertex_id=self._plan.root_vertex,
            ingress_arrival_ns=sent_ns,
            absolute_deadline_ns=self._deadline_ns(intended_ns),
            payload_blob=self._payload,
        )
        outcome, timelines = await self._call(request)
        self._record(index, intended_ns, sent_ns, time.time_ns(), outcome, timelines)

    def _deadline_ns(self, intended_ns: int) -> int:
        if self._plan.sla_ms <= 0.0:
            return 0
        return intended_ns + int(self._plan.sla_ms * NS_PER_MS)

    async def _call(self, request) -> Tuple[int, list]:
        try:
            response = await self._stub.ExecuteHop(request, timeout=self._plan.timeout_s)
        except grpc_aio.AioRpcError as error:
            timed_out = error.code() == grpc.StatusCode.DEADLINE_EXCEEDED
            return (OUTCOME_TIMEOUT if timed_out else OUTCOME_ERROR), []
        outcome = OUTCOME_OK if response.status_code == STATUS_OK else OUTCOME_ERROR
        return outcome, list(response.hop_timelines)

    def _record(self, index, intended_ns, sent_ns, end_ns, outcome, timelines) -> None:
        result = self.result
        result.request_index.append(index)
        result.intended_offset_ns.append(intended_ns - self._origin_ns)
        result.lag_ns.append(sent_ns - intended_ns)
        result.latency_us.append((end_ns - intended_ns) // NS_PER_US)
        result.outcome.append(outcome)
        if index % HOP_SAMPLE_STRIDE == 0:
            result.hop_rows.extend(
                (
                    index, timeline.vertex_id, timeline.pool_wait_us, timeline.business_proc_us,
                    timeline.total_hop_us, timeline.deser_cpu_us, timeline.ser_cpu_us, timeline.proxy_hop_us,
                )
                for timeline in timelines
            )


async def run_open_loop(plan: LoadPlan, should_abort: Callable[[], bool]) -> LoadResult:
    offsets = poisson_offsets_s(plan.rate_rps, plan.duration_s, random.Random(plan.seed))
    channel = grpc_aio.insecure_channel(f"127.0.0.1:{plan.root_port}", options=GRPC_OPTIONS)
    origin_ns = plan.origin_ns or time.time_ns()
    await sleep_until(origin_ns)
    client = OpenLoopClient(pb_grpc.MeshWorkerStub(channel), plan, origin_ns)
    tasks = []
    aborted = False
    for index, offset_s in enumerate(offsets):
        if should_abort():
            aborted = True
            break
        intended_ns = origin_ns + int(offset_s * NS_PER_S)
        await sleep_until(intended_ns)
        tasks.append(asyncio.create_task(client.issue(index, intended_ns)))
    if aborted:
        for task in tasks:
            task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)
    await channel.close()
    client.result.scheduled = len(tasks)
    client.result.generator_aborted = aborted
    return client.result
