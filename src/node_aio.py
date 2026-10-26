"""Asyncio node runtimes. Handlers never hold a thread across downstream calls.

The hybrid runtime keeps the asyncio server but runs local work in a bounded thread pool,
so only local processing time occupies that pool.
"""

import asyncio
import concurrent.futures as futures
import signal
import time
from typing import List, Optional, Tuple

from grpc import aio as grpc_aio

import mesh_service_pb2 as pb
import mesh_service_pb2_grpc as pb_grpc
from mesh_config import DEFAULT_CHILD_TIMEOUT_S, SHUTDOWN_GRACE_S, NodeSpec
from node_common import GRPC_OPTIONS, STATUS_FAILED, STATUS_OK, HopMarks, child_request, hop_timeline
from service_work import SECONDS_PER_MS, ServiceTimeSampler, blocking_work, burn_cpu, compensated_sleep_s

Target = Tuple[object, int]


class AioMeshWorker(pb_grpc.MeshWorkerServicer):
    def __init__(self, spec: NodeSpec, targets: List[Target], local_pool: Optional[futures.Executor]):
        self._vertex_id = spec.vertex_id
        self._targets = targets
        self._local_pool = local_pool
        self._burn_fraction = spec.cpu_burn_fraction
        self._sampler = ServiceTimeSampler(spec.service_mean_ms, spec.service_sigma, spec.seed)

    async def ExecuteHop(self, request, context):
        start_ns = time.time_ns()
        await self._local_work()
        work_done_ns = time.time_ns()
        child_timelines, has_failed = await self._call_children(request)
        marks = HopMarks(request.ingress_arrival_ns, start_ns, work_done_ns, time.time_ns())
        return pb.HopResponse(
            request_id=request.request_id,
            status_code=STATUS_FAILED if has_failed else STATUS_OK,
            hop_timelines=[hop_timeline(self._vertex_id, marks)] + child_timelines,
        )

    async def _local_work(self) -> None:
        service_ms = self._sampler.sample_ms()
        if self._local_pool is not None:
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(self._local_pool, blocking_work, service_ms, self._burn_fraction)
            return
        burn_s = service_ms * self._burn_fraction * SECONDS_PER_MS
        sleep_s = service_ms * SECONDS_PER_MS - burn_s
        await asyncio.sleep(compensated_sleep_s(sleep_s))
        burn_cpu(burn_s)

    async def _call_children(self, request) -> Tuple[List[pb.HopTimeline], bool]:
        if not self._targets:
            return [], False
        results = await asyncio.gather(
            *(self._call_one(stub, vertex_id, request) for stub, vertex_id in self._targets)
        )
        timelines = [timeline for result in results if result is not None for timeline in result]
        return timelines, any(result is None for result in results)

    @staticmethod
    async def _call_one(stub, vertex_id: int, request) -> Optional[List[pb.HopTimeline]]:
        try:
            response = await stub.ExecuteHop(child_request(request, vertex_id), timeout=DEFAULT_CHILD_TIMEOUT_S)
        except grpc_aio.AioRpcError:
            return None
        if response.status_code != STATUS_OK:
            return None
        return list(response.hop_timelines)


async def serve_aio(spec: NodeSpec, local_pool: Optional[futures.Executor]) -> None:
    server = grpc_aio.server(options=GRPC_OPTIONS)
    channels = [grpc_aio.insecure_channel(f"127.0.0.1:{child.port}", options=GRPC_OPTIONS) for child in spec.children]
    targets: List[Target] = [
        (pb_grpc.MeshWorkerStub(channel), child.vertex_id) for channel, child in zip(channels, spec.children)
    ]
    pb_grpc.add_MeshWorkerServicer_to_server(AioMeshWorker(spec, targets, local_pool), server)
    server.add_insecure_port(f"127.0.0.1:{spec.port}")
    await server.start()
    stop_event = asyncio.Event()
    asyncio.get_running_loop().add_signal_handler(signal.SIGTERM, stop_event.set)
    await stop_event.wait()
    await server.stop(SHUTDOWN_GRACE_S)
    for channel in channels:
        await channel.close()
