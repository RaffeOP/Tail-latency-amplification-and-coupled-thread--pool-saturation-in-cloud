"""Entry point for one node process: python3 src/node_server.py <node-spec.json>."""

import asyncio
import concurrent.futures as futures
import sys
from pathlib import Path

import uvloop

from mesh_config import RUNTIME_HYBRID, RUNTIME_PROXY, RUNTIME_SYNC, NodeSpec
from node_aio import serve_aio
from node_proxy import serve_proxy
from node_sync import serve_sync


def run_node(spec: NodeSpec) -> None:
    if spec.runtime == RUNTIME_SYNC:
        serve_sync(spec)
        return
    if spec.runtime == RUNTIME_PROXY:
        serve_proxy(spec)
        return
    local_pool = futures.ThreadPoolExecutor(max_workers=spec.pool_size) if spec.runtime == RUNTIME_HYBRID else None
    with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
        runner.run(serve_aio(spec, local_pool))
    if local_pool is not None:
        local_pool.shutdown(wait=False)


if __name__ == "__main__":
    run_node(NodeSpec.read_json(Path(sys.argv[1])))
