"""Executes one RunSpec end to end: starts the mesh (and sidecars), drives open-loop load, stores the result."""

import asyncio
import gc
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import networkx as nx
import uvloop

from dag_topology_builder import build_topology, compute_critical_path_floor
from host_gate import is_contaminated
from load_generator import LoadPlan, LoadResult, run_open_loop
from mesh_config import (
    DEFAULT_CLIENT_TIMEOUT_S,
    EXPORTER_NONE,
    FAULT_ERROR_PROB,
    FAULT_HANG_MS,
    FAULT_NONE,
    FAULT_SLOW_FACTOR,
    NET_HOP_MS,
    RUNTIME_PROXY,
    SHUTDOWN_GRACE_S,
    ChildRef,
    NodeSpec,
    RunSpec,
)
from run_storage import summarize, write_run_folder
from run_telemetry import ProcessSampler, host_noise_probe_p99_ms

PORT_BASE = 31000
PROXY_PORT_BASE = 40000
PORT_SLOT_COUNT = 40
PORT_SLOT_SIZE = 64
PORT_WAIT_TIMEOUT_S = 30.0
PORT_POLL_S = 0.05
SAMPLE_INTERVAL_S = 0.5
START_DELAY_S = 6.0
NS_PER_S = 1_000_000_000
ROOT_VERTEX = 0
TRACE_DIR_NAME = "spans"
NODE_SCRIPT = Path(__file__).resolve().parent / "node_server.py"
STATUS_OK = "ok"
STATUS_ABORTED = "aborted_deadline"
STATUS_FAILED = "failed"
STATUS_CONTAMINATED = "contaminated"
CONTAMINATED_SUFFIX = ".contaminated"


@dataclass(frozen=True)
class Layout:
    node_base: int
    proxy_base: int
    origin_ns: int
    trace_dir: Path
    floors: Dict[int, float]


@dataclass(frozen=True)
class DriveSpec:
    run: RunSpec
    root_port: int
    origin_ns: int
    node_pids: List[int]
    proxy_pids: List[int]


def port_layout(slot: int) -> Tuple[int, int]:
    offset = (slot % PORT_SLOT_COUNT) * PORT_SLOT_SIZE
    return PORT_BASE + offset, PROXY_PORT_BASE + offset


def critical_floors(graph: nx.DiGraph, run: RunSpec) -> Dict[int, float]:
    baseline = {vertex: run.service_mean_ms + NET_HOP_MS for vertex in graph.nodes()}
    return compute_critical_path_floor(graph, baseline)


def fault_window(run: RunSpec, origin_ns: int) -> Tuple[int, int]:
    span_ns = int(run.duration_s * NS_PER_S)
    return origin_ns + int(run.fault_start_frac * span_ns), origin_ns + int(run.fault_end_frac * span_ns)


def vertex_node_spec(run: RunSpec, graph: nx.DiGraph, vertex: int, layout: Layout) -> NodeSpec:
    child_base = layout.proxy_base if run.sidecar else layout.node_base
    children = [
        ChildRef(vertex_id=child, port=child_base + child, floor_ms=layout.floors[child])
        for child in sorted(graph.successors(vertex))
    ]
    fault_on = run.fault != FAULT_NONE and vertex == run.fault_target
    start_ns, end_ns = fault_window(run, layout.origin_ns)
    traced = run.tracing_rate > 0.0
    return NodeSpec(
        vertex_id=vertex,
        port=layout.node_base + vertex,
        children=children,
        runtime=run.runtime,
        pool_size=run.pool_size,
        service_mean_ms=run.service_mean_ms,
        service_sigma=run.service_sigma,
        cpu_burn_fraction=run.cpu_burn_fraction,
        seed=run.seed + vertex,
        instrument=run.sidecar,
        tracing_rate=run.tracing_rate,
        trace_exporter=run.trace_exporter if traced else EXPORTER_NONE,
        trace_path=str(layout.trace_dir / f"spans_{vertex}.jsonl") if traced else "",
        fault=run.fault if fault_on else FAULT_NONE,
        fault_start_ns=start_ns,
        fault_end_ns=end_ns,
        fault_factor=FAULT_SLOW_FACTOR,
        fault_error_prob=FAULT_ERROR_PROB,
        fault_hang_ms=FAULT_HANG_MS,
        controller=run.controller,
        min_remaining_ms=layout.floors[vertex],
        max_retries=run.max_retries,
    )


def proxy_spec(run: RunSpec, vertex: int, layout: Layout) -> NodeSpec:
    return NodeSpec(
        vertex_id=vertex,
        port=layout.proxy_base + vertex,
        children=[ChildRef(vertex_id=vertex, port=layout.node_base + vertex)],
        runtime=RUNTIME_PROXY,
        pool_size=run.pool_size,
        service_mean_ms=run.service_mean_ms,
        service_sigma=run.service_sigma,
        cpu_burn_fraction=run.cpu_burn_fraction,
        seed=run.seed + vertex,
    )


def port_is_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.2):
            return True
    except OSError:
        return False


def port_is_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def wait_for_ports(ports: List[int], timeout_s: float) -> bool:
    deadline = time.monotonic() + timeout_s
    pending = set(ports)
    while pending and time.monotonic() < deadline:
        pending = {port for port in pending if not port_is_open(port)}
        if pending:
            time.sleep(PORT_POLL_S)
    return not pending


def spawn_nodes(specs: List[NodeSpec], work_dir: Path) -> List[subprocess.Popen]:
    processes = []
    for spec in specs:
        spec_path = work_dir / f"node_{spec.vertex_id}_{spec.runtime}_{spec.port}.json"
        spec.write_json(spec_path)
        with (work_dir / f"node_{spec.vertex_id}_{spec.runtime}_{spec.port}.log").open("w") as log:
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(NODE_SCRIPT), str(spec_path)],
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
    return processes


def stop_nodes(processes: List[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + SHUTDOWN_GRACE_S * 3
    for process in processes:
        remaining_s = max(0.0, deadline - time.monotonic())
        try:
            process.wait(timeout=remaining_s)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()


def drive_load(drive: DriveSpec, should_abort: Callable[[], bool]) -> Tuple[LoadResult, List[dict]]:
    run = drive.run
    plan = LoadPlan(
        root_port=drive.root_port,
        root_vertex=ROOT_VERTEX,
        rate_rps=run.arrival_rate_rps,
        duration_s=run.duration_s,
        seed=run.seed,
        payload_bytes=run.payload_bytes,
        timeout_s=DEFAULT_CLIENT_TIMEOUT_S,
        sla_ms=run.sla_ms,
        origin_ns=drive.origin_ns,
    )
    gc.collect()
    gc.freeze()
    sampler = ProcessSampler(drive.node_pids, SAMPLE_INTERVAL_S, proxy_pids=drive.proxy_pids)
    sampler.start()
    try:
        with asyncio.Runner(loop_factory=uvloop.new_event_loop) as runner:
            load = runner.run(run_open_loop(plan, should_abort))
    finally:
        sampler.stop()
    return load, sampler.rows()


def execute_run(run: RunSpec, run_dir: Path, slot: int, should_abort: Callable[[], bool]) -> str:
    graph = build_topology(run.topology, run.topology_args)
    work_dir = run_dir.with_name(run_dir.name + ".partial")
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)
    trace_dir = work_dir / TRACE_DIR_NAME
    if run.tracing_rate > 0.0:
        trace_dir.mkdir()
    noise_probe_ms = host_noise_probe_p99_ms()
    origin_ns = time.time_ns() + int(START_DELAY_S * NS_PER_S)
    node_base, proxy_base = port_layout(slot)
    layout = Layout(node_base, proxy_base, origin_ns, trace_dir, critical_floors(graph, run))
    specs = [vertex_node_spec(run, graph, vertex, layout) for vertex in sorted(graph.nodes())]
    proxies = [proxy_spec(run, vertex, layout) for vertex in sorted(graph.nodes())] if run.sidecar else []
    ports = [spec.port for spec in specs + proxies]
    if not all(port_is_free(port) for port in ports):
        return STATUS_FAILED
    processes = spawn_nodes(specs + proxies, work_dir)
    try:
        if not wait_for_ports(ports, PORT_WAIT_TIMEOUT_S) or time.time_ns() >= origin_ns:
            return STATUS_FAILED
        drive = DriveSpec(run, node_base + ROOT_VERTEX, origin_ns, [p.pid for p in processes[:len(specs)]],
                          [p.pid for p in processes[len(specs):]])
        load, telemetry = drive_load(drive, should_abort)
        nodes_healthy = all(process.poll() is None for process in processes)
    finally:
        stop_nodes(processes)
    if load.generator_aborted:
        shutil.rmtree(work_dir)
        return STATUS_ABORTED
    if not nodes_healthy or not telemetry:
        return STATUS_FAILED
    summary = summarize(load, int(run.warmup_s * NS_PER_S))
    summary["host_noise_p99_ms"] = noise_probe_ms
    write_run_folder(work_dir, run, load, telemetry, summary)
    if is_contaminated(telemetry):
        return keep_contaminated(work_dir, run_dir)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    work_dir.rename(run_dir)
    return STATUS_OK


def keep_contaminated(work_dir: Path, run_dir: Path) -> str:
    """Keeps a disturbed run for diagnosis under a separate name, so it never reaches the manifest."""
    kept_dir = run_dir.with_name(run_dir.name + CONTAMINATED_SUFFIX)
    if kept_dir.exists():
        shutil.rmtree(kept_dir)
    work_dir.rename(kept_dir)
    return STATUS_CONTAMINATED
