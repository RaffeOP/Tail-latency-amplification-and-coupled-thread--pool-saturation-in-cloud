"""Run and node specifications shared by the orchestrator and the node processes."""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List

RUNTIME_SYNC = "sync"
RUNTIME_AIO = "aio"
RUNTIME_HYBRID = "hybrid"
RUNTIME_PROXY = "proxy"
RUNTIME_MODELS = (RUNTIME_SYNC, RUNTIME_AIO, RUNTIME_HYBRID)

FAULT_NONE = "none"
FAULT_SLOWDOWN = "slowdown"
FAULT_ERRORS = "errors"
FAULT_HANG = "hang"
FAULT_MODES = (FAULT_NONE, FAULT_SLOWDOWN, FAULT_ERRORS, FAULT_HANG)
FAULT_SLOW_FACTOR = 4.0
FAULT_ERROR_PROB = 0.3
FAULT_HANG_MS = 1000.0

CONTROLLER_NONE = "none"
CONTROLLER_NAIVE_RETRY = "naive_retry"
CONTROLLER_STATIC_CB = "static_cb"
CONTROLLER_DACSHED = "dacshed"
CONTROLLER_DACSHED_RETRY = "dacshed_retry"
CONTROLLERS = (
    CONTROLLER_NONE, CONTROLLER_NAIVE_RETRY, CONTROLLER_STATIC_CB, CONTROLLER_DACSHED, CONTROLLER_DACSHED_RETRY,
)
NAIVE_RETRY_ATTEMPTS = 2

EXPORTER_NONE = "none"
EXPORTER_SYNC = "sync"
EXPORTER_BATCH = "batch"

DEFAULT_SERVICE_MEAN_MS = 10.0
DEFAULT_SERVICE_SIGMA = 0.5
DEFAULT_CPU_BURN_FRACTION = 0.1
DEFAULT_POOL_SIZE = 8
DEFAULT_PAYLOAD_BYTES = 64
DEFAULT_CHILD_TIMEOUT_S = 10.0
DEFAULT_CLIENT_TIMEOUT_S = 10.0
SHUTDOWN_GRACE_S = 1.0
NET_HOP_MS = 0.5
CPU_PER_VISIT_MS = 1.5
MAX_MESSAGE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class ChildRef:
    vertex_id: int
    port: int
    floor_ms: float = 0.0


@dataclass(frozen=True)
class NodeSpec:
    vertex_id: int
    port: int
    children: List[ChildRef]
    runtime: str
    pool_size: int
    service_mean_ms: float
    service_sigma: float
    cpu_burn_fraction: float
    seed: int
    instrument: bool = False
    tracing_rate: float = 0.0
    trace_exporter: str = EXPORTER_NONE
    trace_path: str = ""
    fault: str = FAULT_NONE
    fault_start_ns: int = 0
    fault_end_ns: int = 0
    fault_factor: float = 1.0
    fault_error_prob: float = 0.0
    fault_hang_ms: float = 0.0
    controller: str = CONTROLLER_NONE
    min_remaining_ms: float = 0.0
    max_retries: int = 0

    def write_json(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def read_json(cls, path: Path) -> "NodeSpec":
        raw = json.loads(path.read_text())
        children = [ChildRef(**child) for child in raw.pop("children")]
        return cls(children=children, **raw)


@dataclass
class RunSpec:
    run_id: str
    phase: int
    topology: str
    shape: str
    topology_args: Dict[str, int]
    runtime: str
    arrival_rate_rps: float
    duration_s: float
    warmup_s: float
    payload_bytes: int
    pool_size: int
    service_mean_ms: float
    service_sigma: float
    cpu_burn_fraction: float
    replicate: int
    seed: int
    extra: Dict[str, float] = field(default_factory=dict)
    sidecar: bool = False
    tracing_rate: float = 0.0
    trace_exporter: str = EXPORTER_NONE
    fault: str = FAULT_NONE
    fault_target: int = 2
    fault_start_frac: float = 0.35
    fault_end_frac: float = 0.65
    controller: str = CONTROLLER_NONE
    sla_ms: float = 0.0
    max_retries: int = 0
