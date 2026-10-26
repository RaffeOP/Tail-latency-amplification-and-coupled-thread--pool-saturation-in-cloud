"""OpenTelemetry tracing for the mesh nodes: parent-based ratio sampling and sync or batch export.

Spans are exported as JSON lines, so the export path does real I/O. The synchronous processor
exports inside the request, and the batch processor exports from a background thread.
"""

import json
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor, SpanExporter, SpanExportResult
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace.propagation.tracecontext import TraceContextTextMapPropagator

from mesh_config import EXPORTER_SYNC

SPAN_NAME = "ExecuteHop"
TRACEPARENT_KEY = "traceparent"


class JsonlSpanExporter(SpanExporter):
    def __init__(self, path: str):
        self._handle = open(path, "a", buffering=1)

    def export(self, spans) -> SpanExportResult:
        for span in spans:
            context = span.context
            self._handle.write(json.dumps({
                "name": span.name,
                "trace_id": format(context.trace_id, "032x"),
                "span_id": format(context.span_id, "016x"),
                "start_ns": span.start_time,
                "end_ns": span.end_time,
            }) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        self._handle.close()


class HopTracer:
    def __init__(self, rate: float, exporter_mode: str, path: str):
        provider = TracerProvider(sampler=ParentBased(TraceIdRatioBased(rate)))
        exporter = JsonlSpanExporter(path)
        processor = SimpleSpanProcessor(exporter) if exporter_mode == EXPORTER_SYNC else BatchSpanProcessor(exporter)
        provider.add_span_processor(processor)
        self._provider = provider
        self._tracer = provider.get_tracer("cloudmesh.node")
        self._propagator = TraceContextTextMapPropagator()

    def begin(self, parent_traceparent: str):
        context = None
        if parent_traceparent:
            context = self._propagator.extract({TRACEPARENT_KEY: parent_traceparent})
        return self._tracer.start_span(SPAN_NAME, context=context)

    def child_traceparent(self, span) -> str:
        carrier: dict = {}
        self._propagator.inject(carrier, context=trace.set_span_in_context(span))
        return carrier.get(TRACEPARENT_KEY, "")

    def finish(self, span) -> None:
        span.end()

    def shutdown(self) -> None:
        self._provider.shutdown()


def build_tracer(rate: float, exporter_mode: str, path: str) -> Optional[HopTracer]:
    if rate <= 0.0 or not path:
        return None
    return HopTracer(rate, exporter_mode, path)
