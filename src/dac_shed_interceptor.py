"""gRPC server interceptor that applies an overload controller before a unary handler runs."""

import time

import grpc

from overload_controller import OverloadController

STATUS_OK = 200
STATUS_FAILED = 500


class OverloadInterceptor(grpc.ServerInterceptor):
    def __init__(self, controller: OverloadController, min_remaining_ms: float):
        self._controller = controller
        self._min_remaining_ms = min_remaining_ms

    def intercept_service(self, continuation, handler_call_details):
        handler = continuation(handler_call_details)
        if handler is None or handler.unary_unary is None:
            return handler
        return grpc.unary_unary_rpc_method_handler(
            self._guarded(handler.unary_unary),
            request_deserializer=handler.request_deserializer,
            response_serializer=handler.response_serializer,
        )

    def _guarded(self, behavior):
        controller = self._controller
        min_remaining_ms = self._min_remaining_ms

        def guarded(request, context):
            decision = controller.admit(request.absolute_deadline_ns, min_remaining_ms)
            if not decision.admitted:
                context.abort(grpc.StatusCode.RESOURCE_EXHAUSTED, decision.reason)
            started_s = time.monotonic()
            failed = True
            try:
                response = behavior(request, context)
                failed = response.status_code != STATUS_OK
                return response
            finally:
                controller.complete((time.monotonic() - started_s) * 1000.0, failed)

        return guarded
