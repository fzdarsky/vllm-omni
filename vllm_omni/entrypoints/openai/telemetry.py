# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OpenTelemetry setup for World Model endpoints.

Provides tracing for V-JEPA inference sessions with JWMSP span names.
Falls back to no-op implementations when OTel is not available.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Any, Generator

try:
    from opentelemetry import trace
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import (
        OTLPSpanExporter,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    _HAS_OTEL = True
except ImportError:
    _HAS_OTEL = False


_tracer = None
_initialized = False


def init_world_telemetry(
    service_name: str = "vllm-omni-jepa",
    otlp_endpoint: str | None = None,
) -> None:
    """Initialize OpenTelemetry tracing for World Model endpoints.

    Args:
        service_name: Service name for traces
        otlp_endpoint: OTLP gRPC endpoint (e.g. http://localhost:4317)
    """
    global _tracer, _initialized

    if _initialized:
        return

    if not _HAS_OTEL:
        _initialized = True
        return

    # Get endpoint from env if not provided
    if otlp_endpoint is None:
        otlp_endpoint = os.environ.get("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

    if otlp_endpoint is None:
        _initialized = True
        return

    resource = Resource.create(
        {
            "service.name": service_name,
            "service.version": "0.1.0",
        }
    )

    tracer_provider = TracerProvider(resource=resource)
    span_exporter = OTLPSpanExporter(endpoint=otlp_endpoint, insecure=True)
    tracer_provider.add_span_processor(BatchSpanProcessor(span_exporter))
    trace.set_tracer_provider(tracer_provider)
    _tracer = trace.get_tracer("vllm-omni-world")
    _initialized = True


def get_tracer() -> Any:
    """Return the OTel tracer, or a no-op tracer if OTel is not available."""
    if _tracer is not None:
        return _tracer
    if _HAS_OTEL:
        return trace.get_tracer("vllm-omni-world")
    return _NoOpTracer()


class _NoOpTracer:
    """Stub tracer that returns a no-op span context manager."""

    def start_as_current_span(self, name: str, **kwargs):
        return _NoOpSpanContext()

    def start_span(self, name: str, **kwargs):
        return _NoOpSpan()


class _NoOpSpanContext:
    """Context manager that does nothing."""

    def __enter__(self):
        return _NoOpSpan()

    def __exit__(self, *args):
        pass


class _NoOpSpan:
    """Stub span that does nothing."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def end(self) -> None:
        pass


@contextmanager
def trace_span(name: str, **attributes) -> Generator[Any, None, None]:
    """Context manager for creating a trace span.

    Args:
        name: Span name (use JWMSP names: input_receive, input_decode, etc.)
        **attributes: Span attributes
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as span:
        for key, value in attributes.items():
            span.set_attribute(key, value)
        yield span


def record_span(
    name: str,
    start_time_ns: int,
    end_time_ns: int,
    **attributes,
) -> None:
    """Record a span with explicit start/end times.

    Useful for recording timing from other threads where trace context
    isn't available. Creates the span linked to the current trace context.

    Args:
        name: Span name
        start_time_ns: Start time in nanoseconds (perf_counter_ns)
        end_time_ns: End time in nanoseconds (perf_counter_ns)
        **attributes: Span attributes
    """
    if not _HAS_OTEL:
        return

    tracer = get_tracer()
    if isinstance(tracer, _NoOpTracer):
        return

    # Convert perf_counter to wall clock time for OTel
    # OTel expects time in nanoseconds since epoch
    import time
    now_ns = time.time_ns()
    now_perf = time.perf_counter_ns()
    offset = now_ns - now_perf

    # Get current context to link as parent
    from opentelemetry import context as otel_context
    current_ctx = otel_context.get_current()

    span = tracer.start_span(
        name,
        context=current_ctx,
        start_time=start_time_ns + offset,
    )
    for key, value in attributes.items():
        span.set_attribute(key, value)
    span.end(end_time=end_time_ns + offset)
