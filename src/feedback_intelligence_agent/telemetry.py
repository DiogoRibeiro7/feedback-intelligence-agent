"""Structured logging and OpenTelemetry-style telemetry.

The module provides two layers:

1. Lightweight JSON logging helpers (:func:`configure_logging`, :func:`log_event`)
   that keep runtime logs consistent and easy to redirect to systems such as
   CloudWatch, Datadog, or OpenTelemetry collectors.
2. A structured telemetry pipeline (:class:`Telemetry` plus pluggable sinks)
   that records spans and events with timestamps, durations, and correlation
   IDs. Telemetry is a no-op unless a sink is attached, so the default code
   path stays free of side effects.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Protocol, cast

LOGGER_NAME = "feedback_intelligence_agent"
TelemetryAttributeValue = str | bool | int | float


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for scripts and local API usage."""
    logging.basicConfig(level=level, format="%(message)s")


def get_logger() -> logging.Logger:
    """Return the package logger."""
    return logging.getLogger(LOGGER_NAME)


def log_event(event: str, payload: Mapping[str, Any] | None = None) -> None:
    """Emit a JSON log event.

    Args:
        event: Stable event name.
        payload: Optional structured attributes.
    """
    logger = get_logger()
    body = {"event": event, "payload": dict(payload or {})}
    logger.info(json.dumps(body, sort_keys=True, default=str))


@dataclass(frozen=True)
class TelemetryEvent:
    """A single structured telemetry event.

    Attributes:
        name: Stable event name, e.g. ``retrieval_finished``.
        timestamp: ISO-8601 UTC timestamp of when the event was emitted.
        correlation_id: Identifier shared by all events of one logical operation.
        duration_ms: Elapsed time in milliseconds for ``*_finished`` events.
        metadata: Structured attributes describing the event.
    """

    name: str
    timestamp: str
    correlation_id: str
    duration_ms: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation of the event."""
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "duration_ms": self.duration_ms,
            "metadata": dict(self.metadata),
        }


class TelemetrySink(Protocol):
    """Protocol implemented by telemetry sinks."""

    def emit(self, event: TelemetryEvent) -> None:
        """Persist or forward one telemetry event."""
        ...


class OpenTelemetrySpan(Protocol):
    """Minimal span behavior used by the OpenTelemetry sink."""

    def set_attribute(self, key: str, value: TelemetryAttributeValue) -> None:
        """Set one scalar span attribute."""
        ...

    def set_status(self, status: object) -> None:
        """Set the span status."""
        ...

    def end(self) -> None:
        """Finish the span."""
        ...


class OpenTelemetryTracer(Protocol):
    """Minimal tracer behavior used by the OpenTelemetry sink."""

    def start_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, TelemetryAttributeValue],
    ) -> OpenTelemetrySpan:
        """Start a span with initial attributes."""
        ...


class InMemoryTelemetrySink:
    """Sink that keeps events in a list. Intended for tests and inspection."""

    def __init__(self) -> None:
        """Initialise the empty event buffer."""
        self.events: list[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        """Append the event to the in-memory buffer."""
        self.events.append(event)

    def event_names(self) -> list[str]:
        """Return the names of captured events in emission order."""
        return [event.name for event in self.events]


class JsonlTelemetrySink:
    """Sink that appends one JSON object per line to a local JSONL file."""

    def __init__(self, path: str | Path) -> None:
        """Create the sink and ensure the parent directory exists."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: TelemetryEvent) -> None:
        """Append the event as a single JSON line."""
        line = json.dumps(event.to_dict(), sort_keys=True, default=str)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


class TelemetryConfigurationError(RuntimeError):
    """Raised when an optional telemetry backend cannot be configured."""


class OpenTelemetryTraceSink:
    """Sink that converts telemetry event pairs into OpenTelemetry spans.

    The OpenTelemetry API is imported lazily so the default install remains
    lightweight. The sink uses the process-global tracer provider, which lets
    deployments configure exporters with standard OpenTelemetry environment
    variables or instrumentation wrappers.
    """

    def __init__(
        self,
        service_name: str = LOGGER_NAME,
        *,
        tracer: OpenTelemetryTracer | None = None,
    ) -> None:
        """Create a span sink for the configured service/tracer name."""
        self.tracer = tracer if tracer is not None else _build_opentelemetry_tracer(service_name)
        self._active_spans: dict[tuple[str, str], OpenTelemetrySpan] = {}
        self._lock = Lock()

    def emit(self, event: TelemetryEvent) -> None:
        """Start, finish, or emit an instantaneous OpenTelemetry span."""
        span_name = _otel_span_name(event.name)
        span_key = (event.correlation_id, span_name)
        if event.name.endswith("_started"):
            span = self.tracer.start_span(span_name, attributes=_otel_attributes(event))
            with self._lock:
                self._active_spans[span_key] = span
            return
        if event.name.endswith("_finished"):
            with self._lock:
                finished_span = (
                    self._active_spans.pop(span_key) if span_key in self._active_spans else None
                )
            if finished_span is None:
                finished_span = self.tracer.start_span(
                    span_name, attributes=_otel_attributes(event)
                )
            _set_otel_attributes(finished_span, _otel_attributes(event))
            _set_otel_status(finished_span, event)
            finished_span.end()
            return
        span = self.tracer.start_span(span_name, attributes=_otel_attributes(event))
        _set_otel_status(span, event)
        span.end()


class Telemetry:
    """Telemetry emitter that writes structured events to an optional sink.

    Without a sink the emitter is a cheap no-op, so instrumented code never
    needs to guard telemetry calls. Inject a sink (in-memory for tests, JSONL
    for local traces) to capture events.
    """

    def __init__(self, sink: TelemetrySink | None = None) -> None:
        """Attach an optional sink; ``None`` disables telemetry."""
        self.sink = sink

    @property
    def enabled(self) -> bool:
        """True when a sink is attached and events are recorded."""
        return self.sink is not None

    def new_correlation_id(self) -> str:
        """Return a fresh identifier used to correlate events of one operation."""
        return uuid.uuid4().hex

    def emit(
        self,
        name: str,
        *,
        correlation_id: str,
        duration_ms: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Emit one telemetry event to the configured sink, if any."""
        if self.sink is None:
            return
        event = TelemetryEvent(
            name=name,
            timestamp=datetime.now(timezone.utc).isoformat(),
            correlation_id=correlation_id,
            duration_ms=duration_ms,
            metadata=dict(metadata or {}),
        )
        self.sink.emit(event)

    @contextmanager
    def span(
        self,
        started_name: str,
        finished_name: str,
        *,
        correlation_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Emit a started/finished event pair around a block of work.

        Yields a mutable metadata dictionary so callers can attach result
        attributes (e.g. counts, scores) that should appear on the finished
        event. On error the finished event carries ``status: error`` and the
        exception message, and the exception is re-raised.
        """
        span_metadata: dict[str, Any] = dict(metadata or {})
        self.emit(started_name, correlation_id=correlation_id, metadata=span_metadata)
        start = time.perf_counter()
        try:
            yield span_metadata
        except Exception as exc:
            span_metadata["status"] = "error"
            span_metadata["error"] = str(exc)
            self.emit(
                finished_name,
                correlation_id=correlation_id,
                duration_ms=_elapsed_ms(start),
                metadata=span_metadata,
            )
            raise
        span_metadata.setdefault("status", "ok")
        self.emit(
            finished_name,
            correlation_id=correlation_id,
            duration_ms=_elapsed_ms(start),
            metadata=span_metadata,
        )


def _elapsed_ms(start: float) -> float:
    """Milliseconds elapsed since a ``time.perf_counter`` start value."""
    return round((time.perf_counter() - start) * 1000.0, 3)


def _build_opentelemetry_tracer(service_name: str) -> OpenTelemetryTracer:
    """Return the process-global OpenTelemetry tracer."""
    try:
        from opentelemetry import trace
    except ImportError as exc:
        raise TelemetryConfigurationError(
            "OpenTelemetry tracing requires the optional 'otel' extra. "
            "Install it with: poetry install --extras otel."
        ) from exc
    return cast(OpenTelemetryTracer, trace.get_tracer(service_name))


def _otel_span_name(event_name: str) -> str:
    """Convert event pair names into stable span names."""
    for suffix in ("_started", "_finished"):
        if event_name.endswith(suffix):
            return event_name.removesuffix(suffix)
    return event_name


def _otel_attributes(event: TelemetryEvent) -> dict[str, TelemetryAttributeValue]:
    """Convert event metadata into OpenTelemetry-safe scalar attributes."""
    attributes: dict[str, TelemetryAttributeValue] = {
        "fia.event_name": event.name,
        "fia.correlation_id": event.correlation_id,
    }
    if event.duration_ms is not None:
        attributes["fia.duration_ms"] = event.duration_ms
    for key, value in event.metadata.items():
        attributes[f"fia.{key}"] = _otel_attribute_value(value)
    return attributes


def _otel_attribute_value(value: object) -> TelemetryAttributeValue:
    """Return a scalar attribute value accepted by OpenTelemetry exporters."""
    if isinstance(value, str | bool | int | float):
        return value
    if value is None:
        return "null"
    return json.dumps(value, sort_keys=True, default=str)


def _set_otel_attributes(
    span: OpenTelemetrySpan, attributes: Mapping[str, TelemetryAttributeValue]
) -> None:
    """Apply attributes defensively to real or test OpenTelemetry spans."""
    set_attribute = getattr(span, "set_attribute", None)
    if not callable(set_attribute):
        return
    for key, value in attributes.items():
        set_attribute(key, value)


def _set_otel_status(span: OpenTelemetrySpan, event: TelemetryEvent) -> None:
    """Mark failed spans with OpenTelemetry error status when available."""
    if event.metadata.get("status") != "error":
        return
    set_status = getattr(span, "set_status", None)
    if not callable(set_status):
        return
    try:
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        return
    set_status(Status(StatusCode.ERROR, str(event.metadata.get("error", ""))))
