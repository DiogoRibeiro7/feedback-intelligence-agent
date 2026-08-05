"""Streaming ingestion primitives for feedback events.

The local path is deterministic and dependency-free: JSONL or in-memory events
are validated into :class:`FeedbackRecord` objects. Kafka and Kinesis adapters
are thin optional wrappers around provider SDKs and are imported lazily, so the
default install stays lightweight.
"""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from pathlib import Path
from types import ModuleType
from typing import Any, Protocol, cast

from pydantic import BaseModel, Field, ValidationError

from feedback_intelligence_agent.data_contracts import REQUIRED_COLUMNS, ValidationIssue
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.telemetry import Telemetry

__all__ = [
    "FeedbackStream",
    "InMemoryFeedbackStream",
    "JsonlFeedbackStream",
    "KafkaFeedbackStream",
    "KinesisFeedbackStream",
    "StreamEnvelope",
    "StreamIngestionError",
    "StreamIngestionIssue",
    "StreamIngestionResult",
    "StreamOffset",
    "consume_feedback_stream",
    "write_stream_records_csv",
]


class StreamIngestionError(RuntimeError):
    """Raised when a streaming provider cannot be configured or consumed."""


class StreamOffset(BaseModel):
    """Opaque position of one stream event."""

    partition: str
    offset: str


class StreamEnvelope(BaseModel):
    """One raw feedback event read from a stream."""

    payload: dict[str, Any]
    source: str
    offset: StreamOffset


class StreamIngestionIssue(BaseModel):
    """Validation failure for one stream message."""

    source: str
    offset: StreamOffset
    message: str
    errors: list[ValidationIssue] = Field(default_factory=list)
    payload: dict[str, Any] = Field(default_factory=dict)


class StreamIngestionResult(BaseModel):
    """Summary of consuming a bounded stream batch."""

    source: str
    total_messages: int = Field(ge=0)
    accepted_records: int = Field(ge=0)
    rejected_messages: int = Field(ge=0)
    checkpoints: list[StreamOffset] = Field(default_factory=list)
    issues: list[StreamIngestionIssue] = Field(default_factory=list)
    records: list[FeedbackRecord] = Field(default_factory=list)


class FeedbackStream(Protocol):
    """Protocol implemented by bounded feedback stream consumers."""

    source: str

    def poll(self, max_messages: int) -> list[StreamEnvelope]:
        """Return up to ``max_messages`` envelopes."""
        ...

    def ack(self, envelope: StreamEnvelope) -> None:
        """Commit or acknowledge one successfully processed envelope."""
        ...


class InMemoryFeedbackStream:
    """Deterministic stream backed by an iterable of payload dictionaries."""

    source = "memory"

    def __init__(self, payloads: Iterable[dict[str, Any]]) -> None:
        """Store payloads and initialise the read cursor."""
        self._payloads = list(payloads)
        self._cursor = 0
        self.acked_offsets: list[StreamOffset] = []

    def poll(self, max_messages: int) -> list[StreamEnvelope]:
        """Return the next bounded batch of in-memory events."""
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        selected = self._payloads[self._cursor : self._cursor + max_messages]
        envelopes = [
            StreamEnvelope(
                payload=payload,
                source=self.source,
                offset=StreamOffset(partition="0", offset=str(self._cursor + index)),
            )
            for index, payload in enumerate(selected)
        ]
        self._cursor += len(envelopes)
        return envelopes

    def ack(self, envelope: StreamEnvelope) -> None:
        """Record the acknowledged offset for test and local inspection."""
        self.acked_offsets.append(envelope.offset)


class JsonlFeedbackStream(InMemoryFeedbackStream):
    """Local JSONL file treated as a finite stream of feedback events."""

    def __init__(self, path: str | Path) -> None:
        """Load JSON objects from a JSONL file."""
        self.path = Path(path)
        payloads: list[dict[str, Any]] = []
        for line_number, line in enumerate(
            self.path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise StreamIngestionError(
                    f"Invalid JSON in stream file {self.path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(payload, dict):
                raise StreamIngestionError(
                    f"Invalid JSON in stream file {self.path} at line {line_number}: "
                    "expected an object."
                )
            payloads.append(payload)
        super().__init__(payloads)
        self.source = str(self.path)


class KafkaConsumerProtocol(Protocol):
    """Small protocol for confluent-kafka consumer behavior used here."""

    def poll(self, timeout: float) -> object | None:
        """Poll for one message."""
        ...

    def commit(self, message: object, asynchronous: bool = False) -> object:
        """Commit one consumed message."""
        ...


class KafkaMessageProtocol(Protocol):
    """Small protocol for Kafka message methods used here."""

    def value(self) -> bytes | str:
        """Return the raw message value."""
        ...

    def topic(self) -> str:
        """Return the message topic."""
        ...

    def partition(self) -> int:
        """Return the topic partition."""
        ...

    def offset(self) -> int:
        """Return the partition offset."""
        ...


class KafkaFeedbackStream:
    """Kafka-backed feedback stream using the optional ``confluent-kafka`` package."""

    def __init__(
        self,
        *,
        topic: str,
        bootstrap_servers: str,
        group_id: str,
        consumer: KafkaConsumerProtocol | None = None,
        poll_timeout_seconds: float = 1.0,
    ) -> None:
        """Configure a Kafka consumer or use an injected test consumer."""
        self.source = f"kafka:{topic}"
        self.poll_timeout_seconds = poll_timeout_seconds
        if consumer is not None:
            self.consumer = consumer
            return
        kafka = _import_confluent_kafka()
        self.consumer = cast(
            KafkaConsumerProtocol,
            kafka.Consumer(
                {
                    "bootstrap.servers": bootstrap_servers,
                    "group.id": group_id,
                    "auto.offset.reset": "earliest",
                    "enable.auto.commit": False,
                }
            ),
        )
        subscribe = getattr(self.consumer, "subscribe", None)
        if callable(subscribe):
            subscribe([topic])

    def poll(self, max_messages: int) -> list[StreamEnvelope]:
        """Poll Kafka messages and decode JSON feedback payloads."""
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        envelopes: list[StreamEnvelope] = []
        for _ in range(max_messages):
            message = self.consumer.poll(self.poll_timeout_seconds)
            if message is None:
                break
            error = getattr(message, "error", lambda: None)()
            if error:
                raise StreamIngestionError(f"Kafka consumer error: {error}")
            envelopes.append(_kafka_envelope(message, self.source))
        return envelopes

    def ack(self, envelope: StreamEnvelope) -> None:
        """Commit one Kafka message by offset when the consumer supports it."""
        raw = envelope.payload.get("_raw_message")
        if raw is not None:
            self.consumer.commit(raw, asynchronous=False)


class KinesisClientProtocol(Protocol):
    """Small protocol for Kinesis client methods used here."""

    def get_records(self, **kwargs: object) -> dict[str, Any]:
        """Read records from a shard iterator."""
        ...


class KinesisFeedbackStream:
    """Kinesis-backed feedback stream using the optional ``boto3`` package."""

    def __init__(
        self,
        *,
        stream_name: str,
        shard_iterator: str,
        client: KinesisClientProtocol | None = None,
        region_name: str | None = None,
    ) -> None:
        """Configure a Kinesis reader or use an injected test client."""
        self.source = f"kinesis:{stream_name}"
        self.stream_name = stream_name
        self.shard_iterator = shard_iterator
        self.acked_offsets: list[StreamOffset] = []
        if client is not None:
            self.client = client
            return
        boto3 = _import_boto3()
        self.client = cast(KinesisClientProtocol, boto3.client("kinesis", region_name=region_name))

    def poll(self, max_messages: int) -> list[StreamEnvelope]:
        """Read a bounded batch from Kinesis."""
        if max_messages < 1:
            raise ValueError("max_messages must be positive")
        response = self.client.get_records(
            ShardIterator=self.shard_iterator,
            Limit=max_messages,
        )
        next_iterator = response.get("NextShardIterator")
        if isinstance(next_iterator, str):
            self.shard_iterator = next_iterator
        records = response.get("Records", [])
        if not isinstance(records, list):
            raise StreamIngestionError("Kinesis response Records field was not a list.")
        return [_kinesis_envelope(record, self.source) for record in records]

    def ack(self, envelope: StreamEnvelope) -> None:
        """Record acknowledged Kinesis sequence numbers for checkpointing."""
        self.acked_offsets.append(envelope.offset)


def consume_feedback_stream(
    stream: FeedbackStream,
    *,
    max_messages: int,
    dead_letter_path: str | Path | None = None,
    telemetry: Telemetry | None = None,
) -> StreamIngestionResult:
    """Consume, validate, and checkpoint a bounded batch of feedback events."""
    if max_messages < 1:
        raise ValueError("max_messages must be positive")
    telemetry = telemetry or Telemetry()
    correlation_id = telemetry.new_correlation_id()
    with telemetry.span(
        "stream_ingestion_started",
        "stream_ingestion_finished",
        correlation_id=correlation_id,
        metadata={"source": stream.source, "max_messages": max_messages},
    ) as span:
        envelopes = stream.poll(max_messages)
        records: list[FeedbackRecord] = []
        issues: list[StreamIngestionIssue] = []
        seen_ids: set[str] = set()
        for envelope in envelopes:
            record, issue = _validate_envelope(envelope, seen_ids)
            if issue is not None:
                issues.append(issue)
                continue
            if record is None:
                continue
            seen_ids.add(record.feedback_id)
            records.append(record)
            stream.ack(envelope)

        if dead_letter_path is not None and issues:
            _write_dead_letters(dead_letter_path, issues)
        checkpoints = [record.offset for record in _accepted_envelopes(envelopes, issues)]
        result = StreamIngestionResult(
            source=stream.source,
            total_messages=len(envelopes),
            accepted_records=len(records),
            rejected_messages=len(issues),
            checkpoints=checkpoints,
            issues=issues,
            records=records,
        )
        span["total_messages"] = result.total_messages
        span["accepted_records"] = result.accepted_records
        span["rejected_messages"] = result.rejected_messages
    return result


def write_stream_records_csv(records: list[FeedbackRecord], path: str | Path) -> Path:
    """Write validated stream records to a CSV compatible with batch ingestion."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(REQUIRED_COLUMNS))
        writer.writeheader()
        for record in records:
            row = record.model_dump(mode="json")
            writer.writerow({column: row[column] for column in REQUIRED_COLUMNS})
    return output


def _validate_envelope(
    envelope: StreamEnvelope,
    seen_ids: set[str],
) -> tuple[FeedbackRecord | None, StreamIngestionIssue | None]:
    """Validate one stream envelope into a feedback record or issue."""
    try:
        record = FeedbackRecord.model_validate(envelope.payload)
    except ValidationError as exc:
        return None, StreamIngestionIssue(
            source=envelope.source,
            offset=envelope.offset,
            message="stream message failed the feedback data contract",
            errors=_validation_issues(exc),
            payload=_clean_payload(envelope.payload),
        )
    if record.feedback_id in seen_ids:
        return None, StreamIngestionIssue(
            source=envelope.source,
            offset=envelope.offset,
            message=f"duplicate feedback_id '{record.feedback_id}' in stream batch",
            errors=[
                ValidationIssue(
                    severity="error",
                    column="feedback_id",
                    message=f"duplicate feedback_id '{record.feedback_id}'",
                )
            ],
            payload=_clean_payload(envelope.payload),
        )
    return record, None


def _validation_issues(exc: ValidationError) -> list[ValidationIssue]:
    """Translate Pydantic validation errors into data-contract style issues."""
    issues: list[ValidationIssue] = []
    for error in exc.errors():
        field = str(error["loc"][0]) if error["loc"] else "message"
        issues.append(
            ValidationIssue(
                severity="error",
                column=field,
                message=str(error["msg"]),
            )
        )
    return issues


def _accepted_envelopes(
    envelopes: list[StreamEnvelope],
    issues: list[StreamIngestionIssue],
) -> list[StreamEnvelope]:
    """Return envelopes not represented by a validation issue."""
    rejected = {(issue.offset.partition, issue.offset.offset) for issue in issues}
    return [
        envelope
        for envelope in envelopes
        if (envelope.offset.partition, envelope.offset.offset) not in rejected
    ]


def _write_dead_letters(path: str | Path, issues: list[StreamIngestionIssue]) -> None:
    """Append rejected messages to a JSONL dead-letter file."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as handle:
        for issue in issues:
            handle.write(issue.model_dump_json() + "\n")


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Remove non-serialisable transport objects from a payload copy."""
    return {key: value for key, value in payload.items() if not key.startswith("_")}


def _kafka_envelope(message: object, source: str) -> StreamEnvelope:
    """Decode one confluent-kafka message into a stream envelope."""
    kafka_message = cast(KafkaMessageProtocol, message)
    raw_value = kafka_message.value()
    payload = _json_payload(raw_value, source=source)
    payload["_raw_message"] = message
    topic = str(kafka_message.topic())
    partition = str(kafka_message.partition())
    offset = str(kafka_message.offset())
    return StreamEnvelope(
        payload=payload,
        source=source,
        offset=StreamOffset(partition=f"{topic}:{partition}", offset=offset),
    )


def _kinesis_envelope(record: object, source: str) -> StreamEnvelope:
    """Decode one Kinesis record into a stream envelope."""
    if not isinstance(record, dict):
        raise StreamIngestionError("Kinesis record was not an object.")
    payload = _json_payload(record.get("Data"), source=source)
    partition = str(record.get("PartitionKey", "default"))
    sequence = str(record.get("SequenceNumber", "unknown"))
    return StreamEnvelope(
        payload=payload,
        source=source,
        offset=StreamOffset(partition=partition, offset=sequence),
    )


def _json_payload(raw_value: object, *, source: str) -> dict[str, Any]:
    """Decode bytes/string JSON into a payload dictionary."""
    if isinstance(raw_value, bytes):
        text = raw_value.decode("utf-8")
    elif isinstance(raw_value, str):
        text = raw_value
    else:
        raise StreamIngestionError(f"{source} record payload was not bytes or text.")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise StreamIngestionError(f"{source} record payload was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise StreamIngestionError(f"{source} record payload must be a JSON object.")
    return payload


def _import_confluent_kafka() -> ModuleType:
    """Import optional confluent-kafka with an actionable setup message."""
    try:
        import confluent_kafka
    except ImportError as exc:
        raise StreamIngestionError(
            "The 'confluent-kafka' package is required for Kafka ingestion. "
            "Install it with: poetry install --extras streaming."
        ) from exc
    module: ModuleType = confluent_kafka
    return module


def _import_boto3() -> ModuleType:
    """Import optional boto3 with an actionable Kinesis setup message."""
    try:
        import boto3
    except ImportError as exc:
        raise StreamIngestionError(
            "The 'boto3' package is required for Kinesis ingestion. "
            "Install it with: poetry install --extras streaming."
        ) from exc
    module: ModuleType = boto3
    return module
