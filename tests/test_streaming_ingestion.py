from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from feedback_intelligence_agent.cli import app
from feedback_intelligence_agent.streaming_ingestion import (
    InMemoryFeedbackStream,
    JsonlFeedbackStream,
    KafkaFeedbackStream,
    KinesisFeedbackStream,
    StreamIngestionError,
    StreamOffset,
    consume_feedback_stream,
    write_stream_records_csv,
)
from feedback_intelligence_agent.telemetry import InMemoryTelemetrySink, Telemetry

runner = CliRunner()


def valid_payload(feedback_id: str = "fb-stream-1") -> dict[str, object]:
    return {
        "feedback_id": feedback_id,
        "customer_segment": "enterprise",
        "channel": "support_ticket",
        "rating": 2,
        "text": "Onboarding checklist was unclear in the streamed event.",
        "created_at": "2026-08-01T09:00:00Z",
    }


def test_in_memory_stream_consumes_valid_records_and_checkpoints() -> None:
    stream = InMemoryFeedbackStream([valid_payload("a"), valid_payload("b")])
    sink = InMemoryTelemetrySink()

    result = consume_feedback_stream(
        stream,
        max_messages=10,
        telemetry=Telemetry(sink=sink),
    )

    assert result.total_messages == 2
    assert result.accepted_records == 2
    assert result.rejected_messages == 0
    assert [record.feedback_id for record in result.records] == ["a", "b"]
    assert stream.acked_offsets == [
        StreamOffset(partition="0", offset="0"),
        StreamOffset(partition="0", offset="1"),
    ]
    assert sink.event_names() == ["stream_ingestion_started", "stream_ingestion_finished"]
    assert sink.events[-1].metadata["accepted_records"] == 2


def test_stream_validation_rejects_invalid_and_duplicate_messages(tmp_path: Path) -> None:
    dead_letter = tmp_path / "dead.jsonl"
    stream = InMemoryFeedbackStream(
        [
            valid_payload("dup"),
            {**valid_payload("bad"), "rating": 9},
            valid_payload("dup"),
        ]
    )

    result = consume_feedback_stream(stream, max_messages=10, dead_letter_path=dead_letter)

    assert result.accepted_records == 1
    assert result.rejected_messages == 2
    assert [issue.offset.offset for issue in result.issues] == ["1", "2"]
    assert stream.acked_offsets == [StreamOffset(partition="0", offset="0")]
    lines = dead_letter.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["message"] == "stream message failed the feedback data contract"
    assert "duplicate feedback_id" in json.loads(lines[1])["message"]


def test_jsonl_stream_reads_payloads(tmp_path: Path) -> None:
    path = tmp_path / "stream.jsonl"
    path.write_text(json.dumps(valid_payload("jsonl-1")) + "\n", encoding="utf-8")

    stream = JsonlFeedbackStream(path)
    result = consume_feedback_stream(stream, max_messages=1)

    assert result.source == str(path)
    assert result.records[0].feedback_id == "jsonl-1"


def test_jsonl_stream_rejects_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("not-json\n", encoding="utf-8")

    with pytest.raises(StreamIngestionError, match="Invalid JSON"):
        JsonlFeedbackStream(path)


def test_write_stream_records_csv_round_trip(tmp_path: Path) -> None:
    result = consume_feedback_stream(
        InMemoryFeedbackStream([valid_payload("csv-1")]), max_messages=1
    )
    output = write_stream_records_csv(result.records, tmp_path / "accepted.csv")

    text = output.read_text(encoding="utf-8")
    assert "feedback_id,customer_segment,channel,rating,text,created_at" in text
    assert "csv-1" in text


def test_kafka_stream_decodes_injected_consumer_and_commits() -> None:
    consumer = FakeKafkaConsumer([FakeKafkaMessage(valid_payload("kafka-1"))])
    stream = KafkaFeedbackStream(
        topic="feedback",
        bootstrap_servers="localhost:9092",
        group_id="tests",
        consumer=consumer,
    )

    result = consume_feedback_stream(stream, max_messages=1)

    assert result.source == "kafka:feedback"
    assert result.records[0].feedback_id == "kafka-1"
    assert consumer.committed == 1


def test_kinesis_stream_decodes_injected_client_and_tracks_sequence() -> None:
    client = FakeKinesisClient([valid_payload("kinesis-1")])
    stream = KinesisFeedbackStream(
        stream_name="feedback",
        shard_iterator="iterator-1",
        client=client,
    )

    result = consume_feedback_stream(stream, max_messages=1)

    assert result.source == "kinesis:feedback"
    assert result.records[0].feedback_id == "kinesis-1"
    assert stream.shard_iterator == "iterator-2"
    assert stream.acked_offsets == [StreamOffset(partition="pk-0", offset="seq-0")]


def test_stream_ingest_cli_writes_accepted_csv_and_dead_letters(tmp_path: Path) -> None:
    stream_path = tmp_path / "events.jsonl"
    output = tmp_path / "accepted.csv"
    dead = tmp_path / "dead.jsonl"
    stream_path.write_text(
        "\n".join(
            [
                json.dumps(valid_payload("cli-1")),
                json.dumps({**valid_payload("cli-bad"), "text": ""}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = runner.invoke(
        app,
        [
            "stream-ingest",
            "--input",
            str(stream_path),
            "--output",
            str(output),
            "--dead-letter",
            str(dead),
            "--max-messages",
            "10",
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"accepted_records": 1' in result.output
    assert '"rejected_messages": 1' in result.output
    assert "cli-1" in output.read_text(encoding="utf-8")
    assert dead.exists()


class FakeKafkaConsumer:
    def __init__(self, messages: list[FakeKafkaMessage]) -> None:
        self.messages = messages
        self.committed = 0

    def poll(self, timeout: float) -> FakeKafkaMessage | None:
        assert timeout > 0
        return self.messages.pop(0) if self.messages else None

    def commit(self, message: object, asynchronous: bool = False) -> object:
        assert isinstance(message, FakeKafkaMessage)
        assert asynchronous is False
        self.committed += 1
        return message


class FakeKafkaMessage:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def value(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def topic(self) -> str:
        return "feedback"

    def partition(self) -> int:
        return 0

    def offset(self) -> int:
        return 42

    def error(self) -> None:
        return None


class FakeKinesisClient:
    def __init__(self, payloads: list[dict[str, object]]) -> None:
        self.payloads = payloads

    def get_records(self, **kwargs: object) -> dict[str, object]:
        assert kwargs["ShardIterator"] == "iterator-1"
        assert kwargs["Limit"] == 1
        return {
            "NextShardIterator": "iterator-2",
            "Records": [
                {
                    "Data": json.dumps(payload).encode("utf-8"),
                    "PartitionKey": f"pk-{index}",
                    "SequenceNumber": f"seq-{index}",
                }
                for index, payload in enumerate(self.payloads)
            ],
        }
