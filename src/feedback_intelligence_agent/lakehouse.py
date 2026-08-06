"""Local lakehouse-style feedback exports.

The exporter intentionally avoids heavyweight table runtimes in the default
install. It writes deterministic partitioned JSONL data files plus Delta-style
or Iceberg-style metadata so local demos and CI can exercise the data-product
contract without requiring Spark, delta-rs, PyArrow, or an object store.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field

from feedback_intelligence_agent.data_contracts import REQUIRED_COLUMNS
from feedback_intelligence_agent.privacy import redact_feedback_record
from feedback_intelligence_agent.schemas import FeedbackRecord
from feedback_intelligence_agent.telemetry import Telemetry

DEFAULT_PARTITION_COLUMNS: tuple[str, ...] = ("created_date", "channel")
ALLOWED_PARTITION_COLUMNS: tuple[str, ...] = (
    "tenant_id",
    "created_date",
    "created_month",
    "channel",
    "customer_segment",
    "rating",
)


class LakehouseTableFormat(str, Enum):
    """Metadata layout to write around exported feedback files."""

    delta = "delta"
    iceberg = "iceberg"


class LakehouseDataFormat(str, Enum):
    """Data file encoding used by the dependency-light exporter."""

    jsonl = "jsonl"


class LakehouseDataFile(BaseModel):
    """One data file produced by a lakehouse export."""

    path: str
    records: int = Field(ge=0)
    partition_values: dict[str, str]
    min_created_at: datetime | None = None
    max_created_at: datetime | None = None


class LakehouseExportResult(BaseModel):
    """Summary of a local lakehouse export."""

    table_path: str
    table_format: LakehouseTableFormat
    data_format: LakehouseDataFormat
    partition_columns: list[str]
    records: int = Field(ge=0)
    partitions: int = Field(ge=0)
    data_files: list[LakehouseDataFile]
    manifest_path: str
    metadata_path: str


def export_feedback_lakehouse(
    records: Iterable[FeedbackRecord],
    output_dir: str | Path,
    *,
    table_format: LakehouseTableFormat = LakehouseTableFormat.delta,
    data_format: LakehouseDataFormat = LakehouseDataFormat.jsonl,
    partition_columns: Sequence[str] = DEFAULT_PARTITION_COLUMNS,
    telemetry: Telemetry | None = None,
) -> LakehouseExportResult:
    """Export feedback records to a partitioned local lakehouse-style table."""
    if data_format is not LakehouseDataFormat.jsonl:
        raise ValueError("only jsonl data files are supported by the local exporter")

    partitions = _normalise_partition_columns(partition_columns)
    table_path = Path(output_dir)
    telemetry = telemetry or Telemetry()
    redacted_records = [redact_feedback_record(record) for record in records]

    correlation_id = telemetry.new_correlation_id()
    with telemetry.span(
        "lakehouse_export_started",
        "lakehouse_export_finished",
        correlation_id=correlation_id,
        metadata={
            "table_path": str(table_path),
            "table_format": table_format.value,
            "records": len(redacted_records),
            "partition_columns": list(partitions),
        },
    ) as span:
        grouped = _group_records(redacted_records, partitions)
        table_path.mkdir(parents=True, exist_ok=True)
        data_files = _write_data_files(
            grouped,
            table_path=table_path,
            partition_columns=partitions,
        )
        manifest_path = _write_manifest(
            table_path=table_path,
            table_format=table_format,
            data_format=data_format,
            partition_columns=partitions,
            records=len(redacted_records),
            data_files=data_files,
        )
        metadata_path = _write_table_metadata(
            table_path=table_path,
            table_format=table_format,
            data_format=data_format,
            partition_columns=partitions,
            data_files=data_files,
        )
        result = LakehouseExportResult(
            table_path=str(table_path),
            table_format=table_format,
            data_format=data_format,
            partition_columns=list(partitions),
            records=len(redacted_records),
            partitions=len(grouped),
            data_files=data_files,
            manifest_path=str(manifest_path),
            metadata_path=str(metadata_path),
        )
        span["partitions"] = result.partitions
        span["data_files"] = len(result.data_files)
        span["manifest_path"] = result.manifest_path
        span["metadata_path"] = result.metadata_path
    return result


def _normalise_partition_columns(partition_columns: Sequence[str]) -> tuple[str, ...]:
    if not partition_columns:
        raise ValueError("partition_columns must contain at least one column")
    normalised: list[str] = []
    for column in partition_columns:
        if column not in ALLOWED_PARTITION_COLUMNS:
            allowed = ", ".join(ALLOWED_PARTITION_COLUMNS)
            raise ValueError(f"unknown partition column {column!r}; allowed columns: {allowed}")
        if column not in normalised:
            normalised.append(column)
    return tuple(normalised)


def _group_records(
    records: list[FeedbackRecord],
    partition_columns: tuple[str, ...],
) -> dict[tuple[str, ...], list[FeedbackRecord]]:
    grouped: dict[tuple[str, ...], list[FeedbackRecord]] = {}
    for record in sorted(records, key=lambda item: (item.tenant_id, item.feedback_id)):
        key = tuple(_partition_value(record, column) for column in partition_columns)
        grouped.setdefault(key, []).append(record)
    return dict(sorted(grouped.items(), key=lambda item: item[0]))


def _write_data_files(
    grouped: dict[tuple[str, ...], list[FeedbackRecord]],
    *,
    table_path: Path,
    partition_columns: tuple[str, ...],
) -> list[LakehouseDataFile]:
    data_files: list[LakehouseDataFile] = []
    for index, (key, records) in enumerate(grouped.items()):
        partition_values = dict(zip(partition_columns, key, strict=True))
        partition_path = table_path.joinpath(
            *[
                f"{column}={_safe_partition_value(value)}"
                for column, value in partition_values.items()
            ]
        )
        partition_path.mkdir(parents=True, exist_ok=True)
        data_path = partition_path / f"part-{index:05d}.jsonl"
        lines = [_record_json_line(record) for record in records]
        data_path.write_text("".join(lines), encoding="utf-8")
        created_values = [record.created_at for record in records]
        data_files.append(
            LakehouseDataFile(
                path=str(data_path),
                records=len(records),
                partition_values=partition_values,
                min_created_at=min(created_values) if created_values else None,
                max_created_at=max(created_values) if created_values else None,
            )
        )
    return data_files


def _write_manifest(
    *,
    table_path: Path,
    table_format: LakehouseTableFormat,
    data_format: LakehouseDataFormat,
    partition_columns: tuple[str, ...],
    records: int,
    data_files: list[LakehouseDataFile],
) -> Path:
    manifest_path = table_path / "_lakehouse_manifest.json"
    payload = {
        "table_format": table_format.value,
        "data_format": data_format.value,
        "schema": _schema(),
        "partition_columns": list(partition_columns),
        "records": records,
        "data_files": [data_file.model_dump(mode="json") for data_file in data_files],
        "generated_at": _stable_generated_at(),
    }
    manifest_path.write_text(_json(payload, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def _write_table_metadata(
    *,
    table_path: Path,
    table_format: LakehouseTableFormat,
    data_format: LakehouseDataFormat,
    partition_columns: tuple[str, ...],
    data_files: list[LakehouseDataFile],
) -> Path:
    if table_format is LakehouseTableFormat.delta:
        return _write_delta_log(
            table_path=table_path,
            data_format=data_format,
            partition_columns=partition_columns,
            data_files=data_files,
        )
    return _write_iceberg_metadata(
        table_path=table_path,
        data_format=data_format,
        partition_columns=partition_columns,
        data_files=data_files,
    )


def _write_delta_log(
    *,
    table_path: Path,
    data_format: LakehouseDataFormat,
    partition_columns: tuple[str, ...],
    data_files: list[LakehouseDataFile],
) -> Path:
    log_dir = table_path / "_delta_log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "00000000000000000000.json"
    actions: list[dict[str, object]] = [
        {"protocol": {"minReaderVersion": 1, "minWriterVersion": 2}},
        {
            "metaData": {
                "id": "feedback-intelligence-agent-local",
                "format": {"provider": data_format.value, "options": {}},
                "schemaString": _json({"type": "struct", "fields": _schema()}),
                "partitionColumns": list(partition_columns),
                "configuration": {"delta.appendOnly": "true"},
            }
        },
    ]
    for data_file in data_files:
        actions.append(
            {
                "add": {
                    "path": _relative_posix(table_path, Path(data_file.path)),
                    "partitionValues": data_file.partition_values,
                    "size": Path(data_file.path).stat().st_size,
                    "dataChange": True,
                }
            }
        )
    log_path.write_text("".join(_json(action) + "\n" for action in actions), encoding="utf-8")
    return log_path


def _write_iceberg_metadata(
    *,
    table_path: Path,
    data_format: LakehouseDataFormat,
    partition_columns: tuple[str, ...],
    data_files: list[LakehouseDataFile],
) -> Path:
    metadata_dir = table_path / "metadata"
    manifest_dir = metadata_dir / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "manifest-00000.json"
    manifest_payload = {
        "data_files": [
            {
                **data_file.model_dump(mode="json"),
                "path": _relative_posix(table_path, Path(data_file.path)),
            }
            for data_file in data_files
        ]
    }
    manifest_path.write_text(_json(manifest_payload, indent=2) + "\n", encoding="utf-8")

    metadata_path = metadata_dir / "v1.metadata.json"
    payload = {
        "format-version": 2,
        "table-uuid": "feedback-intelligence-agent-local",
        "location": str(table_path),
        "last-sequence-number": 1,
        "last-updated-ms": 0,
        "schemas": [{"schema-id": 0, "fields": _schema()}],
        "current-schema-id": 0,
        "partition-specs": [
            {
                "spec-id": 0,
                "fields": [
                    {
                        "name": column,
                        "transform": "identity",
                        "source-name": column,
                    }
                    for column in partition_columns
                ],
            }
        ],
        "default-spec-id": 0,
        "properties": {"write.format.default": data_format.value},
        "snapshots": [
            {
                "snapshot-id": 1,
                "sequence-number": 1,
                "manifest-list": _relative_posix(table_path, manifest_path),
            }
        ],
        "current-snapshot-id": 1,
    }
    metadata_path.write_text(_json(payload, indent=2) + "\n", encoding="utf-8")
    return metadata_path


def _record_json_line(record: FeedbackRecord) -> str:
    row = record.model_dump(mode="json")
    payload = {column: row[column] for column in REQUIRED_COLUMNS}
    payload["tenant_id"] = row["tenant_id"]
    payload["created_date"] = _partition_value(record, "created_date")
    payload["created_month"] = _partition_value(record, "created_month")
    return _json(payload) + "\n"


def _schema() -> list[dict[str, str]]:
    return [
        {"name": "tenant_id", "type": "string"},
        {"name": "feedback_id", "type": "string"},
        {"name": "customer_segment", "type": "string"},
        {"name": "channel", "type": "string"},
        {"name": "rating", "type": "integer"},
        {"name": "text", "type": "string"},
        {"name": "created_at", "type": "timestamp"},
        {"name": "created_date", "type": "date"},
        {"name": "created_month", "type": "string"},
    ]


def _partition_value(record: FeedbackRecord, column: str) -> str:
    if column == "tenant_id":
        return record.tenant_id
    if column == "created_date":
        return record.created_at.date().isoformat()
    if column == "created_month":
        return record.created_at.date().isoformat()[:7]
    if column == "channel":
        return record.channel.value
    if column == "customer_segment":
        return record.customer_segment
    if column == "rating":
        return str(record.rating)
    raise ValueError(f"unknown partition column {column!r}")


def _safe_partition_value(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.=-]+", "_", value.strip())
    return safe or "unknown"


def _relative_posix(root: Path, path: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return path.as_posix()
    return relative.as_posix()


def _stable_generated_at() -> str:
    return datetime.fromtimestamp(0, tz=timezone.utc).isoformat()


def _json(payload: object, *, indent: int | None = None) -> str:
    return json.dumps(payload, indent=indent, sort_keys=True, separators=(",", ": "))
