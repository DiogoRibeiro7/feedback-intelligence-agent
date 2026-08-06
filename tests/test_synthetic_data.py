from __future__ import annotations

from collections import Counter
from pathlib import Path

import pytest
from typer.testing import CliRunner

from feedback_intelligence_agent.cli import app
from feedback_intelligence_agent.data_contracts import validate_feedback_csv
from feedback_intelligence_agent.ingestion import load_feedback_csv
from feedback_intelligence_agent.synthetic_data import (
    SENTIMENTS,
    SyntheticDataConfig,
    generate_feedback_rows,
    write_feedback_csv,
)

runner = CliRunner()


def test_generated_rows_pass_data_contract(tmp_path: Path) -> None:
    out = tmp_path / "synthetic.csv"
    write_feedback_csv(SyntheticDataConfig(rows=50, seed=42), out)
    report, records = validate_feedback_csv(out, strict=True)
    assert report.is_valid
    assert report.total_rows == 50
    assert report.valid_rows == 50
    assert len(records) == 50
    # The optional sentiment column is recognised by the contract, not flagged.
    assert not any(issue.column == "sentiment" for issue in report.warnings)


def test_generated_data_works_with_ingestion(tmp_path: Path) -> None:
    out = tmp_path / "synthetic.csv"
    write_feedback_csv(SyntheticDataConfig(rows=25, seed=7), out)
    records = load_feedback_csv(out)
    assert len(records) == 25
    assert all(1 <= record.rating <= 5 for record in records)


def test_row_count_and_rating_range() -> None:
    rows = generate_feedback_rows(SyntheticDataConfig(rows=200, seed=1))
    assert len(rows) == 200
    assert all(1 <= int(row["rating"]) <= 5 for row in rows)
    assert len({row["feedback_id"] for row in rows}) == 200


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    write_feedback_csv(SyntheticDataConfig(rows=100, seed=123), first)
    write_feedback_csv(SyntheticDataConfig(rows=100, seed=123), second)
    assert first.read_bytes() == second.read_bytes()


def test_different_seed_changes_output(tmp_path: Path) -> None:
    first = tmp_path / "a.csv"
    second = tmp_path / "b.csv"
    write_feedback_csv(SyntheticDataConfig(rows=100, seed=1), first)
    write_feedback_csv(SyntheticDataConfig(rows=100, seed=2), second)
    assert first.read_bytes() != second.read_bytes()


def test_sentiment_distribution_approximately_honored() -> None:
    distribution = {"positive": 0.6, "neutral": 0.1, "negative": 0.3}
    rows = generate_feedback_rows(
        SyntheticDataConfig(rows=4000, seed=99, sentiment_distribution=distribution)
    )
    counts = Counter(row["sentiment"] for row in rows)
    total = sum(counts.values())
    for sentiment, target in distribution.items():
        observed = counts[sentiment] / total
        assert abs(observed - target) < 0.05, (sentiment, observed, target)


def test_rating_aligns_with_sentiment() -> None:
    rows = generate_feedback_rows(SyntheticDataConfig(rows=300, seed=5))
    for row in rows:
        rating = int(row["rating"])
        if row["sentiment"] == "positive":
            assert rating >= 4
        elif row["sentiment"] == "negative":
            assert rating <= 2
        else:
            assert rating == 3


def test_distribution_normalisation_allows_unnormalised_weights() -> None:
    config = SyntheticDataConfig(
        rows=10, seed=1, sentiment_distribution={"positive": 2, "negative": 2}
    )
    assert pytest.approx(sum(config.sentiment_distribution.values())) == 1.0
    assert config.sentiment_distribution["neutral"] == 0.0


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="rows must be"):
        SyntheticDataConfig(rows=0)
    with pytest.raises(ValueError, match="unknown sentiment"):
        SyntheticDataConfig(sentiment_distribution={"elated": 1.0})
    with pytest.raises(ValueError, match="positive value"):
        SyntheticDataConfig(sentiment_distribution={"positive": 0.0})


def test_optional_sentiment_column_can_be_omitted(tmp_path: Path) -> None:
    out = tmp_path / "no_sentiment.csv"
    write_feedback_csv(SyntheticDataConfig(rows=10, seed=3, include_sentiment_column=False), out)
    header = out.read_text(encoding="utf-8").splitlines()[0]
    assert "sentiment" not in header
    report, _ = validate_feedback_csv(out, strict=True)
    assert report.is_valid


def test_optional_tenant_column_can_be_emitted(tmp_path: Path) -> None:
    out = tmp_path / "tenants.csv"
    write_feedback_csv(
        SyntheticDataConfig(
            rows=20,
            seed=4,
            tenant_ids=("acme", "cobalt"),
            include_tenant_column=True,
        ),
        out,
    )

    header = out.read_text(encoding="utf-8").splitlines()[0]
    report, records = validate_feedback_csv(out, strict=True)

    assert header.startswith("tenant_id,feedback_id")
    assert report.is_valid
    assert {record.tenant_id for record in records} <= {"acme", "cobalt"}


def test_all_sentiments_appear_with_uniform_distribution() -> None:
    uniform = dict.fromkeys(SENTIMENTS, 1.0)
    rows = generate_feedback_rows(
        SyntheticDataConfig(rows=500, seed=11, sentiment_distribution=uniform)
    )
    seen = {row["sentiment"] for row in rows}
    assert seen == set(SENTIMENTS)


def test_cli_generate_data_produces_valid_csv(tmp_path: Path) -> None:
    out = tmp_path / "synthetic.csv"
    result = runner.invoke(
        app,
        ["generate-data", "--rows", "50", "--output", str(out), "--seed", "42"],
    )
    assert result.exit_code == 0, result.output
    assert out.exists()
    validate = runner.invoke(app, ["validate-data", str(out), "--strict"])
    assert validate.exit_code == 0, validate.output
    assert '"valid_rows": 50' in validate.stdout
