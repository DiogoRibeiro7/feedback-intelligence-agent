from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from feedback_intelligence_agent.cli import app
from feedback_intelligence_agent.vector_store import InMemoryVectorStore

runner = CliRunner()
stdout_runner = CliRunner(mix_stderr=False)


def test_chat_command_single_message_then_followup(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    store_path = tmp_path / "conversations"
    common = ["--index-path", str(index_path), "--store-path", str(store_path)]

    first = stdout_runner.invoke(
        app,
        ["chat", "--message", "Why are enterprise customers unhappy with onboarding?", *common],
    )
    assert first.exit_code == 0, first.output
    first_payload = json.loads(first.stdout)
    conversation_id = first_payload["conversation_id"]
    assert conversation_id
    assert first_payload["result"]["citations"]

    second = stdout_runner.invoke(
        app,
        ["chat", "--message", "What about pricing?", "--conversation-id", conversation_id, *common],
    )
    assert second.exit_code == 0, second.output
    second_payload = json.loads(second.stdout)
    assert second_payload["conversation_id"] == conversation_id
    diagnostics = second_payload["result"]["diagnostics"]
    assert diagnostics["query_rewritten"] is True
    assert "onboarding" in diagnostics["retrieval_question"].lower()

    stored = json.loads((store_path / f"{conversation_id}.json").read_text(encoding="utf-8"))
    assert [turn["user_message"] for turn in stored["turns"]] == [
        "Why are enterprise customers unhappy with onboarding?",
        "What about pricing?",
    ]


def test_ingest_job_command_succeeds(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    store_path = tmp_path / "jobs"
    result = stdout_runner.invoke(
        app,
        [
            "ingest-job",
            "--input",
            "data/sample_feedback.csv",
            "--index-path",
            str(index_path),
            "--store-path",
            str(store_path),
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded"
    assert payload["chunks"] > 0
    assert payload["error"] is None
    assert index_path.exists()


def test_ingest_job_command_failure_exits_nonzero_with_clean_error(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    store_path = tmp_path / "jobs"
    result = stdout_runner.invoke(
        app,
        [
            "ingest-job",
            "--input",
            str(missing),
            "--index-path",
            str(tmp_path / "out.json"),
            "--store-path",
            str(store_path),
        ],
    )
    assert result.exit_code == 1, result.output
    payload = json.loads(result.stdout)
    assert payload["status"] == "failed"
    assert "missing.csv" not in payload["error"]
    assert "Ingestion failed" in payload["error"]


def test_update_index_command_merges_csv_batch(tmp_path: Path) -> None:
    batch = tmp_path / "batch.csv"
    index_path = tmp_path / "vector_store.json"
    batch.write_text(
        "\n".join(
            [
                "feedback_id,customer_segment,channel,rating,text,created_at",
                (
                    "inc-1,enterprise,support_ticket,2,"
                    "Incremental onboarding update,2026-08-01T09:00:00Z"
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = stdout_runner.invoke(
        app,
        [
            "update-index",
            "--input",
            str(batch),
            "--index-path",
            str(index_path),
            "--embedding-dim",
            "128",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["inserted_records"] == 1
    assert payload["total_chunks"] == 1
    assert InMemoryVectorStore.load(index_path).chunks[0].source_id == "inc-1"


def test_stream_ingest_command_can_update_index(tmp_path: Path) -> None:
    stream = tmp_path / "events.jsonl"
    index_path = tmp_path / "vector_store.json"
    output = tmp_path / "accepted.csv"
    stream.write_text(
        json.dumps(
            {
                "feedback_id": "stream-inc-1",
                "customer_segment": "enterprise",
                "channel": "support_ticket",
                "rating": 2,
                "text": "Streamed incremental index update",
                "created_at": "2026-08-01T09:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    result = stdout_runner.invoke(
        app,
        [
            "stream-ingest",
            "--input",
            str(stream),
            "--output",
            str(output),
            "--update-index",
            "--index-path",
            str(index_path),
            "--embedding-dim",
            "128",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["accepted_records"] == 1
    assert output.exists()
    assert InMemoryVectorStore.load(index_path).chunks[0].source_id == "stream-inc-1"


def test_export_lakehouse_command_writes_table_metadata(tmp_path: Path) -> None:
    output = tmp_path / "lakehouse" / "feedback"

    result = stdout_runner.invoke(
        app,
        [
            "export-lakehouse",
            "--input",
            "data/sample_feedback.csv",
            "--output",
            str(output),
            "--table-format",
            "iceberg",
            "--partition-column",
            "created_month",
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["table_format"] == "iceberg"
    assert payload["records"] == 12
    assert payload["partition_columns"] == ["created_month"]
    assert (output / "_lakehouse_manifest.json").exists()
    assert (output / "metadata" / "v1.metadata.json").exists()


def test_reports_commands_save_list_and_get(tmp_path: Path) -> None:
    store_path = tmp_path / "reports"
    index_path = tmp_path / "vector_store.json"

    saved = stdout_runner.invoke(
        app,
        [
            "reports",
            "save",
            "Why are enterprise customers unhappy with onboarding?",
            "--title",
            "Enterprise onboarding",
            "--tag",
            "Enterprise",
            "--tag",
            "onboarding",
            "--store-path",
            str(store_path),
            "--index-path",
            str(index_path),
        ],
    )

    assert saved.exit_code == 0, saved.output
    report = json.loads(saved.stdout)
    assert report["title"] == "Enterprise onboarding"
    assert report["tags"] == ["enterprise", "onboarding"]
    report_id = report["report_id"]

    listed = stdout_runner.invoke(app, ["reports", "list", "--store-path", str(store_path)])
    assert listed.exit_code == 0, listed.output
    summaries = json.loads(listed.stdout)
    assert summaries[0]["report_id"] == report_id

    fetched = stdout_runner.invoke(
        app,
        ["reports", "get", report_id, "--store-path", str(store_path)],
    )
    assert fetched.exit_code == 0, fetched.output
    assert json.loads(fetched.stdout)["report_id"] == report_id


def test_reports_get_missing_exits_nonzero(tmp_path: Path) -> None:
    result = stdout_runner.invoke(
        app,
        ["reports", "get", "missing", "--store-path", str(tmp_path / "reports")],
    )

    assert result.exit_code == 1
    assert "Report not found" in result.stderr


def test_reports_email_summary_command_renders_dry_run(tmp_path: Path) -> None:
    store_path = tmp_path / "reports"
    index_path = tmp_path / "vector_store.json"
    saved = stdout_runner.invoke(
        app,
        [
            "reports",
            "save",
            "Why are enterprise customers unhappy with onboarding?",
            "--title",
            "Enterprise onboarding",
            "--store-path",
            str(store_path),
            "--index-path",
            str(index_path),
        ],
    )
    assert saved.exit_code == 0, saved.output
    report_id = json.loads(saved.stdout)["report_id"]

    result = stdout_runner.invoke(
        app,
        [
            "reports",
            "email-summary",
            "--recipient",
            "pm@example.com",
            "--report-id",
            report_id,
            "--store-path",
            str(store_path),
        ],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["sent"] is False
    assert payload["summary"]["report_ids"] == [report_id]
    assert "Enterprise onboarding" in payload["summary"]["body_text"]


def test_reports_email_summary_requires_recipient(tmp_path: Path) -> None:
    result = stdout_runner.invoke(
        app,
        ["reports", "email-summary", "--store-path", str(tmp_path / "reports")],
    )

    assert result.exit_code == 2
    assert "At least one --recipient" in result.stderr


def test_answer_feedback_commands_submit_list_and_get(tmp_path: Path) -> None:
    store_path = tmp_path / "human_feedback"
    index_path = tmp_path / "vector_store.json"

    saved = stdout_runner.invoke(
        app,
        [
            "answer-feedback",
            "submit",
            "Why are enterprise customers unhappy with onboarding?",
            "--rating",
            "useful",
            "--comment",
            "Grounded and actionable.",
            "--tag",
            "Enterprise",
            "--tag",
            "onboarding",
            "--store-path",
            str(store_path),
            "--index-path",
            str(index_path),
        ],
    )

    assert saved.exit_code == 0, saved.output
    record = json.loads(saved.stdout)
    assert record["rating"] == "useful"
    assert record["comment"] == "Grounded and actionable."
    assert record["tags"] == ["enterprise", "onboarding"]
    feedback_id = record["feedback_id"]

    listed = stdout_runner.invoke(
        app,
        ["answer-feedback", "list", "--store-path", str(store_path)],
    )
    assert listed.exit_code == 0, listed.output
    summaries = json.loads(listed.stdout)
    assert summaries[0]["feedback_id"] == feedback_id

    fetched = stdout_runner.invoke(
        app,
        ["answer-feedback", "get", feedback_id, "--store-path", str(store_path)],
    )
    assert fetched.exit_code == 0, fetched.output
    assert json.loads(fetched.stdout)["feedback_id"] == feedback_id


def test_answer_feedback_get_missing_exits_nonzero(tmp_path: Path) -> None:
    result = stdout_runner.invoke(
        app,
        ["answer-feedback", "get", "missing", "--store-path", str(tmp_path / "feedback")],
    )

    assert result.exit_code == 1
    assert "Answer feedback not found" in result.stderr


def test_query_command_applies_metadata_filters(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    result = stdout_runner.invoke(
        app,
        [
            "query",
            "Which feedback mentions support?",
            "--index-path",
            str(index_path),
            "--channel",
            "support_ticket",
            "--max-rating",
            "2",
            "--created-after",
            "2026-02-01T00:00:00",
        ],
    )
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    citations = payload["citations"]
    assert citations
    assert {citation["document_id"] for citation in citations} == {"fb-005", "fb-009"}
    assert payload["diagnostics"]["metadata_filters"]["channel"] == "support_ticket"


def test_chat_command_interactive_repl(tmp_path: Path) -> None:
    index_path = tmp_path / "vector_store.json"
    store_path = tmp_path / "conversations"
    result = runner.invoke(
        app,
        ["chat", "--index-path", str(index_path), "--store-path", str(store_path)],
        input="Why is onboarding slow?\nexit\n",
    )
    assert result.exit_code == 0, result.output
    assert "The strongest signal" in result.output
    conversations = list(store_path.glob("*.json"))
    assert len(conversations) == 1


def test_evaluate_command_writes_structured_report(tmp_path: Path) -> None:
    output = tmp_path / "evaluation_report.json"
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            "examples/queries.jsonl",
            "--output",
            str(output),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["total_cases"] == 5
    assert report["top_k"] == 4
    for metric in ("precision_at_k", "recall_at_k", "mean_reciprocal_rank", "context_hit_rate"):
        assert 0.0 <= report["retrieval"][metric] <= 1.0
    for metric in ("keyword_coverage", "groundedness", "refusal_correctness"):
        assert 0.0 <= report["answers"][metric] <= 1.0
    assert len(report["cases"]) == 5
    # The stdout report matches the file, so the command is scriptable.
    assert '"total_cases": 5' in result.output


def test_evaluate_command_creates_output_directory(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "dir" / "report.json"
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            "examples/queries.jsonl",
            "--output",
            str(output),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert output.exists()


def test_prompts_list_shows_registered_prompts() -> None:
    result = runner.invoke(app, ["prompts", "list"])
    assert result.exit_code == 0, result.output
    assert "rag_answer v1 (latest)" in result.output
    assert "rag_system v1 (latest)" in result.output
    assert "required variables: question" in result.output
    assert "changelog:" in result.output


def test_prompts_render_with_question_only_uses_defaults() -> None:
    result = runner.invoke(
        app,
        [
            "prompts",
            "render",
            "--name",
            "rag_answer",
            "--version",
            "latest",
            "--var",
            "question=Why is onboarding slow?",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Why is onboarding slow?" in result.output
    assert "Route: general_insight" in result.output


def test_prompts_render_supports_repeated_vars() -> None:
    result = runner.invoke(
        app,
        [
            "prompts",
            "render",
            "--name",
            "rag_answer",
            "--var",
            "question=Why is onboarding slow?",
            "--var",
            "route=onboarding",
            "--var",
            "context=text: setup took weeks",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Route: onboarding" in result.output
    assert "text: setup took weeks" in result.output


def test_prompts_render_missing_required_variable_fails_clearly() -> None:
    result = runner.invoke(app, ["prompts", "render", "--name", "rag_answer"])
    assert result.exit_code == 1
    assert "missing required variable" in result.output
    assert "question" in result.output


def test_prompts_render_unknown_prompt_fails_clearly() -> None:
    result = runner.invoke(app, ["prompts", "render", "--name", "nope"])
    assert result.exit_code == 1
    assert "unknown prompt 'nope'" in result.output


def test_prompts_render_rejects_malformed_var() -> None:
    result = runner.invoke(app, ["prompts", "render", "--name", "rag_answer", "--var", "question"])
    assert result.exit_code == 2
    assert "expected key=value" in result.output


def test_evaluate_command_fails_on_invalid_queries_file(tmp_path: Path) -> None:
    queries = tmp_path / "bad.jsonl"
    queries.write_text("not json\n", encoding="utf-8")
    index_path = tmp_path / "vector_store.json"
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--queries",
            str(queries),
            "--output",
            str(tmp_path / "report.json"),
            "--index-path",
            str(index_path),
        ],
    )
    assert result.exit_code != 0
