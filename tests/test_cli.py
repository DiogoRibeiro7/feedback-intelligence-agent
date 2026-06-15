from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from ai_engineering_showcase.cli import app

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
