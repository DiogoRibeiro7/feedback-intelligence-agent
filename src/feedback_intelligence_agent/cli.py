"""Command-line interface for the feedback intelligence agent."""

from __future__ import annotations

import json
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from feedback_intelligence_agent.benchmarking import run_benchmark, write_benchmark_outputs
from feedback_intelligence_agent.citations import render_citations
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.data_contracts import DataContractError, validate_feedback_csv
from feedback_intelligence_agent.email_summaries import (
    EmailSummaryDelivery,
    EmailSummaryRequest,
    deliver_email_summary,
    render_email_summary,
    select_reports_for_summary,
)
from feedback_intelligence_agent.evaluation import evaluate_system, load_evaluation_cases
from feedback_intelligence_agent.experiments import (
    ExperimentConfig,
    collect_run_metadata,
    run_experiment,
    track_experiment_run,
    write_experiment_outputs,
)
from feedback_intelligence_agent.factory import (
    build_agent,
    build_conversation_store,
    build_index,
    build_job_store,
    build_retriever,
    build_telemetry,
    load_or_build_index,
)
from feedback_intelligence_agent.human_feedback import (
    HumanFeedbackRating,
    JsonHumanFeedbackStore,
    SubmitHumanFeedbackRequest,
)
from feedback_intelligence_agent.index_updates import update_json_index
from feedback_intelligence_agent.ingestion import load_feedback_csv
from feedback_intelligence_agent.jobs import JobRequest, run_ingestion_job
from feedback_intelligence_agent.lakehouse import (
    DEFAULT_PARTITION_COLUMNS,
    LakehouseTableFormat,
    export_feedback_lakehouse,
)
from feedback_intelligence_agent.prompt_registry import (
    LATEST_VERSION,
    PromptNotFoundError,
    PromptVariableError,
)
from feedback_intelligence_agent.prompts import PROMPT_REGISTRY
from feedback_intelligence_agent.reports import JsonInsightReportStore, SaveInsightReportRequest
from feedback_intelligence_agent.schemas import ChatResponse, FeedbackChannel, MetadataFilters
from feedback_intelligence_agent.streaming_ingestion import (
    JsonlFeedbackStream,
    consume_feedback_stream,
    write_stream_records_csv,
)
from feedback_intelligence_agent.synthetic_data import SyntheticDataConfig, write_feedback_csv
from feedback_intelligence_agent.telemetry import configure_logging

app = typer.Typer(help="Feedback Intelligence Agent CLI")
experiment_app = typer.Typer(help="Run repeatable experiments over RAG configurations.")
app.add_typer(experiment_app, name="experiment")
prompts_app = typer.Typer(help="Inspect and render versioned prompt templates.")
app.add_typer(prompts_app, name="prompts")
reports_app = typer.Typer(help="Create and inspect saved insight reports.")
app.add_typer(reports_app, name="reports")
answer_feedback_app = typer.Typer(help="Capture human feedback on generated answers.")
app.add_typer(answer_feedback_app, name="answer-feedback")


class RetrieverChoice(str, Enum):
    """Retriever strategies selectable from the command line."""

    dense = "dense"
    lexical = "lexical"
    hybrid = "hybrid"


def _metadata_filters(
    *,
    customer_segment: str | None,
    channel: FeedbackChannel | None,
    min_rating: int | None,
    max_rating: int | None,
    created_after: datetime | None,
    created_before: datetime | None,
) -> MetadataFilters | None:
    """Build metadata filters from optional CLI parameters."""
    filters = MetadataFilters(
        customer_segment=customer_segment,
        channel=channel,
        min_rating=min_rating,
        max_rating=max_rating,
        created_after=created_after,
        created_before=created_before,
    )
    return None if filters.is_empty else filters


def _default_report_title(question: str) -> str:
    """Create a compact report title from a question."""
    title = " ".join(question.strip().rstrip("?!.").split())
    return title[:120] or "Saved insight report"


@app.command()
def index(
    input: Annotated[Path, typer.Option(help="Path to feedback CSV.")] = Path(
        "data/sample_feedback.csv"
    ),
    index_path: Annotated[Path, typer.Option(help="Output path for vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    embedding_dim: Annotated[int, typer.Option(help="Hashing embedding dimension.")] = 512,
) -> None:
    """Build a local vector index."""
    configure_logging()
    telemetry = build_telemetry(Settings(index_path=index_path))
    vector_store = build_index(input, index_path, embedding_dim=embedding_dim, telemetry=telemetry)
    typer.echo(f"Indexed {vector_store.size} chunks into {index_path}")


@app.command("ingest-job")
def ingest_job(
    input: Annotated[Path, typer.Option(help="Path to feedback CSV to ingest.")] = Path(
        "data/sample_feedback.csv"
    ),
    index_path: Annotated[Path, typer.Option(help="Output path for vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    embedding_dim: Annotated[int, typer.Option(help="Hashing embedding dimension.")] = 512,
    store_path: Annotated[
        Path, typer.Option(help="Directory holding ingestion job JSON files.")
    ] = Path(".artifacts/jobs"),
) -> None:
    """Run an ingestion job locally and print its JobResult.

    Uses the same job pipeline the API schedules in the background, but runs it
    synchronously so the terminal status (succeeded/failed) and the resulting
    JobResult are printed immediately. Exits non-zero on a failed job.
    """
    configure_logging()
    settings = Settings(
        index_path=index_path, embedding_dim=embedding_dim, job_store_path=store_path
    )
    store = build_job_store(settings)
    job = store.create(JobRequest(input_path=str(input), index_path=str(index_path)))
    result = run_ingestion_job(
        job.job_id,
        store,
        embedding_dim=settings.embedding_dim,
        default_index_path=str(settings.index_path),
        telemetry=build_telemetry(settings),
    )
    typer.echo(result.model_dump_json(indent=2))
    if result.status.value == "failed":
        raise typer.Exit(code=1)


@app.command("stream-ingest")
def stream_ingest(
    input: Annotated[Path, typer.Option(help="JSONL stream file of feedback events.")],
    output: Annotated[Path, typer.Option(help="CSV path for accepted feedback records.")] = Path(
        ".artifacts/stream_feedback.csv"
    ),
    dead_letter: Annotated[
        Path | None, typer.Option(help="Optional JSONL path for rejected stream messages.")
    ] = Path(".artifacts/stream_dead_letters.jsonl"),
    max_messages: Annotated[
        int, typer.Option(help="Maximum stream messages to consume in this batch.")
    ] = 100,
    update_index: Annotated[
        bool, typer.Option(help="Merge accepted stream records into a JSON vector index.")
    ] = False,
    index_path: Annotated[Path, typer.Option(help="JSON vector index path to update.")] = Path(
        ".artifacts/vector_store.json"
    ),
    embedding_dim: Annotated[int, typer.Option(help="Hashing embedding dimension.")] = 512,
) -> None:
    """Consume a bounded local stream batch and write validated feedback records.

    This command uses the same streaming validation path as Kafka/Kinesis adapters,
    but reads from a JSONL file so local demos and CI remain deterministic.
    """
    configure_logging()
    settings = Settings()
    stream = JsonlFeedbackStream(input)
    result = consume_feedback_stream(
        stream,
        max_messages=max_messages,
        dead_letter_path=dead_letter,
        telemetry=build_telemetry(settings),
    )
    if result.records:
        write_stream_records_csv(result.records, output)
    if update_index and result.records:
        index_result = update_json_index(
            result.records,
            index_path,
            embedding_dim=embedding_dim,
            telemetry=build_telemetry(settings),
        )
        typer.echo(f"index update: {index_result.model_dump_json()}", err=True)
    typer.echo(result.model_dump_json(indent=2))
    if result.rejected_messages:
        typer.echo(f"{result.rejected_messages} message(s) written to {dead_letter}", err=True)


@app.command("update-index")
def update_index(
    input: Annotated[Path, typer.Option(help="Feedback CSV batch to merge into the index.")],
    index_path: Annotated[Path, typer.Option(help="JSON vector index path to update.")] = Path(
        ".artifacts/vector_store.json"
    ),
    embedding_dim: Annotated[int, typer.Option(help="Hashing embedding dimension.")] = 512,
    chunk_size: Annotated[int, typer.Option(help="Maximum words per chunk.")] = 80,
    chunk_overlap: Annotated[int, typer.Option(help="Words shared between adjacent chunks.")] = 16,
) -> None:
    """Incrementally merge a validated feedback CSV batch into a JSON index."""
    configure_logging()
    settings = Settings(index_path=index_path, embedding_dim=embedding_dim)
    records = load_feedback_csv(input, telemetry=build_telemetry(settings))
    result = update_json_index(
        records,
        index_path,
        embedding_dim=embedding_dim,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        telemetry=build_telemetry(settings),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("export-lakehouse")
def export_lakehouse(
    input: Annotated[Path, typer.Option(help="Feedback CSV to export.")],
    output: Annotated[Path, typer.Option(help="Output table directory.")] = Path(
        ".artifacts/lakehouse/feedback"
    ),
    table_format: Annotated[
        LakehouseTableFormat,
        typer.Option(help="Lakehouse metadata layout to write."),
    ] = LakehouseTableFormat.delta,
    partition_column: Annotated[
        list[str] | None,
        typer.Option(
            "--partition-column",
            help="Partition column. Repeat to override the default created_date/channel layout.",
        ),
    ] = None,
) -> None:
    """Export validated feedback to a local lakehouse-style table."""
    configure_logging()
    settings = Settings()
    records = load_feedback_csv(input, telemetry=build_telemetry(settings))
    result = export_feedback_lakehouse(
        records,
        output,
        table_format=table_format,
        partition_columns=partition_column or DEFAULT_PARTITION_COLUMNS,
        telemetry=build_telemetry(settings),
    )
    typer.echo(result.model_dump_json(indent=2))


@app.command("validate-data")
def validate_data(
    input: Annotated[Path, typer.Argument(help="Path to feedback CSV to validate.")],
    strict: Annotated[
        bool, typer.Option(help="Fail with a non-zero exit code on any validation error.")
    ] = False,
) -> None:
    """Validate a feedback CSV against the data contract and print a report."""
    configure_logging()
    try:
        report, _ = validate_feedback_csv(input, strict=strict)
    except DataContractError as exc:
        typer.echo(exc.report.model_dump_json(indent=2))
        typer.echo(exc.report.summary(), err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(report.summary(), err=True)


@app.command("generate-data")
def generate_data(
    rows: Annotated[int, typer.Option(help="Number of feedback records to generate.")] = 1000,
    output: Annotated[Path, typer.Option(help="Destination CSV path.")] = Path(
        "data/synthetic_feedback.csv"
    ),
    seed: Annotated[
        int, typer.Option(help="Random seed; same seed and options give identical output.")
    ] = 42,
) -> None:
    """Generate a synthetic feedback CSV compatible with the data contract."""
    configure_logging()
    config = SyntheticDataConfig(rows=rows, seed=seed)
    written = write_feedback_csv(config, output)
    typer.echo(f"Wrote {rows} synthetic feedback rows to {written}")


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask over feedback data.")],
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
    retriever: Annotated[
        RetrieverChoice, typer.Option(help="Retrieval strategy: dense, lexical, or hybrid.")
    ] = RetrieverChoice.dense,
    dense_weight: Annotated[
        float, typer.Option(help="Dense score weight for hybrid retrieval.")
    ] = 0.6,
    lexical_weight: Annotated[
        float, typer.Option(help="Lexical score weight for hybrid retrieval.")
    ] = 0.4,
    customer_segment: Annotated[
        str | None, typer.Option(help="Filter retrieved feedback by customer segment.")
    ] = None,
    channel: Annotated[
        FeedbackChannel | None, typer.Option(help="Filter retrieved feedback by channel.")
    ] = None,
    min_rating: Annotated[
        int | None, typer.Option(help="Minimum feedback rating to retrieve.")
    ] = None,
    max_rating: Annotated[
        int | None, typer.Option(help="Maximum feedback rating to retrieve.")
    ] = None,
    created_after: Annotated[
        datetime | None, typer.Option(help="Only retrieve feedback created at or after this time.")
    ] = None,
    created_before: Annotated[
        datetime | None, typer.Option(help="Only retrieve feedback created at or before this time.")
    ] = None,
) -> None:
    """Ask a question against the indexed feedback."""
    configure_logging()
    settings = Settings(
        index_path=index_path,
        retriever_type=retriever.value,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )
    agent = build_agent(settings)
    filters = _metadata_filters(
        customer_segment=customer_segment,
        channel=channel,
        min_rating=min_rating,
        max_rating=max_rating,
        created_after=created_after,
        created_before=created_before,
    )
    answer = agent.answer(question, top_k=top_k, filters=filters)
    typer.echo(answer.model_dump_json(indent=2))
    typer.echo(render_citations(answer.citations), err=True)


@app.command()
def chat(
    message: Annotated[
        str | None,
        typer.Option(help="Single message for non-interactive mode; omit to start a REPL."),
    ] = None,
    conversation_id: Annotated[
        str | None,
        typer.Option(help="Conversation to continue; omit to start a new one."),
    ] = None,
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    store_path: Annotated[
        Path, typer.Option(help="Directory holding conversation JSON files.")
    ] = Path(".artifacts/conversations"),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
    customer_segment: Annotated[
        str | None, typer.Option(help="Filter retrieved feedback by customer segment.")
    ] = None,
    channel: Annotated[
        FeedbackChannel | None, typer.Option(help="Filter retrieved feedback by channel.")
    ] = None,
    min_rating: Annotated[
        int | None, typer.Option(help="Minimum feedback rating to retrieve.")
    ] = None,
    max_rating: Annotated[
        int | None, typer.Option(help="Maximum feedback rating to retrieve.")
    ] = None,
    created_after: Annotated[
        datetime | None, typer.Option(help="Only retrieve feedback created at or after this time.")
    ] = None,
    created_before: Annotated[
        datetime | None, typer.Option(help="Only retrieve feedback created at or before this time.")
    ] = None,
) -> None:
    """Chat with the agent using persistent conversation memory.

    With ``--message`` the command answers one message and prints a JSON
    response containing the answer and the ``conversation_id`` to reuse.
    Without ``--message`` it starts an interactive REPL reading from stdin
    (finish with ``exit``, ``quit``, or end-of-input).
    """
    configure_logging()
    settings = Settings(index_path=index_path, conversation_store_path=store_path)
    agent = build_agent(settings)
    store = build_conversation_store(settings)
    filters = _metadata_filters(
        customer_segment=customer_segment,
        channel=channel,
        min_rating=min_rating,
        max_rating=max_rating,
        created_after=created_after,
        created_before=created_before,
    )
    if message is not None:
        answer, resolved_id = agent.chat(
            message,
            store=store,
            conversation_id=conversation_id,
            top_k=top_k,
            filters=filters,
        )
        response = ChatResponse(conversation_id=resolved_id, result=answer)
        typer.echo(response.model_dump_json(indent=2))
        return
    typer.echo("Interactive chat. Type 'exit' or 'quit' to leave.", err=True)
    while True:
        try:
            line = input("you> ")
        except EOFError:
            break
        question = line.strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            break
        answer, conversation_id = agent.chat(
            question,
            store=store,
            conversation_id=conversation_id,
            top_k=top_k,
            filters=filters,
        )
        typer.echo(f"[conversation {conversation_id}]", err=True)
        typer.echo(answer.answer)
        typer.echo(render_citations(answer.citations), err=True)


@reports_app.command("save")
def reports_save(
    question: Annotated[str, typer.Argument(help="Question to answer and save.")],
    title: Annotated[str | None, typer.Option(help="Report title.")] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Report tag. Repeat for multiple tags."),
    ] = None,
    notes: Annotated[str | None, typer.Option(help="Optional report notes.")] = None,
    store_path: Annotated[
        Path, typer.Option(help="Directory holding saved report JSON files.")
    ] = Path(".artifacts/reports"),
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
) -> None:
    """Answer a question and save the resulting insight report."""
    configure_logging()
    settings = Settings(index_path=index_path, report_store_path=store_path)
    answer = build_agent(settings).answer(question, top_k=top_k)
    report = JsonInsightReportStore(store_path).save(
        SaveInsightReportRequest(
            title=title or _default_report_title(question),
            result=answer,
            tags=tag or [],
            notes=notes,
        )
    )
    typer.echo(report.model_dump_json(indent=2))


@reports_app.command("list")
def reports_list(
    store_path: Annotated[
        Path, typer.Option(help="Directory holding saved report JSON files.")
    ] = Path(".artifacts/reports"),
) -> None:
    """List saved insight report summaries."""
    configure_logging()
    summaries = JsonInsightReportStore(store_path).list()
    typer.echo(json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2))


@reports_app.command("get")
def reports_get(
    report_id: Annotated[str, typer.Argument(help="Report identifier.")],
    store_path: Annotated[
        Path, typer.Option(help="Directory holding saved report JSON files.")
    ] = Path(".artifacts/reports"),
) -> None:
    """Print one saved insight report."""
    configure_logging()
    try:
        report = JsonInsightReportStore(store_path).get(report_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if report is None:
        typer.echo(f"Report not found: {report_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(report.model_dump_json(indent=2))


@reports_app.command("email-summary")
def reports_email_summary(
    recipient: Annotated[
        list[str] | None,
        typer.Option("--recipient", help="Email recipient. Repeat for multiple recipients."),
    ] = None,
    report_id: Annotated[
        list[str] | None,
        typer.Option("--report-id", help="Report id to include. Repeat for multiple reports."),
    ] = None,
    subject: Annotated[str | None, typer.Option(help="Email subject override.")] = None,
    max_reports: Annotated[
        int, typer.Option(help="Maximum latest reports to include when --report-id is omitted.")
    ] = 5,
    send: Annotated[
        bool, typer.Option(help="Send through configured SMTP instead of dry-run.")
    ] = False,
    store_path: Annotated[
        Path, typer.Option(help="Directory holding saved report JSON files.")
    ] = Path(".artifacts/reports"),
) -> None:
    """Render or send an email digest for saved insight reports."""
    configure_logging()
    recipients = recipient or []
    if not recipients:
        typer.echo("At least one --recipient is required.", err=True)
        raise typer.Exit(code=2)
    try:
        request = EmailSummaryRequest(
            recipients=recipients,
            report_ids=report_id or [],
            subject=subject,
            max_reports=max_reports,
            send=send,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    settings = Settings(report_store_path=store_path)
    store = JsonInsightReportStore(store_path)
    try:
        reports = select_reports_for_summary(
            store,
            report_ids=request.report_ids,
            max_reports=request.max_reports,
        )
        summary = render_email_summary(
            reports,
            recipients=request.recipients,
            subject=request.subject,
        )
        result = (
            deliver_email_summary(
                summary,
                smtp_host=settings.email_smtp_host,
                smtp_port=settings.email_smtp_port,
                from_address=settings.email_from_address,
                username=settings.email_smtp_username,
                password=settings.email_smtp_password,
                use_tls=settings.email_smtp_use_tls,
            )
            if request.send
            else EmailSummaryDelivery(summary=summary, sent=False)
        )
    except LookupError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(result.model_dump_json(indent=2))


@answer_feedback_app.command("submit")
def answer_feedback_submit(
    question: Annotated[str, typer.Argument(help="Question to answer and review.")],
    rating: Annotated[
        HumanFeedbackRating,
        typer.Option(help="Human rating for the generated answer."),
    ],
    comment: Annotated[str | None, typer.Option(help="Optional human review comment.")] = None,
    report_id: Annotated[
        str | None, typer.Option(help="Optional saved report id linked to this feedback.")
    ] = None,
    tag: Annotated[
        list[str] | None,
        typer.Option("--tag", help="Feedback tag. Repeat for multiple tags."),
    ] = None,
    store_path: Annotated[
        Path, typer.Option(help="Directory holding answer feedback JSON files.")
    ] = Path(".artifacts/human_feedback"),
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
) -> None:
    """Answer a question and save the human judgement for that answer."""
    configure_logging()
    settings = Settings(index_path=index_path, human_feedback_store_path=store_path)
    answer = build_agent(settings).answer(question, top_k=top_k)
    try:
        record = JsonHumanFeedbackStore(store_path).save(
            SubmitHumanFeedbackRequest(
                result=answer,
                rating=rating,
                comment=comment,
                report_id=report_id,
                tags=tag or [],
            )
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    typer.echo(record.model_dump_json(indent=2))


@answer_feedback_app.command("list")
def answer_feedback_list(
    store_path: Annotated[
        Path, typer.Option(help="Directory holding answer feedback JSON files.")
    ] = Path(".artifacts/human_feedback"),
) -> None:
    """List human feedback summaries."""
    configure_logging()
    summaries = JsonHumanFeedbackStore(store_path).list()
    typer.echo(json.dumps([summary.model_dump(mode="json") for summary in summaries], indent=2))


@answer_feedback_app.command("get")
def answer_feedback_get(
    feedback_id: Annotated[str, typer.Argument(help="Answer feedback identifier.")],
    store_path: Annotated[
        Path, typer.Option(help="Directory holding answer feedback JSON files.")
    ] = Path(".artifacts/human_feedback"),
) -> None:
    """Print one human feedback record."""
    configure_logging()
    try:
        record = JsonHumanFeedbackStore(store_path).get(feedback_id)
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=2) from exc
    if record is None:
        typer.echo(f"Answer feedback not found: {feedback_id}", err=True)
        raise typer.Exit(code=1)
    typer.echo(record.model_dump_json(indent=2))


@app.command()
def evaluate(
    queries: Annotated[Path, typer.Option(help="Path to JSONL evaluation cases.")] = Path(
        "examples/queries.jsonl"
    ),
    output: Annotated[Path, typer.Option(help="Path for the JSON evaluation report.")] = Path(
        ".artifacts/evaluation_report.json"
    ),
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
    retriever: Annotated[
        RetrieverChoice, typer.Option(help="Retrieval strategy: dense, lexical, or hybrid.")
    ] = RetrieverChoice.dense,
    dense_weight: Annotated[
        float, typer.Option(help="Dense score weight for hybrid retrieval.")
    ] = 0.6,
    lexical_weight: Annotated[
        float, typer.Option(help="Lexical score weight for hybrid retrieval.")
    ] = 0.4,
) -> None:
    """Run offline retrieval and answer-quality evaluation and write a JSON report."""
    configure_logging()
    settings = Settings(
        index_path=index_path,
        retriever_type=retriever.value,
        dense_weight=dense_weight,
        lexical_weight=lexical_weight,
    )
    telemetry = build_telemetry(settings)
    vector_store = load_or_build_index(settings, telemetry=telemetry)
    query_engine = build_retriever(settings, vector_store)
    agent = build_agent(settings, telemetry=telemetry)
    cases = load_evaluation_cases(queries)
    report = evaluate_system(query_engine, agent, cases, top_k=top_k, telemetry=telemetry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"Evaluation report written to {output}", err=True)


@app.command()
def benchmark(
    queries: Annotated[Path, typer.Option(help="Path to JSONL benchmark queries.")] = Path(
        "examples/queries.jsonl"
    ),
    output: Annotated[
        Path, typer.Option(help="Directory for benchmark_results.json and .md.")
    ] = Path(".artifacts/benchmark_results"),
    dataset: Annotated[Path, typer.Option(help="Feedback CSV used to build the index.")] = Path(
        "data/sample_feedback.csv"
    ),
    repetitions: Annotated[int, typer.Option(help="Measured repetitions per phase.")] = 5,
    warmup: Annotated[int, typer.Option(help="Discarded warmup repetitions per phase.")] = 1,
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
) -> None:
    """Benchmark indexing, embedding, retrieval, and agent latency.

    Runs fully locally with the deterministic provider and writes a JSON report
    plus a Markdown results table into the output directory.
    """
    configure_logging()
    report = run_benchmark(
        dataset_path=dataset,
        queries_path=queries,
        repetitions=repetitions,
        warmup=warmup,
        top_k=top_k,
    )
    paths = write_benchmark_outputs(report, output)
    typer.echo(report.model_dump_json(indent=2))
    for filename, path in paths.items():
        typer.echo(f"{filename} written to {path}", err=True)


@experiment_app.command("run")
def experiment_run(
    config: Annotated[
        Path,
        typer.Option("--config", help="Path to a YAML experiment configuration."),
    ],
) -> None:
    """Run a configured experiment and write results, metrics, and metadata."""
    configure_logging()
    experiment_config = ExperimentConfig.from_yaml(config)
    result = run_experiment(experiment_config)
    metadata = collect_run_metadata(experiment_config)
    paths = write_experiment_outputs(result, metadata)
    tracking_run_id = track_experiment_run(result, metadata, paths)
    typer.echo(result.metrics.model_dump_json(indent=2))
    for filename, path in paths.items():
        typer.echo(f"{filename} written to {path}", err=True)
    if tracking_run_id is not None:
        typer.echo(f"tracking run id: {tracking_run_id}", err=True)


@prompts_app.command("list")
def prompts_list() -> None:
    """List registered prompts with versions, variables, and changelog notes."""
    for name in PROMPT_REGISTRY.names():
        latest_version = PROMPT_REGISTRY.get(name).version
        for template in PROMPT_REGISTRY.list_templates(name):
            marker = " (latest)" if template.version == latest_version else ""
            required = ", ".join(template.required_variables) or "-"
            optional = ", ".join(template.optional_variables) or "-"
            typer.echo(f"{template.name} {template.version}{marker}")
            typer.echo(f"  required variables: {required}")
            typer.echo(f"  optional variables: {optional}")
            typer.echo(f"  changelog: {template.changelog}")


@prompts_app.command("render")
def prompts_render(
    name: Annotated[str, typer.Option("--name", help="Prompt name, e.g. rag_answer.")],
    version: Annotated[
        str, typer.Option("--version", help="Prompt version, e.g. v1 or latest.")
    ] = LATEST_VERSION,
    var: Annotated[
        list[str] | None,
        typer.Option("--var", help="Template variable as key=value. Repeat for multiple."),
    ] = None,
) -> None:
    """Render a registered prompt template with the given variables."""
    variables: dict[str, str] = {}
    for item in var or []:
        key, separator, value = item.partition("=")
        if not separator or not key:
            typer.echo(f"Invalid --var {item!r}: expected key=value", err=True)
            raise typer.Exit(code=2)
        variables[key] = value
    try:
        rendered = PROMPT_REGISTRY.render(name, version, **variables)
    except (PromptNotFoundError, PromptVariableError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
    typer.echo(rendered)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host interface.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port number.")] = 8000,
) -> None:
    """Serve the FastAPI app with Uvicorn."""
    import uvicorn

    uvicorn.run("feedback_intelligence_agent.api:create_app", factory=True, host=host, port=port)
