"""Command-line interface for the AI engineering showcase."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from ai_engineering_showcase.citations import render_citations
from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.data_contracts import DataContractError, validate_feedback_csv
from ai_engineering_showcase.evaluation import evaluate_system, load_evaluation_cases
from ai_engineering_showcase.experiments import (
    ExperimentConfig,
    collect_run_metadata,
    run_experiment,
    write_experiment_outputs,
)
from ai_engineering_showcase.factory import (
    build_agent,
    build_index,
    build_retriever,
    load_or_build_index,
)
from ai_engineering_showcase.telemetry import configure_logging

app = typer.Typer(help="AI Engineering Showcase CLI")
experiment_app = typer.Typer(help="Run repeatable experiments over RAG configurations.")
app.add_typer(experiment_app, name="experiment")


class RetrieverChoice(str, Enum):
    """Retriever strategies selectable from the command line."""

    dense = "dense"
    lexical = "lexical"
    hybrid = "hybrid"


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
    vector_store = build_index(input, index_path, embedding_dim=embedding_dim)
    typer.echo(f"Indexed {vector_store.size} chunks into {index_path}")


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
    answer = agent.answer(question, top_k=top_k)
    typer.echo(answer.model_dump_json(indent=2))
    typer.echo(render_citations(answer.citations), err=True)


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
    vector_store = load_or_build_index(settings)
    query_engine = build_retriever(settings, vector_store)
    agent = build_agent(settings)
    cases = load_evaluation_cases(queries)
    report = evaluate_system(query_engine, agent, cases, top_k=top_k)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(report.model_dump_json(indent=2))
    typer.echo(f"Evaluation report written to {output}", err=True)


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
    typer.echo(result.metrics.model_dump_json(indent=2))
    for filename, path in paths.items():
        typer.echo(f"{filename} written to {path}", err=True)


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host interface.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port number.")] = 8000,
) -> None:
    """Serve the FastAPI app with Uvicorn."""
    import uvicorn

    uvicorn.run("ai_engineering_showcase.api:create_app", factory=True, host=host, port=port)
