"""Command-line interface for the AI engineering showcase."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Annotated

import typer

from ai_engineering_showcase.benchmarking import run_benchmark, write_benchmark_outputs
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
    build_conversation_store,
    build_index,
    build_retriever,
    build_telemetry,
    load_or_build_index,
)
from ai_engineering_showcase.prompt_registry import (
    LATEST_VERSION,
    PromptNotFoundError,
    PromptVariableError,
)
from ai_engineering_showcase.prompts import PROMPT_REGISTRY
from ai_engineering_showcase.schemas import ChatResponse
from ai_engineering_showcase.synthetic_data import SyntheticDataConfig, write_feedback_csv
from ai_engineering_showcase.telemetry import configure_logging

app = typer.Typer(help="AI Engineering Showcase CLI")
experiment_app = typer.Typer(help="Run repeatable experiments over RAG configurations.")
app.add_typer(experiment_app, name="experiment")
prompts_app = typer.Typer(help="Inspect and render versioned prompt templates.")
app.add_typer(prompts_app, name="prompts")


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
    telemetry = build_telemetry(Settings(index_path=index_path))
    vector_store = build_index(input, index_path, embedding_dim=embedding_dim, telemetry=telemetry)
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
    if message is not None:
        answer, resolved_id = agent.chat(
            message, store=store, conversation_id=conversation_id, top_k=top_k
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
            question, store=store, conversation_id=conversation_id, top_k=top_k
        )
        typer.echo(f"[conversation {conversation_id}]", err=True)
        typer.echo(answer.answer)
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
    typer.echo(result.metrics.model_dump_json(indent=2))
    for filename, path in paths.items():
        typer.echo(f"{filename} written to {path}", err=True)


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

    uvicorn.run("ai_engineering_showcase.api:create_app", factory=True, host=host, port=port)
