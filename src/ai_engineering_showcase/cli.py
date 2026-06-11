"""Command-line interface for the AI engineering showcase."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ai_engineering_showcase.config import Settings
from ai_engineering_showcase.embeddings import HashingEmbeddingModel
from ai_engineering_showcase.evaluation import evaluate_system, load_evaluation_cases
from ai_engineering_showcase.factory import build_agent, build_index, load_or_build_index
from ai_engineering_showcase.retrieval import QueryEngine
from ai_engineering_showcase.telemetry import configure_logging

app = typer.Typer(help="AI Engineering Showcase CLI")


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


@app.command()
def query(
    question: Annotated[str, typer.Argument(help="Question to ask over feedback data.")],
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
) -> None:
    """Ask a question against the indexed feedback."""
    configure_logging()
    settings = Settings(index_path=index_path)
    agent = build_agent(settings)
    answer = agent.answer(question, top_k=top_k)
    typer.echo(answer.model_dump_json(indent=2))


@app.command()
def evaluate(
    eval_path: Annotated[Path, typer.Option(help="Path to JSONL evaluation cases.")] = Path(
        "examples/queries.jsonl"
    ),
    index_path: Annotated[Path, typer.Option(help="Path to vector index.")] = Path(
        ".artifacts/vector_store.json"
    ),
    top_k: Annotated[int, typer.Option(help="Number of chunks to retrieve.")] = 4,
) -> None:
    """Run offline retrieval and answer quality evaluation."""
    configure_logging()
    settings = Settings(index_path=index_path)
    vector_store = load_or_build_index(settings)
    embedding_model = HashingEmbeddingModel(dim=vector_store.dim)
    query_engine = QueryEngine(embedding_model=embedding_model, vector_store=vector_store)
    agent = build_agent(settings)
    cases = load_evaluation_cases(eval_path)
    report = evaluate_system(query_engine, agent, cases, top_k=top_k)
    typer.echo(report.model_dump_json(indent=2))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Host interface.")] = "127.0.0.1",
    port: Annotated[int, typer.Option(help="Port number.")] = 8000,
) -> None:
    """Serve the FastAPI app with Uvicorn."""
    import uvicorn

    uvicorn.run("ai_engineering_showcase.api:create_app", factory=True, host=host, port=port)
