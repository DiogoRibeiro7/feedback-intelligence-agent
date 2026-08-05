from __future__ import annotations

from pathlib import Path

import pytest

from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.evaluation import load_evaluation_cases, recall_at_k
from feedback_intelligence_agent.factory import build_agent, build_retriever, load_or_build_index
from feedback_intelligence_agent.retrieval import Retriever


def build_settings(tmp_path: Path, retriever_type: str) -> Settings:
    return Settings.model_validate(
        {
            "data_path": Path("data/sample_feedback.csv"),
            "index_path": tmp_path / f"{retriever_type}_vector_store.json",
            "retriever_type": retriever_type,
            "dense_weight": 0.5,
            "lexical_weight": 0.5,
        }
    )


def build_sample_retriever(tmp_path: Path, retriever_type: str) -> Retriever:
    settings = build_settings(tmp_path, retriever_type)
    vector_store = load_or_build_index(settings)
    return build_retriever(settings, vector_store)


@pytest.mark.parametrize("retriever_type", ["lexical", "hybrid"])
def test_expanded_retrievers_cite_relevant_context(
    tmp_path: Path,
    retriever_type: str,
) -> None:
    settings = build_settings(tmp_path, retriever_type)
    agent = build_agent(settings)
    cases = [case for case in load_evaluation_cases("examples/queries.jsonl") if case.is_answerable]

    for case in cases:
        answer = agent.answer(case.question, top_k=4)
        cited_ids = [citation.document_id for citation in answer.citations]
        assert set(cited_ids).intersection(case.relevant_document_ids), case.question


def test_dense_retriever_cites_core_known_context(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, "dense")
    agent = build_agent(settings)
    cases = [
        case
        for case in load_evaluation_cases("examples/queries.jsonl")
        if case.question
        in {
            "Why are enterprise customers unhappy with onboarding?",
            "What product issues affect reporting workflows?",
            "What should we improve in support operations?",
        }
    ]

    for case in cases:
        cited_ids = [
            citation.document_id for citation in agent.answer(case.question, top_k=4).citations
        ]
        assert set(cited_ids).intersection(case.relevant_document_ids), case.question


@pytest.mark.parametrize(
    ("retriever_type", "question", "expected_cited_ids", "top_n"),
    [
        (
            "dense",
            "Why are enterprise customers unhappy with onboarding?",
            ["fb-001", "fb-007", "fb-009"],
            3,
        ),
        (
            "lexical",
            "Which integrations should the product team prioritise?",
            ["fb-005", "fb-010"],
            4,
        ),
        (
            "hybrid",
            "What product issues affect reporting workflows?",
            ["fb-002", "fb-011"],
            2,
        ),
        (
            "hybrid",
            "Which CRM connector problems were reported?",
            ["fb-005", "fb-010"],
            4,
        ),
    ],
)
def test_known_cited_context_does_not_drift(
    tmp_path: Path,
    retriever_type: str,
    question: str,
    expected_cited_ids: list[str],
    top_n: int,
) -> None:
    agent = build_agent(build_settings(tmp_path, retriever_type))

    cited_ids = [citation.document_id for citation in agent.answer(question, top_k=4).citations]

    assert set(expected_cited_ids).issubset(cited_ids[:top_n])


def test_sample_retrieval_regression_recall_threshold(tmp_path: Path) -> None:
    settings = build_settings(tmp_path, "hybrid")
    agent = build_agent(settings)
    cases = [case for case in load_evaluation_cases("examples/queries.jsonl") if case.is_answerable]
    recalls = []

    for case in cases:
        cited_ids = [
            citation.document_id for citation in agent.answer(case.question, top_k=4).citations
        ]
        recalls.append(recall_at_k(cited_ids, case.relevant_document_ids, 4))

    assert sum(recalls) / len(recalls) >= 0.75


def test_plural_product_terms_retrieve_singular_feedback_terms(tmp_path: Path) -> None:
    retriever = build_sample_retriever(tmp_path, "lexical")

    retrieved_ids = [
        result.chunk.source_id
        for result in retriever.search(
            "Which integrations should the product team prioritise?",
            top_k=4,
        )
    ]

    assert {"fb-005", "fb-010"}.issubset(retrieved_ids)
