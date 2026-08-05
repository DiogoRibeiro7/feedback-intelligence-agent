from __future__ import annotations

from feedback_intelligence_agent.lexical_search import BM25Retriever
from feedback_intelligence_agent.query_expansion import ProductTerminologyExpander
from feedback_intelligence_agent.schemas import DocumentChunk


def make_chunk(suffix: str, text: str) -> DocumentChunk:
    return DocumentChunk(chunk_id=f"c-{suffix}", source_id=f"fb-{suffix}", text=text, metadata={})


def test_product_terminology_expander_adds_domain_aliases() -> None:
    expander = ProductTerminologyExpander()
    expansion = expander.expand("Which CRM connector is broken?")
    assert expansion.was_expanded is True
    assert expansion.original_query == "Which CRM connector is broken?"
    assert "salesforce" in expansion.expanded_query.lower()
    assert "hubspot" in expansion.expanded_query.lower()
    assert "integration" in expansion.added_terms
    assert "sync" in expansion.added_terms


def test_product_terminology_expander_leaves_unknown_queries_unchanged() -> None:
    expander = ProductTerminologyExpander()
    expansion = expander.expand("Why is onboarding slow?")
    assert expansion.was_expanded is False
    assert expansion.expanded_query == "Why is onboarding slow?"
    assert expansion.added_terms == ()


def test_query_expansion_improves_lexical_retrieval_for_product_aliases() -> None:
    chunks = [
        make_chunk("salesforce", "Salesforce integration broke after the release."),
        make_chunk("hubspot", "HubSpot integration is required before expansion."),
        make_chunk("pricing", "Pricing renewal was hard to explain to finance."),
    ]
    raw = BM25Retriever(chunks)
    expanded = BM25Retriever(chunks, query_expander=ProductTerminologyExpander())

    assert raw.search("CRM connector", top_k=3) == []

    results = expanded.search("CRM connector", top_k=3)
    assert [result.chunk.source_id for result in results[:2]] == [
        "fb-salesforce",
        "fb-hubspot",
    ]
    assert all(result.score > 0 for result in results)


def test_query_expansion_supports_reporting_shorthand() -> None:
    chunks = [
        make_chunk("dashboard", "The dashboard export failed during reporting."),
        make_chunk("setup", "Setup was fast and templates were useful."),
    ]
    expanded = BM25Retriever(chunks, query_expander=ProductTerminologyExpander())

    results = expanded.search("BI problem", top_k=2)
    assert results[0].chunk.source_id == "fb-dashboard"
