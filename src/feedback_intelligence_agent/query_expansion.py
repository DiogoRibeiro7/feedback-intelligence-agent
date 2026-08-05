"""Deterministic query expansion for product-specific terminology."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class QueryExpansion:
    """Result of expanding a user query with domain terminology."""

    original_query: str
    expanded_query: str
    added_terms: tuple[str, ...]

    @property
    def was_expanded(self) -> bool:
        """Return True when expansion added at least one term."""
        return bool(self.added_terms)


class QueryExpander(Protocol):
    """Protocol for deterministic or model-backed query expanders."""

    def expand(self, query: str) -> QueryExpansion:
        """Return an expanded query plus the terms that were added."""
        ...


TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+")


PRODUCT_TERMINOLOGY: dict[str, tuple[str, ...]] = {
    "crm": ("salesforce", "hubspot", "integration", "sync"),
    "integration": ("salesforce", "hubspot", "crm", "sync"),
    "integrations": ("integration", "salesforce", "hubspot", "crm", "sync"),
    "connector": ("integration", "salesforce", "hubspot", "sync"),
    "connectors": ("integration", "salesforce", "hubspot", "sync"),
    "sfdc": ("salesforce", "integration", "sync"),
    "salesforce": ("crm", "integration", "sync"),
    "hubspot": ("crm", "integration", "sync"),
    "bi": ("reporting", "dashboard", "export", "analytics"),
    "report": ("reporting", "dashboard", "export", "analytics"),
    "reports": ("reporting", "dashboard", "export", "analytics"),
    "analytics": ("reporting", "dashboard", "export"),
    "csm": ("customer success", "support", "onboarding", "implementation"),
    "nps": ("survey", "feedback", "satisfaction", "score"),
    "renewal": ("pricing", "value", "finance"),
    "month-end": ("reporting", "export", "analytics"),
}


class ProductTerminologyExpander:
    """Expand product-specific aliases into terms present in feedback data.

    The mapping is intentionally small and deterministic. It handles common
    shorthand that product, support, and customer-success teams use in questions
    but that raw feedback often expresses differently.
    """

    def __init__(
        self,
        terminology: dict[str, tuple[str, ...]] | None = None,
    ) -> None:
        """Create an expander from a trigger-to-terms map."""
        self.terminology = terminology or PRODUCT_TERMINOLOGY

    def expand(self, query: str) -> QueryExpansion:
        """Append mapped product terms when trigger terms appear in the query."""
        if not query.strip():
            return QueryExpansion(original_query=query, expanded_query=query, added_terms=())

        tokens = {token.lower() for token in TOKEN_PATTERN.findall(query)}
        lower_query = query.lower()
        added: list[str] = []
        for trigger, terms in self.terminology.items():
            if trigger not in tokens and trigger not in lower_query:
                continue
            for term in terms:
                if term.lower() in lower_query or term.lower() in added:
                    continue
                added.append(term)

        if not added:
            return QueryExpansion(original_query=query, expanded_query=query, added_terms=())
        expanded_query = f"{query} {' '.join(added)}"
        return QueryExpansion(
            original_query=query,
            expanded_query=expanded_query,
            added_terms=tuple(added),
        )
