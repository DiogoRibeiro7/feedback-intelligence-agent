from __future__ import annotations

import pytest

from feedback_intelligence_agent.prompt_registry import (
    LATEST_VERSION,
    PromptNotFoundError,
    PromptRegistry,
    PromptTemplate,
    PromptVariableError,
    template_placeholders,
)
from feedback_intelligence_agent.prompts import PROMPT_REGISTRY


def build_registry() -> PromptRegistry:
    registry = PromptRegistry()
    registry.register(
        PromptTemplate(
            name="greeting",
            version="v1",
            template="Hello {name}!",
            required_variables=("name",),
            changelog="Initial greeting.",
        )
    )
    registry.register(
        PromptTemplate(
            name="greeting",
            version="v2",
            template="Hello {name}, welcome to {place}!",
            required_variables=("name",),
            optional_variables={"place": "the platform"},
            changelog="Added a place variable with a default.",
        )
    )
    return registry


def test_template_placeholders_extracts_named_fields() -> None:
    assert template_placeholders("A {x} and {y} and {x}") == {"x", "y"}


def test_template_placeholders_rejects_non_identifier_fields() -> None:
    with pytest.raises(PromptVariableError):
        template_placeholders("Bad {0} placeholder")


def test_latest_resolves_to_most_recent_version() -> None:
    registry = build_registry()
    assert registry.get("greeting").version == "v2"
    assert registry.get("greeting", LATEST_VERSION).version == "v2"


def test_old_versions_remain_available() -> None:
    registry = build_registry()
    template = registry.get("greeting", "v1")
    assert template.version == "v1"
    assert template.render(name="Ada") == "Hello Ada!"


def test_render_applies_optional_variable_defaults() -> None:
    registry = build_registry()
    assert registry.render("greeting", name="Ada") == "Hello Ada, welcome to the platform!"
    assert (
        registry.render("greeting", "v2", name="Ada", place="Lisbon")
        == "Hello Ada, welcome to Lisbon!"
    )


def test_missing_required_variable_raises_clear_error() -> None:
    registry = build_registry()
    with pytest.raises(PromptVariableError) as excinfo:
        registry.render("greeting", "v1")
    message = str(excinfo.value)
    assert "missing required variable" in message
    assert "greeting" in message
    assert "v1" in message
    assert "name" in message


def test_unknown_variable_raises_clear_error() -> None:
    registry = build_registry()
    with pytest.raises(PromptVariableError, match="unknown variable"):
        registry.render("greeting", "v1", name="Ada", surprise="x")


def test_unknown_prompt_name_raises() -> None:
    registry = build_registry()
    with pytest.raises(PromptNotFoundError, match="unknown prompt 'nope'"):
        registry.get("nope")


def test_unknown_version_raises() -> None:
    registry = build_registry()
    with pytest.raises(PromptNotFoundError, match="unknown version 'v9'"):
        registry.get("greeting", "v9")


def test_duplicate_registration_rejected() -> None:
    registry = build_registry()
    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            PromptTemplate(
                name="greeting", version="v1", template="Hi {name}", required_variables=("name",)
            )
        )


def test_reserved_latest_version_rejected() -> None:
    registry = PromptRegistry()
    with pytest.raises(ValueError, match="reserved"):
        registry.register(PromptTemplate(name="x", version=LATEST_VERSION, template="hi"))


def test_template_declaration_must_match_placeholders() -> None:
    with pytest.raises(PromptVariableError, match="placeholders missing from declaration"):
        PromptTemplate(name="x", version="v1", template="Hello {name}")
    with pytest.raises(PromptVariableError, match="declared variables missing from template"):
        PromptTemplate(name="x", version="v1", template="Hello", required_variables=("name",))


def test_variable_cannot_be_required_and_optional() -> None:
    with pytest.raises(PromptVariableError, match="both required and optional"):
        PromptTemplate(
            name="x",
            version="v1",
            template="Hello {name}",
            required_variables=("name",),
            optional_variables={"name": "Ada"},
        )


def test_default_registry_exposes_production_prompts() -> None:
    assert PROMPT_REGISTRY.names() == ("rag_answer", "rag_system")
    rag_answer = PROMPT_REGISTRY.get("rag_answer")
    assert rag_answer.version == "v1"
    assert rag_answer.required_variables == ("question",)
    assert set(rag_answer.optional_variables) == {"route", "context"}
    assert rag_answer.changelog
    assert PROMPT_REGISTRY.versions("rag_answer") == ("v1",)
