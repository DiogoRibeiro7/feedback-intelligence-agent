from __future__ import annotations

from feedback_intelligence_agent.output_parser import parse_llm_output


def test_parse_llm_output_accepts_valid_json() -> None:
    parsed = parse_llm_output(
        '{"answer": "Onboarding is slow [1].", '
        '"recommended_actions": ["Create checklist.", "Assign owner."]}'
    )

    assert parsed.payload.answer == "Onboarding is slow [1]."
    assert parsed.payload.recommended_actions == ["Create checklist.", "Assign owner."]
    assert parsed.output_format == "json"
    assert parsed.repair_applied is False
    assert parsed.validation_error is None


def test_parse_llm_output_repairs_embedded_json_and_aliases() -> None:
    parsed = parse_llm_output(
        'Here is the answer:\n{"answer": "Exports failed [2].", '
        '"actions": "- Improve errors.\\n- Retry export jobs."}\nThanks.'
    )

    assert parsed.payload.answer == "Exports failed [2]."
    assert parsed.payload.recommended_actions == ["Improve errors.", "Retry export jobs."]
    assert parsed.output_format == "repaired_json"
    assert parsed.repair_applied is True


def test_parse_llm_output_repairs_sectioned_legacy_output() -> None:
    parsed = parse_llm_output(
        "\n".join(
            [
                "Answer:",
                "The strongest signal is onboarding [1].",
                "",
                "Recommended actions:",
                "- Create a checklist.",
                "* Assign one owner.",
                "",
                "Citations:",
                "- [1] fb-001",
            ]
        )
    )

    assert parsed.payload.answer == "The strongest signal is onboarding [1]."
    assert parsed.payload.recommended_actions == ["Create a checklist.", "Assign one owner."]
    assert parsed.output_format == "sectioned"
    assert parsed.repair_applied is True


def test_parse_llm_output_falls_back_to_raw_text() -> None:
    parsed = parse_llm_output("Plain answer without a declared structure.")

    assert parsed.payload.answer == "Plain answer without a declared structure."
    assert parsed.payload.recommended_actions == []
    assert parsed.output_format == "raw"
    assert parsed.repair_applied is True
