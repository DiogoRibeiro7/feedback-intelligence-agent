from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from feedback_intelligence_agent.agent import FeedbackInsightAgent
from feedback_intelligence_agent.api import create_app
from feedback_intelligence_agent.config import Settings
from feedback_intelligence_agent.factory import build_conversation_store

KEYS = {
    "reader": "test-reader-secret",
    "writer": "test-writer-secret",
    "admin": "test-admin-secret",
}


def headers(role: str) -> dict[str, str]:
    return {"X-Feedback-Agent-Key": KEYS[role]}


@pytest.fixture()
def private_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data_path = tmp_path / "feedback.csv"
    data_path.write_text(
        "tenant_id,feedback_id,customer_segment,channel,rating,text,created_at\n"
        "acme,shared,enterprise,support_ticket,1,Acme onboarding needs owners.,2026-01-01\n"
        "cobalt,shared,enterprise,support_ticket,2,Cobalt onboarding needs guides.,2026-01-01\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("FEEDBACK_AGENT_DATA_PATH", str(data_path))
    monkeypatch.setenv("FEEDBACK_AGENT_INDEX_PATH", str(tmp_path / "index.json"))
    monkeypatch.setenv("FEEDBACK_AGENT_LLM_PROVIDER", "local")
    monkeypatch.setenv("FEEDBACK_AGENT_API_AUTH_ENABLED", "true")
    for role, key in KEYS.items():
        monkeypatch.setenv(f"FEEDBACK_AGENT_API_{role.upper()}_KEY", key)
    for name in ("CONVERSATION", "JOB", "REPORT", "HUMAN_FEEDBACK", "ACTIVE_LEARNING_STATE"):
        monkeypatch.setenv(f"FEEDBACK_AGENT_{name}_STORE_PATH", str(tmp_path / name.lower()))
    return TestClient(create_app())


def start_conversation(client: TestClient, role: str = "writer", **fields: object) -> str:
    response = client.post(
        "/chat", headers=headers(role), json={"message": "Why is onboarding slow?", **fields}
    )
    assert response.status_code == 200, response.text
    return str(response.json()["conversation_id"])


@pytest.mark.parametrize(
    ("owner", "other"), [("writer", "reader"), ("writer", "admin"), ("admin", "writer")]
)
def test_conversation_reads_are_private_to_the_authenticated_key(
    private_client: TestClient, owner: str, other: str
) -> None:
    conversation_id = start_conversation(private_client, owner)
    assert (
        private_client.get(f"/conversations/{conversation_id}", headers=headers(owner)).status_code
        == 200
    )
    denied = private_client.get(f"/conversations/{conversation_id}", headers=headers(other))
    unknown = private_client.get("/conversations/unknown", headers=headers(other))
    assert denied.status_code == unknown.status_code == 404
    assert denied.json() == unknown.json()


@pytest.mark.parametrize(("owner", "other"), [("writer", "admin"), ("admin", "writer")])
def test_other_key_cannot_continue_or_overwrite_a_conversation(
    private_client: TestClient,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    owner: str,
    other: str,
) -> None:
    conversation_id = start_conversation(private_client, owner)
    before = {path: path.read_bytes() for path in (tmp_path / "conversation").rglob("*.json")}

    def unexpected_answer(*args: object, **kwargs: object) -> None:
        pytest.fail("unauthorized history reached answer generation")

    monkeypatch.setattr(FeedbackInsightAgent, "answer", unexpected_answer)
    response = private_client.post(
        "/chat",
        headers=headers(other),
        json={"message": "What about pricing?", "conversation_id": conversation_id},
    )
    assert response.status_code == 404
    assert {
        path: path.read_bytes() for path in (tmp_path / "conversation").rglob("*.json")
    } == before


def test_authenticated_callers_cannot_create_chosen_conversation_ids(
    private_client: TestClient,
) -> None:
    response = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={"message": "Why is onboarding slow?", "conversation_id": "chosen-id"},
    )
    assert response.status_code == 404
    assert (
        private_client.get("/conversations/chosen-id", headers=headers("writer")).status_code == 404
    )


def test_owned_history_survives_restart_and_does_not_contain_api_keys(
    private_client: TestClient, tmp_path: Path
) -> None:
    conversation_id = start_conversation(private_client)
    with TestClient(create_app()) as restarted:
        response = restarted.post(
            "/chat",
            headers=headers("writer"),
            json={"message": "What about pricing?", "conversation_id": conversation_id},
        )
        assert response.status_code == 200
        stored = restarted.get(f"/conversations/{conversation_id}", headers=headers("writer"))
        assert len(stored.json()["turns"]) == 2
    files = list((tmp_path / "conversation").rglob("*.json"))
    assert len(files) == 1
    assert files[0].parent != tmp_path / "conversation"
    for key in KEYS.values():
        assert key not in str(files[0])
        assert key not in files[0].read_text(encoding="utf-8")
        assert key not in stored.text


def test_rotated_key_does_not_inherit_the_previous_keys_history(
    private_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = start_conversation(private_client)
    monkeypatch.setenv("FEEDBACK_AGENT_API_WRITER_KEY", "rotated-writer-key")
    with TestClient(create_app()) as restarted:
        response = restarted.get(
            f"/conversations/{conversation_id}",
            headers={"X-Feedback-Agent-Key": "rotated-writer-key"},
        )
        assert response.status_code == 404


def test_authenticated_key_cannot_adopt_legacy_unowned_history(
    private_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_API_AUTH_ENABLED", "false")
    with TestClient(create_app()) as local:
        response = local.post(
            "/chat", json={"message": "Why is onboarding slow?", "conversation_id": "legacy"}
        )
        assert response.status_code == 200
    assert private_client.get("/conversations/legacy", headers=headers("writer")).status_code == 404
    response = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={"message": "What about pricing?", "conversation_id": "legacy"},
    )
    assert response.status_code == 404


def test_followup_inherits_its_conversation_tenant(private_client: TestClient) -> None:
    conversation_id = start_conversation(private_client, tenant_id="Acme")
    response = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={"message": "What about onboarding?", "conversation_id": conversation_id},
    )
    assert response.status_code == 200
    citations = response.json()["result"]["citations"]
    assert citations and all(item["document_id"].startswith("acme:") for item in citations)
    stored = private_client.get(f"/conversations/{conversation_id}", headers=headers("writer"))
    assert stored.json()["tenant_id"] == "acme"


@pytest.mark.parametrize("initial_tenant", [None, "acme"])
def test_conversation_tenant_cannot_change(
    private_client: TestClient, tmp_path: Path, initial_tenant: str | None
) -> None:
    conversation_id = start_conversation(private_client, tenant_id=initial_tenant)
    response = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={
            "message": "What about pricing?",
            "conversation_id": conversation_id,
            "tenant_id": "cobalt",
        },
    )
    assert response.status_code == 400
    assert "tenant" in response.json()["detail"]
    path = next((tmp_path / "conversation").rglob(f"{conversation_id}.json"))
    assert len(json.loads(path.read_text(encoding="utf-8"))["turns"]) == 1


def test_same_tenant_and_other_filters_remain_usable(private_client: TestClient) -> None:
    conversation_id = start_conversation(private_client, tenant_id="acme")
    same_tenant = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={
            "message": "What about onboarding?",
            "conversation_id": conversation_id,
            "tenant_id": " ACME ",
        },
    )
    assert same_tenant.status_code == 200
    assert same_tenant.json()["result"]["citations"]
    filtered = private_client.post(
        "/chat",
        headers=headers("writer"),
        json={
            "message": "What about onboarding?",
            "conversation_id": conversation_id,
            "min_rating": 2,
        },
    )
    assert filtered.status_code == 200
    assert filtered.json()["result"]["citations"] == []


def test_promoted_and_demoted_key_keeps_its_own_history(
    private_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = start_conversation(private_client)
    monkeypatch.setenv("FEEDBACK_AGENT_API_ADMIN_KEY", KEYS["writer"])
    with TestClient(create_app()) as promoted:
        assert (
            promoted.get(f"/conversations/{conversation_id}", headers=headers("writer")).status_code
            == 200
        )
    monkeypatch.setenv("FEEDBACK_AGENT_API_ADMIN_KEY", KEYS["admin"])
    monkeypatch.setenv("FEEDBACK_AGENT_API_READER_KEY", KEYS["writer"])
    monkeypatch.setenv("FEEDBACK_AGENT_API_WRITER_KEY", "")
    with TestClient(create_app()) as demoted:
        assert (
            demoted.get(f"/conversations/{conversation_id}", headers=headers("writer")).status_code
            == 200
        )
        assert (
            demoted.post(
                "/chat",
                headers=headers("writer"),
                json={"message": "Follow up", "conversation_id": conversation_id},
            ).status_code
            == 403
        )


def test_disabling_auth_does_not_expose_authenticated_namespaces(
    private_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    conversation_id = start_conversation(private_client)
    monkeypatch.setenv("FEEDBACK_AGENT_API_AUTH_ENABLED", "false")
    with TestClient(create_app()) as local:
        assert local.get(f"/conversations/{conversation_id}").status_code == 404


def test_namespaces_cannot_escape_the_conversation_root(tmp_path: Path) -> None:
    settings = Settings(conversation_store_path=tmp_path / "conversations")
    first = build_conversation_store(settings, principal_id="../../outside")
    second = build_conversation_store(settings, principal_id="other")
    assert first.root.is_relative_to(settings.conversation_store_path)
    assert first.root != second.root
    assert "outside" not in str(first.root)


@pytest.mark.parametrize(
    "question", ["What is the sentiment distribution?", "What are the most common issues?"]
)
def test_query_and_stream_tools_respect_tenant_filters(
    private_client: TestClient, question: str
) -> None:
    payload = {"question": question, "tenant_id": " ACME ", "top_k": 1}
    response = private_client.post("/query", headers=headers("reader"), json=payload)
    assert response.status_code == 200
    output = response.json()["result"]["tool_run"]
    assert output["status"] == "ok"
    assert output["output"]["total_records"] == 1
    assert "cobalt" not in json.dumps(output).lower()

    streamed = private_client.post("/query/stream", headers=headers("reader"), json=payload)
    assert streamed.status_code == 200
    content = "".join(
        json.loads(line.removeprefix("data: ")).get("text", "")
        for line in streamed.text.splitlines()
        if line.startswith("data: ")
    )
    assert content == response.json()["result"]["answer"]
    assert "across 1 feedback records" in content


@pytest.mark.parametrize(
    "question", ["What is the sentiment distribution?", "What are the most common issues?"]
)
def test_chat_tools_inherit_tenant_and_apply_followup_filters(
    private_client: TestClient, question: str
) -> None:
    conversation_id = start_conversation(private_client, tenant_id="acme")
    payload = {"message": question, "conversation_id": conversation_id, "top_k": 1}
    response = private_client.post("/chat", headers=headers("writer"), json=payload)
    assert response.status_code == 200
    output = response.json()["result"]["tool_run"]
    assert output["status"] == "ok"
    assert output["output"]["total_records"] == 1
    assert "cobalt" not in json.dumps(output).lower()

    empty = private_client.post(
        "/chat", headers=headers("writer"), json={**payload, "min_rating": 2}
    )
    assert empty.status_code == 200
    assert empty.json()["result"]["tool_run"]["output"]["total_records"] == 0
    stored = private_client.get(f"/conversations/{conversation_id}", headers=headers("writer"))
    assert "cobalt" not in stored.text.lower()
