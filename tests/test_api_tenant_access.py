"""Exercise tenant authorization through real API routes and persisted resources."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from feedback_intelligence_agent.api import create_app

READ = {"X-Feedback-Agent-Key": "tenant-read"}
WRITE = {"X-Feedback-Agent-Key": "tenant-write"}
ADMIN = {"X-Feedback-Agent-Key": "global-admin"}
BINDINGS = {"tenant-read": "acme", "tenant-write": "cobalt"}
QUESTION = "What feedback mentions the onboarding checklist?"


@pytest.fixture()
def tenant_api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    data = tmp_path / "feedback.csv"
    data.write_text(
        "tenant_id,feedback_id,customer_segment,channel,rating,text,created_at\n"
        "acme,shared,enterprise,support_ticket,1,Acme onboarding checklist is unclear.,2026-05-01\n"
        "cobalt,shared,enterprise,support_ticket,2,"
        "Cobalt onboarding checklist lacks owners.,2026-05-02\n"
        "default,shared,enterprise,support_ticket,3,"
        "Default onboarding checklist needs help.,2026-05-03\n",
        encoding="utf-8",
    )
    for setting, value in {
        "DATA_PATH": str(data),
        "INDEX_PATH": str(tmp_path / "index.json"),
        "CONVERSATION_STORE_PATH": str(tmp_path / "conversations"),
        "REPORT_STORE_PATH": str(tmp_path / "reports"),
        "HUMAN_FEEDBACK_STORE_PATH": str(tmp_path / "feedback"),
        "ACTIVE_LEARNING_STATE_STORE_PATH": str(tmp_path / "states"),
        "JOB_STORE_PATH": str(tmp_path / "jobs"),
        "API_AUTH_ENABLED": "true",
        "API_READER_KEY": "tenant-read",
        "API_WRITER_KEY": "tenant-write",
        "API_ADMIN_KEY": "global-admin",
        "API_TENANT_BINDINGS": json.dumps(BINDINGS),
        "LLM_PROVIDER": "local",
    }.items():
        monkeypatch.setenv(f"FEEDBACK_AGENT_{setting}", value)
    return TestClient(create_app())


def answer(client: TestClient, tenant: str = "cobalt") -> dict[str, Any]:
    response = client.post(
        "/query", json={"question": QUESTION, "tenant_id": tenant}, headers=ADMIN
    )
    assert response.status_code == 200, response.text
    return response.json()["result"]


@pytest.fixture()
def resources(tenant_api: TestClient) -> dict[str, dict[str, str]]:
    saved = {}
    for tenant in ("acme", "cobalt", "default"):
        result = answer(tenant_api, tenant)
        report = tenant_api.post(
            "/reports",
            json={"title": f"{tenant} private report", "result": result, "tenant_id": tenant},
            headers=ADMIN,
        )
        assert report.status_code == 201
        report_id = report.json()["report_id"]
        feedback = tenant_api.post(
            "/answer-feedback",
            json={
                "result": result,
                "rating": "not_useful",
                "tenant_id": tenant,
                "report_id": report_id,
            },
            headers=ADMIN,
        )
        assert feedback.status_code == 201
        feedback_id = feedback.json()["feedback_id"]
        state = tenant_api.patch(
            f"/answer-feedback/active-learning/{feedback_id}",
            json={"status": "assigned", "notes": f"{tenant} private notes"},
            headers=ADMIN,
        )
        assert state.status_code == 200
        saved[tenant] = {"report": report_id, "feedback": feedback_id}
    return saved


@pytest.mark.parametrize("path", ["/query", "/query/stream"])
@pytest.mark.parametrize("headers,tenant", [(READ, "acme"), (WRITE, "cobalt")])
def test_queries_inherit_key_tenant(
    tenant_api: TestClient, path: str, headers: dict[str, str], tenant: str
) -> None:
    response = tenant_api.post(path, json={"question": QUESTION}, headers=headers)
    assert response.status_code == 200
    if path.endswith("stream"):
        event = next(block for block in response.text.split("\n\n") if "event: metadata" in block)
        payload = json.loads(event.split("data: ", 1)[1])
        citations = payload["citations"]
    else:
        result = response.json()["result"]
        assert result["diagnostics"]["metadata_filters"]["tenant_id"] == tenant
        citations = result["citations"]
    assert citations
    assert all(item["document_id"].startswith(f"{tenant}:") for item in citations)


@pytest.mark.parametrize(
    "path,field", [("/query", "question"), ("/query/stream", "question"), ("/chat", "message")]
)
@pytest.mark.parametrize("tenant", ["acme", "default", " "])
def test_cross_tenant_queries_fail_before_returning_answers(
    tenant_api: TestClient, path: str, field: str, tenant: str
) -> None:
    response = tenant_api.post(path, json={field: QUESTION, "tenant_id": tenant}, headers=WRITE)
    assert response.status_code == 403
    assert response.headers["content-type"] == "application/json"
    assert "citations" not in response.text


def test_matching_tenant_preserves_other_filters_and_scopes_tools(tenant_api: TestClient) -> None:
    response = tenant_api.post(
        "/query",
        headers=READ,
        json={"question": "Summarize sentiment", "tenant_id": " ACME ", "min_rating": 2},
    )
    assert response.status_code == 200
    result = response.json()["result"]
    assert result["tool_run"]["tool_name"] == "sentiment_summary"
    assert result["diagnostics"]["metadata_filters"]["tenant_id"] == "acme"
    assert result["diagnostics"]["metadata_filters"]["min_rating"] == 2
    assert result["tool_run"]["output"]["total_records"] == 0


def test_global_admin_can_query_all_tenants(tenant_api: TestClient) -> None:
    response = tenant_api.post("/query", json={"question": QUESTION, "top_k": 10}, headers=ADMIN)
    assert response.status_code == 200
    assert {item["document_id"] for item in response.json()["result"]["citations"]} == {
        "acme:shared",
        "cobalt:shared",
        "shared",
    }


def test_default_tenant_binding_excludes_named_tenants(
    tenant_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FEEDBACK_AGENT_API_TENANT_BINDINGS",
        json.dumps({**BINDINGS, "tenant-read": "default"}),
    )
    client = TestClient(create_app())
    response = client.post("/query", json={"question": QUESTION, "tenant_id": None}, headers=READ)
    assert response.status_code == 200
    assert [item["document_id"] for item in response.json()["result"]["citations"]] == ["shared"]


def test_bound_admin_has_no_cross_tenant_resource_bypass(
    tenant_api: TestClient, resources: dict[str, dict[str, str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(
        "FEEDBACK_AGENT_API_TENANT_BINDINGS", json.dumps({**BINDINGS, "global-admin": "acme"})
    )
    client = TestClient(create_app())
    assert client.get(f"/reports/{resources['acme']['report']}", headers=ADMIN).status_code == 200
    assert client.get(f"/reports/{resources['cobalt']['report']}", headers=ADMIN).status_code == 404
    assert (
        client.post(
            "/query", json={"question": QUESTION, "tenant_id": "cobalt"}, headers=ADMIN
        ).status_code
        == 403
    )


@pytest.mark.parametrize(
    "path",
    [
        "/reports",
        "/answer-feedback",
        "/answer-feedback/analytics",
        "/answer-feedback/active-learning",
    ],
)
def test_resource_lists_reject_tenant_override(tenant_api: TestClient, path: str) -> None:
    assert tenant_api.get(path, params={"tenant_id": "cobalt"}, headers=READ).status_code == 403


def test_lists_analytics_and_states_only_include_bound_tenant(
    tenant_api: TestClient, resources: dict[str, dict[str, str]]
) -> None:
    for path in ("/reports", "/answer-feedback"):
        response = tenant_api.get(path, headers=READ)
        assert response.status_code == 200
        assert [item["tenant_id"] for item in response.json()] == ["acme"]
    analytics = tenant_api.get("/answer-feedback/analytics", headers=READ).json()
    assert analytics["total"] == 1
    assert [item["key"] for item in analytics["by_tenant"]] == ["acme"]
    queue = tenant_api.get("/answer-feedback/active-learning", headers=READ).json()
    assert [item["feedback_id"] for item in queue["items"]] == [resources["acme"]["feedback"]]
    states = tenant_api.get("/answer-feedback/active-learning/states", headers=READ).json()
    assert [item["feedback_id"] for item in states] == [resources["acme"]["feedback"]]
    assert (
        tenant_api.get(
            "/answer-feedback/active-learning/states", params={"status": "resolved"}, headers=READ
        ).json()
        == []
    )
    assert len(tenant_api.get("/reports", headers=ADMIN).json()) == 3


@pytest.mark.parametrize(
    "path,resource",
    [
        ("/reports/{id}", "report"),
        ("/reports/{id}/markdown", "report"),
        ("/answer-feedback/{id}", "feedback"),
        ("/answer-feedback/active-learning/{id}", "feedback"),
    ],
)
def test_foreign_resource_ids_are_indistinguishable_from_unknown_ids(
    tenant_api: TestClient, resources: dict[str, dict[str, str]], path: str, resource: str
) -> None:
    own = tenant_api.get(path.format(id=resources["acme"][resource]), headers=READ)
    foreign = tenant_api.get(path.format(id=resources["cobalt"][resource]), headers=READ)
    unknown = tenant_api.get(path.format(id="unknown"), headers=READ)
    assert own.status_code == 200
    assert foreign.status_code == unknown.status_code == 404
    assert foreign.json() == unknown.json()


@pytest.mark.parametrize("path", ["/reports", "/answer-feedback"])
def test_saved_resources_inherit_tenant_and_reject_explicit_other_tenants(
    tenant_api: TestClient, path: str
) -> None:
    payload = {"result": answer(tenant_api), "title": "Bound report", "rating": "useful"}
    created = tenant_api.post(path, json=payload, headers=WRITE)
    assert created.status_code == 201
    assert created.json()["tenant_id"] == "cobalt"
    for tenant in ("acme", "default", " "):
        denied = tenant_api.post(path, json={**payload, "tenant_id": tenant}, headers=WRITE)
        assert denied.status_code == 403
    assert len(tenant_api.get(path, headers=ADMIN).json()) == 1


def test_feedback_cannot_reference_foreign_reports(
    tenant_api: TestClient, resources: dict[str, dict[str, str]]
) -> None:
    payload = {"result": answer(tenant_api), "rating": "useful"}
    for report_id in (resources["acme"]["report"], "unknown"):
        response = tenant_api.post(
            "/answer-feedback", json={**payload, "report_id": report_id}, headers=WRITE
        )
        assert response.status_code == 404
        assert response.json() == {"detail": "report not found"}
    own = tenant_api.post(
        "/answer-feedback",
        json={**payload, "report_id": resources["cobalt"]["report"]},
        headers=WRITE,
    )
    assert own.status_code == 201
    assert len(tenant_api.get("/answer-feedback", headers=ADMIN).json()) == 4


def test_email_summary_cannot_select_foreign_reports(
    tenant_api: TestClient, resources: dict[str, dict[str, str]]
) -> None:
    payload = {"recipients": ["reviewer@example.test"]}
    own = tenant_api.post("/reports/email-summary", json=payload, headers=WRITE)
    assert own.status_code == 200
    assert own.json()["summary"]["report_ids"] == [resources["cobalt"]["report"]]
    for extra, code in (
        ({"tenant_id": "acme"}, 403),
        ({"report_ids": [resources["acme"]["report"]]}, 404),
    ):
        denied = tenant_api.post("/reports/email-summary", json={**payload, **extra}, headers=WRITE)
        assert denied.status_code == code
        assert "acme private report" not in denied.text


def test_review_state_cannot_be_changed_for_foreign_feedback(
    tenant_api: TestClient, resources: dict[str, dict[str, str]]
) -> None:
    foreign_path = f"/answer-feedback/active-learning/{resources['acme']['feedback']}"
    before = tenant_api.get(foreign_path, headers=ADMIN).json()
    denied = tenant_api.patch(foreign_path, json={"status": "resolved"}, headers=WRITE)
    assert denied.status_code == 404
    assert tenant_api.get(foreign_path, headers=ADMIN).json() == before
    own = tenant_api.patch(
        f"/answer-feedback/active-learning/{resources['cobalt']['feedback']}",
        json={"status": "resolved"},
        headers=WRITE,
    )
    assert own.status_code == 200


def test_chat_inherits_binding_and_followups_remain_scoped(tenant_api: TestClient) -> None:
    first = tenant_api.post("/chat", json={"message": QUESTION}, headers=WRITE)
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    second = tenant_api.post(
        "/chat", json={"message": QUESTION, "conversation_id": conversation_id}, headers=WRITE
    )
    assert second.status_code == 200
    assert second.json()["result"]["diagnostics"]["metadata_filters"]["tenant_id"] == "cobalt"
    history = tenant_api.get(f"/conversations/{conversation_id}", headers=WRITE)
    assert history.json()["tenant_id"] == "cobalt"
    assert tenant_api.get(f"/conversations/{conversation_id}", headers=READ).status_code == 404


@pytest.mark.parametrize("old_tenant", [None, "cobalt"])
def test_binding_a_key_hides_its_old_unscoped_or_foreign_conversations(
    tenant_api: TestClient, monkeypatch: pytest.MonkeyPatch, old_tenant: str | None
) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_API_TENANT_BINDINGS", "{}")
    legacy = TestClient(create_app())
    first = legacy.post("/chat", json={"message": QUESTION, "tenant_id": old_tenant}, headers=WRITE)
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    before = legacy.get(f"/conversations/{conversation_id}", headers=WRITE).json()
    monkeypatch.setenv(
        "FEEDBACK_AGENT_API_TENANT_BINDINGS", json.dumps({**BINDINGS, "tenant-write": "acme"})
    )
    rebound = TestClient(create_app())
    assert rebound.get(f"/conversations/{conversation_id}", headers=WRITE).status_code == 404
    denied = rebound.post(
        "/chat", json={"message": QUESTION, "conversation_id": conversation_id}, headers=WRITE
    )
    assert denied.status_code == 404
    assert legacy.get(f"/conversations/{conversation_id}", headers=WRITE).json() == before


def test_binding_a_key_preserves_existing_conversations_in_its_tenant(
    tenant_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_API_TENANT_BINDINGS", "{}")
    legacy = TestClient(create_app())
    first = legacy.post("/chat", json={"message": QUESTION, "tenant_id": "cobalt"}, headers=WRITE)
    assert first.status_code == 200
    conversation_id = first.json()["conversation_id"]
    assert tenant_api.get(f"/conversations/{conversation_id}", headers=WRITE).status_code == 200
    assert (
        tenant_api.post(
            "/chat", json={"message": QUESTION, "conversation_id": conversation_id}, headers=WRITE
        ).status_code
        == 200
    )


@pytest.mark.parametrize("bound_admin", [False, True])
def test_global_ingestion_is_unavailable_to_tenant_bound_keys(
    tenant_api: TestClient, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, bound_admin: bool
) -> None:
    client, headers = tenant_api, WRITE
    if bound_admin:
        monkeypatch.setenv(
            "FEEDBACK_AGENT_API_TENANT_BINDINGS", json.dumps({**BINDINGS, "global-admin": "acme"})
        )
        client, headers = TestClient(create_app()), ADMIN
    assert client.get("/ingestion/jobs/unknown", headers=headers).status_code == 403
    original = (tmp_path / "index.json").read_bytes()
    for path in ("/index", "/ingestion/jobs"):
        response = client.post(
            path, json={"input_path": str(tmp_path / "feedback.csv")}, headers=headers
        )
        assert response.status_code == 403
    assert (tmp_path / "index.json").read_bytes() == original
    assert not list((tmp_path / "jobs").glob("*.json"))


def test_unrestricted_admin_can_submit_and_read_ingestion_jobs(
    tenant_api: TestClient, tmp_path: Path
) -> None:
    response = tenant_api.post(
        "/ingestion/jobs", json={"input_path": str(tmp_path / "feedback.csv")}, headers=ADMIN
    )
    assert response.status_code == 202
    path = f"/ingestion/jobs/{response.json()['job_id']}"
    assert tenant_api.get(path, headers=ADMIN).status_code == 200
    assert tenant_api.get(path, headers=READ).status_code == 403


def test_misconfigured_bindings_fail_closed_but_health_remains_open(
    tenant_api: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FEEDBACK_AGENT_API_TENANT_BINDINGS", '{"tenant-read":"acme"}')
    misconfigured = TestClient(create_app())
    assert misconfigured.get("/health").status_code == 200
    assert misconfigured.get("/ready").status_code == 200
    response = misconfigured.post("/query", json={"question": QUESTION}, headers=ADMIN)
    assert response.status_code == 500
    assert "tenant-read" not in response.text
