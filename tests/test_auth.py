from __future__ import annotations

import pytest

from feedback_intelligence_agent.auth import (
    ApiAuthConfigurationError,
    ApiAuthenticationError,
    ApiPermissionError,
    ApiRole,
    require_api_role,
    resolve_api_principal,
)


def test_resolve_api_principal_returns_admin_when_auth_is_disabled() -> None:
    principal = resolve_api_principal(
        None,
        auth_enabled=False,
        reader_key=None,
        writer_key=None,
        admin_key=None,
    )

    assert principal.role == ApiRole.admin
    assert principal.authenticated is False
    assert principal.subject_id is None


def test_resolve_api_principal_requires_configured_keys_when_enabled() -> None:
    with pytest.raises(ApiAuthConfigurationError, match="no API keys"):
        resolve_api_principal(
            "key",
            auth_enabled=True,
            reader_key=None,
            writer_key=None,
            admin_key=None,
        )


def test_resolve_api_principal_rejects_missing_or_unknown_keys() -> None:
    with pytest.raises(ApiAuthenticationError, match="missing API key"):
        resolve_api_principal(
            None,
            auth_enabled=True,
            reader_key="read",
            writer_key=None,
            admin_key=None,
        )
    with pytest.raises(ApiAuthenticationError, match="invalid API key"):
        resolve_api_principal(
            "wrong",
            auth_enabled=True,
            reader_key="read",
            writer_key=None,
            admin_key=None,
        )


def test_require_api_role_allows_higher_privilege_keys() -> None:
    principal = require_api_role(
        ApiRole.reader,
        "write",
        auth_enabled=True,
        reader_key="read",
        writer_key="write",
        admin_key="admin",
    )

    assert principal.role == ApiRole.writer
    assert principal.authenticated is True


def test_require_api_role_rejects_insufficient_privileges() -> None:
    with pytest.raises(ApiPermissionError, match="cannot access"):
        require_api_role(
            ApiRole.writer,
            "read",
            auth_enabled=True,
            reader_key="read",
            writer_key="write",
            admin_key="admin",
        )


def test_duplicate_api_keys_resolve_to_highest_configured_role() -> None:
    principal = require_api_role(
        ApiRole.admin,
        "shared",
        auth_enabled=True,
        reader_key="shared",
        writer_key="shared",
        admin_key="shared",
    )

    assert principal.role == ApiRole.admin


def test_caller_identity_is_key_specific_and_stable_across_role_changes() -> None:
    writer = resolve_api_principal(
        " shared ", auth_enabled=True, reader_key=None, writer_key="shared", admin_key=None
    )
    promoted = resolve_api_principal(
        "shared", auth_enabled=True, reader_key=None, writer_key=None, admin_key="shared"
    )
    other = resolve_api_principal(
        "other", auth_enabled=True, reader_key=None, writer_key="other", admin_key=None
    )
    assert writer.subject_id == promoted.subject_id
    assert writer.subject_id and writer.subject_id != "shared"
    assert other.subject_id != writer.subject_id
