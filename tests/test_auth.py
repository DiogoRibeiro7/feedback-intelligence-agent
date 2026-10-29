from __future__ import annotations

import pytest

from feedback_intelligence_agent import auth
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


def test_caller_identity_uses_a_stable_stretched_derivation() -> None:
    principal = resolve_api_principal(
        "shared", auth_enabled=True, reader_key=None, writer_key="shared", admin_key=None
    )
    # This vector also protects persisted history paths from accidental changes.
    assert principal.subject_id == (
        "51d6224539d64eb3ad5d900612677bf4732884f1378b416795c785e50af14274"
    )


def test_rejected_keys_and_disabled_auth_do_not_derive_identities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_derivation(*args: object, **kwargs: object) -> bytes:
        pytest.fail("unauthenticated input reached identity derivation")

    monkeypatch.setattr(auth, "pbkdf2_hmac", unexpected_derivation)
    for key in (None, "", "unknown-key-not-in-cache"):
        with pytest.raises(ApiAuthenticationError):
            resolve_api_principal(
                key, auth_enabled=True, reader_key=None, writer_key="configured", admin_key=None
            )
    assert (
        resolve_api_principal(
            "ignored", auth_enabled=False, reader_key=None, writer_key=None, admin_key=None
        ).subject_id
        is None
    )


def test_cached_identity_avoids_rederivation_but_does_not_authorize_a_revoked_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    auth._subject_id.cache_clear()
    try:
        first = resolve_api_principal(
            "cached-key",
            auth_enabled=True,
            reader_key=None,
            writer_key="cached-key",
            admin_key=None,
        )

        def unexpected_derivation(*args: object, **kwargs: object) -> bytes:
            pytest.fail("a cached identity was derived again")

        monkeypatch.setattr(auth, "pbkdf2_hmac", unexpected_derivation)
        second = resolve_api_principal(
            " cached-key ",
            auth_enabled=True,
            reader_key="cached-key",
            writer_key=None,
            admin_key=None,
        )
        assert second.subject_id == first.subject_id
        assert second.role == ApiRole.reader
        with pytest.raises(ApiAuthenticationError, match="invalid API key"):
            resolve_api_principal(
                "cached-key",
                auth_enabled=True,
                reader_key=None,
                writer_key="replacement-key",
                admin_key=None,
            )
    finally:
        auth._subject_id.cache_clear()
