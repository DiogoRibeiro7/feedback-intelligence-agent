from __future__ import annotations

import pytest

from feedback_intelligence_agent import auth
from feedback_intelligence_agent.auth import (
    ApiAuthConfigurationError,
    ApiAuthenticationError,
    ApiPermissionError,
    ApiPrincipal,
    ApiRole,
    require_api_role,
    resolve_api_principal,
)


def tenant_principal(key: str, bindings: dict[str, str], *, enabled: bool = True) -> ApiPrincipal:
    return resolve_api_principal(
        key,
        auth_enabled=enabled,
        reader_key="read",
        writer_key="write",
        admin_key="admin",
        tenant_bindings=bindings,
    )


def test_tenant_bindings_follow_normalized_credentials_and_keep_admin_global() -> None:
    bindings = {" read ": " Acme ", "write": "Cobalt"}
    reader = tenant_principal(" read ", bindings)
    assert reader.tenant_id == "acme"
    assert reader.permits_tenant(" ACME ")
    assert not reader.permits_tenant("cobalt")
    assert not reader.permits_tenant(None)
    assert tenant_principal("write", bindings).tenant_id == "cobalt"
    assert tenant_principal("admin", bindings).permits_tenant(None)


def test_admin_can_be_bound_without_changing_its_role_or_identity() -> None:
    global_admin = tenant_principal("admin", {})
    bound_admin = tenant_principal("admin", {"read": "a", "write": "b", "admin": "a"})
    assert bound_admin.role == ApiRole.admin
    assert bound_admin.subject_id == global_admin.subject_id
    assert bound_admin.tenant_id == "a"
    assert not bound_admin.permits_tenant("b")


@pytest.mark.parametrize(
    "bindings",
    [
        {"read": "acme"},
        {"read": "acme", "write": " "},
        {"read": "acme", "write": "cobalt", "unknown-secret": "acme"},
        {"read": "acme", " read ": "other", "write": "cobalt"},
    ],
)
def test_invalid_tenant_policy_fails_closed_without_disclosing_keys(
    bindings: dict[str, str],
) -> None:
    with pytest.raises(ApiAuthConfigurationError) as failure:
        tenant_principal("admin", bindings)
    assert not any(key in str(failure.value) for key in ("unknown-secret", "cobalt"))


def test_tenant_bindings_require_authentication() -> None:
    with pytest.raises(ApiAuthConfigurationError, match="require API authentication"):
        tenant_principal("read", {"read": "acme"}, enabled=False)


def test_tenant_binding_changes_do_not_reuse_cached_authorization() -> None:
    first = tenant_principal("write", {"read": "acme", "write": "cobalt"})
    second = tenant_principal("write", {"read": "acme", "write": "acme"})
    assert first.subject_id == second.subject_id
    assert second.tenant_id == "acme"
    assert not second.permits_tenant("cobalt")


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
