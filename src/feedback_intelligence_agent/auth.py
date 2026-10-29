"""Optional API-key authorization for the FastAPI service."""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from functools import lru_cache
from hashlib import pbkdf2_hmac

from pydantic import BaseModel


class ApiRole(str, Enum):
    """API roles ordered from least to most privileged."""

    reader = "reader"
    writer = "writer"
    admin = "admin"


class ApiPrincipal(BaseModel):
    """Resolved role and opaque API-key identity, independent of role assignment."""

    role: ApiRole
    authenticated: bool
    subject_id: str | None = None
    tenant_id: str | None = None

    def permits_tenant(self, tenant_id: str | None) -> bool:
        """Allow global principals or resources belonging to the bound tenant."""
        return self.tenant_id is None or (
            tenant_id is not None and tenant_id.strip().lower() == self.tenant_id
        )


class ApiAuthError(ValueError):
    """Base class for API authorization failures."""


class ApiAuthConfigurationError(ApiAuthError):
    """Raised when API credentials or tenant bindings are misconfigured."""


class ApiAuthenticationError(ApiAuthError):
    """Raised when the request does not provide a valid API key."""


class ApiPermissionError(ApiAuthError):
    """Raised when a valid API key lacks the required role."""


_ROLE_RANK = {
    ApiRole.reader: 1,
    ApiRole.writer: 2,
    ApiRole.admin: 3,
}


@lru_cache(maxsize=3)
def _subject_id(validated_key: str) -> str:
    """Stretch a validated key into a stable opaque conversation identity."""
    # Fixed domain separation preserves identities across restarts and role changes.
    # This identifies a namespace; it is not a stored password verifier.
    # Cache only the three configured keys, after authentication, to avoid paying
    # the derivation cost on every request or caching arbitrary rejected input.
    return pbkdf2_hmac(
        "sha256",
        validated_key.encode("utf-8"),
        b"feedback-intelligence-agent:api-subject:v1",
        600_000,
    ).hex()


def resolve_api_principal(
    api_key: str | None,
    *,
    auth_enabled: bool,
    reader_key: str | None,
    writer_key: str | None,
    admin_key: str | None,
    tenant_bindings: Mapping[str, str] | None = None,
) -> ApiPrincipal:
    """Resolve an optional API key into a principal.

    When auth is disabled, local callers receive an unauthenticated admin
    principal so route-level dependencies become no-ops.
    Authenticated identities are derived from the normalized API key; key
    rotation changes identity, while changing its role preserves identity.
    """
    if not auth_enabled:
        if tenant_bindings:
            raise ApiAuthConfigurationError("Tenant bindings require API authentication")
        return ApiPrincipal(role=ApiRole.admin, authenticated=False)

    keys = _configured_keys(
        reader_key=reader_key,
        writer_key=writer_key,
        admin_key=admin_key,
    )
    if not keys:
        raise ApiAuthConfigurationError("API auth is enabled but no API keys are configured")
    tenants = _configured_tenants(keys, tenant_bindings)

    if api_key is None or not api_key.strip():
        raise ApiAuthenticationError("missing API key")

    role = keys.get(api_key.strip())
    if role is None:
        raise ApiAuthenticationError("invalid API key")
    return ApiPrincipal(
        role=role,
        authenticated=True,
        subject_id=_subject_id(api_key.strip()),
        tenant_id=tenants.get(api_key.strip()),
    )


def require_api_role(
    required_role: ApiRole,
    api_key: str | None,
    *,
    auth_enabled: bool,
    reader_key: str | None,
    writer_key: str | None,
    admin_key: str | None,
    tenant_bindings: Mapping[str, str] | None = None,
) -> ApiPrincipal:
    """Resolve a principal and require the requested role."""
    principal = resolve_api_principal(
        api_key,
        auth_enabled=auth_enabled,
        reader_key=reader_key,
        writer_key=writer_key,
        admin_key=admin_key,
        tenant_bindings=tenant_bindings,
    )
    if _ROLE_RANK[principal.role] < _ROLE_RANK[required_role]:
        raise ApiPermissionError(
            f"API key role {principal.role.value!r} cannot access {required_role.value!r} endpoints"
        )
    return principal


def _configured_tenants(
    keys: Mapping[str, ApiRole], bindings: Mapping[str, str] | None
) -> dict[str, str]:
    """Validate an optional policy without including credential values in errors."""
    if not bindings:
        return {}
    tenants: dict[str, str] = {}
    for raw_key, raw_tenant in bindings.items():
        key, tenant = raw_key.strip(), raw_tenant.strip().lower()
        if key not in keys or not tenant:
            raise ApiAuthConfigurationError(
                "Tenant bindings must reference configured API keys and nonblank tenants"
            )
        if key in tenants and tenants[key] != tenant:
            raise ApiAuthConfigurationError("Conflicting tenant bindings for one API key")
        tenants[key] = tenant
    if any(role != ApiRole.admin and key not in tenants for key, role in keys.items()):
        raise ApiAuthConfigurationError("Every non-admin API key requires a tenant binding")
    return tenants


def _configured_keys(
    *,
    reader_key: str | None,
    writer_key: str | None,
    admin_key: str | None,
) -> dict[str, ApiRole]:
    keys: dict[str, ApiRole] = {}
    for key, role in (
        (reader_key, ApiRole.reader),
        (writer_key, ApiRole.writer),
        (admin_key, ApiRole.admin),
    ):
        if key is not None and key.strip():
            current = keys.get(key.strip())
            if current is None or _ROLE_RANK[role] > _ROLE_RANK[current]:
                keys[key.strip()] = role
    return keys
