"""Optional API-key authorization for the FastAPI service."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel


class ApiRole(str, Enum):
    """API roles ordered from least to most privileged."""

    reader = "reader"
    writer = "writer"
    admin = "admin"


class ApiPrincipal(BaseModel):
    """Resolved API caller identity."""

    role: ApiRole
    authenticated: bool


class ApiAuthError(ValueError):
    """Base class for API authorization failures."""


class ApiAuthConfigurationError(ApiAuthError):
    """Raised when auth is enabled without any configured keys."""


class ApiAuthenticationError(ApiAuthError):
    """Raised when the request does not provide a valid API key."""


class ApiPermissionError(ApiAuthError):
    """Raised when a valid API key lacks the required role."""


_ROLE_RANK = {
    ApiRole.reader: 1,
    ApiRole.writer: 2,
    ApiRole.admin: 3,
}


def resolve_api_principal(
    api_key: str | None,
    *,
    auth_enabled: bool,
    reader_key: str | None,
    writer_key: str | None,
    admin_key: str | None,
) -> ApiPrincipal:
    """Resolve an optional API key into a principal.

    When auth is disabled, local callers receive an unauthenticated admin
    principal so route-level dependencies become no-ops.
    """
    if not auth_enabled:
        return ApiPrincipal(role=ApiRole.admin, authenticated=False)

    keys = _configured_keys(
        reader_key=reader_key,
        writer_key=writer_key,
        admin_key=admin_key,
    )
    if not keys:
        raise ApiAuthConfigurationError("API auth is enabled but no API keys are configured")

    if api_key is None or not api_key.strip():
        raise ApiAuthenticationError("missing API key")

    role = keys.get(api_key.strip())
    if role is None:
        raise ApiAuthenticationError("invalid API key")
    return ApiPrincipal(role=role, authenticated=True)


def require_api_role(
    required_role: ApiRole,
    api_key: str | None,
    *,
    auth_enabled: bool,
    reader_key: str | None,
    writer_key: str | None,
    admin_key: str | None,
) -> ApiPrincipal:
    """Resolve a principal and require the requested role."""
    principal = resolve_api_principal(
        api_key,
        auth_enabled=auth_enabled,
        reader_key=reader_key,
        writer_key=writer_key,
        admin_key=admin_key,
    )
    if _ROLE_RANK[principal.role] < _ROLE_RANK[required_role]:
        raise ApiPermissionError(
            f"API key role {principal.role.value!r} cannot access "
            f"{required_role.value!r} endpoints"
        )
    return principal


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
