"""Shared metadata matching for retrieval, dataset tools, and final evidence checks."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone

from feedback_intelligence_agent.schemas import MetadataFilters


def matches_metadata_filters(metadata: Mapping[str, object], filters: MetadataFilters) -> bool:
    """Match all requested fields using the same semantics for every backend.

    Tenant and segment comparisons ignore case. Missing tenant metadata denotes
    the default tenant; missing or invalid ratings retain the legacy fallback
    of 5. Date bounds are inclusive, and naive dates are interpreted as UTC.
    Missing or invalid dates cannot satisfy a date filter.
    """
    if (
        filters.tenant_id is not None
        and str(metadata.get("tenant_id", "default")).lower() != filters.tenant_id.lower()
    ):
        return False
    if (
        filters.customer_segment is not None
        and str(metadata.get("customer_segment", "")).lower() != filters.customer_segment.lower()
    ):
        return False
    if filters.channel is not None and str(metadata.get("channel", "")) != filters.channel:
        return False
    rating = _metadata_rating(metadata.get("rating", 5))
    if filters.min_rating is not None and rating < filters.min_rating:
        return False
    if filters.max_rating is not None and rating > filters.max_rating:
        return False
    if filters.created_after is None and filters.created_before is None:
        return True
    created_at = _metadata_datetime(metadata.get("created_at"))
    if filters.created_after is not None and (
        created_at is None or created_at < _comparable_datetime(filters.created_after)
    ):
        return False
    return filters.created_before is None or (
        created_at is not None and created_at <= _comparable_datetime(filters.created_before)
    )


def _metadata_rating(value: object) -> int:
    """Read a numeric rating or retain the existing fallback value."""
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            pass
    return 5


def _metadata_datetime(value: object) -> datetime | None:
    """Parse stored ISO timestamps, including trailing Z on Python 3.10."""
    if isinstance(value, str) and value:
        try:
            value = datetime.fromisoformat(
                value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
            )
        except ValueError:
            return None
    return _comparable_datetime(value) if isinstance(value, datetime) else None


def _comparable_datetime(value: datetime) -> datetime:
    """Compare timestamps in UTC without mixing aware and naive datetimes."""
    if value.tzinfo is None:
        return value
    return value.astimezone(timezone.utc).replace(tzinfo=None)
