"""Collision-free source identifiers and migration of legacy chunk identities."""

from __future__ import annotations

from urllib.parse import quote

from feedback_intelligence_agent.schemas import DocumentChunk


class SourceIdMigrationError(ValueError):
    """Raised when legacy chunk identities cannot be migrated unambiguously."""


def encode_source_id(tenant_id: str, feedback_id: str) -> str:
    """Encode each identity component, reserving a colon for the tenant separator.

    The default tenant has no prefix. Percent signs are escaped along with
    colons, so a literal escape sequence cannot impersonate an encoded ID.
    Ordinary IDs such as ``fb-1`` and ``acme:fb-1`` remain unchanged.
    """
    encoded_feedback = quote(feedback_id, safe="")
    if tenant_id == "default":
        return encoded_feedback
    return f"{quote(tenant_id, safe='')}:{encoded_feedback}"


def migrate_legacy_chunks(chunks: list[DocumentChunk]) -> list[DocumentChunk]:
    """Rebuild version-1 source and chunk IDs without changing text or metadata.

    Metadata distinguishes records whose old identifiers collided. Older
    default-tenant chunks without identity metadata use their raw source ID
    as the feedback ID. Chunk positions are retained, as is list order, so
    vectors remain aligned without recomputing embeddings.

    Raises:
        SourceIdMigrationError: If identities disagree, chunk positions cannot
            be recovered, or migration would produce duplicate chunk IDs.
    """
    migrated: list[DocumentChunk] = []
    seen_ids: set[str] = set()
    for chunk in chunks:
        tenant = chunk.metadata.get("tenant_id", "default")
        if not isinstance(tenant, str) or not tenant or tenant != tenant.strip().lower():
            raise SourceIdMigrationError("Invalid legacy tenant metadata; rebuild the index")
        feedback_id = chunk.metadata.get("feedback_id")
        if "feedback_id" not in chunk.metadata:
            if tenant == "default":
                feedback_id = chunk.source_id
            elif chunk.source_id.startswith(f"{tenant}:"):
                feedback_id = chunk.source_id[len(tenant) + 1 :]
        if not isinstance(feedback_id, str) or not feedback_id:
            raise SourceIdMigrationError("Missing legacy feedback identity; rebuild the index")
        legacy_source = feedback_id if tenant == "default" else f"{tenant}:{feedback_id}"
        if legacy_source != chunk.source_id:
            raise SourceIdMigrationError(
                "Legacy source ID disagrees with metadata; rebuild the index"
            )
        source_id = encode_source_id(tenant, feedback_id)
        chunk_id = chunk.chunk_id
        if source_id != chunk.source_id:
            prefix = f"{chunk.source_id}::chunk-"
            position = chunk_id.removeprefix(prefix)
            if (
                not chunk_id.startswith(prefix)
                or not position.isascii()
                or not position.isdecimal()
            ):
                raise SourceIdMigrationError(
                    "Cannot recover legacy chunk position; rebuild the index"
                )
            chunk_id = f"{source_id}::chunk-{position}"
        if chunk_id in seen_ids:
            raise SourceIdMigrationError("Duplicate migrated chunk IDs; rebuild the index")
        seen_ids.add(chunk_id)
        migrated.append(chunk.model_copy(update={"source_id": source_id, "chunk_id": chunk_id}))
    return migrated
