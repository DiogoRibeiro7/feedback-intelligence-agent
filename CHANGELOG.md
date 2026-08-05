# Changelog

All notable changes to this project are documented here.

## Unreleased

### Added

- AWS Bedrock Runtime Converse API provider via `FEEDBACK_AGENT_LLM_PROVIDER=bedrock`.

## 0.2.0 - 2026-08-05

### Added

- Metadata filters for retrieval by segment, channel, rating, and creation date.
- Deterministic product terminology query expansion for retrieval.
- Deterministic judge reranking for cited evidence selection.
- Retrieval regression tests over known sample questions.
- OpenAI Responses API provider via `FEEDBACK_AGENT_LLM_PROVIDER=openai_responses`.
- Repository health files and release validation workflows.

### Changed

- Documented retrieval regression coverage and the OpenAI Responses provider.
- Updated roadmap status for completed retrieval and OpenAI provider milestones.
- Kept `openai` mapped to Chat Completions for OpenAI-compatible gateway support.

## 0.1.1 - 2026-08-05

### Added

- Zenodo and citation metadata for archival releases.

## 0.1.0 - 2026-08-05

### Added

- Initial Feedback Intelligence Agent release.
