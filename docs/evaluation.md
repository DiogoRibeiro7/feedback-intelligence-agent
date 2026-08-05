# Evaluation

RAG systems fail in two distinct places: retrieval can miss the evidence, or generation can ignore or contradict the evidence it was given. A single end-to-end "did the answer look right" check cannot tell those failures apart, so this project measures each stage separately and emits a typed, machine-readable report that can gate CI and track regressions over time.

## Running an evaluation

```bash
poetry run feedback-agent evaluate --queries examples/queries.jsonl --output evaluation_report.json
```

By default the report is written to `.artifacts/evaluation_report.json` (the `.artifacts/` folder is gitignored). The command prints the same JSON to stdout, so it can be piped into other tooling. Evaluation uses the deterministic local LLM provider, so two runs over the same index and dataset produce identical reports — a requirement for using the report as a CI regression gate.

## Dataset format

Evaluation cases live in a JSONL file. Each line is one case:

```json
{
  "question": "Why are enterprise customers unhappy with onboarding?",
  "expected_keywords": ["onboarding", "checklist"],
  "relevant_document_ids": ["fb-001", "fb-007", "fb-009"],
  "is_answerable": true
}
```

- `question` — the user query sent through the full pipeline.
- `expected_keywords` — terms a correct answer should mention (case-insensitive). Optional.
- `relevant_document_ids` — ground-truth feedback IDs the retriever should surface. The legacy field name `relevant_source_ids` is still accepted.
- `is_answerable` — `false` marks questions the corpus cannot answer; the correct behaviour is to refuse.

## Retrieval metrics

Computed per case and averaged over the answerable cases. Duplicate retrieved documents are collapsed before scoring so a retriever cannot inflate metrics by returning the same document twice.

| Metric | What it measures | Why it matters in production |
|---|---|---|
| `precision_at_k` | Share of the top-k results that are relevant | Low precision means the prompt is padded with noise, which raises token cost and invites hallucination. |
| `recall_at_k` | Share of the relevant documents found in the top-k | Low recall means the answer is built on incomplete evidence no matter how good the LLM is. |
| `mean_reciprocal_rank` | How early the first relevant document appears | LLMs weight early context more heavily; relevant evidence buried at rank k is often ignored. |
| `context_hit_rate` | Share of cases where at least one relevant document was retrieved | The hard floor: when this misses, generation cannot possibly be grounded. |

## Answer metrics

Computed over all cases (refusal correctness is the only metric that uses unanswerable cases).

| Metric | What it measures | Why it matters in production |
|---|---|---|
| `keyword_coverage` | Share of expected keywords present in the answer | A cheap, deterministic proxy for topical correctness that catches off-topic answers without an LLM judge. |
| `groundedness` | Share of answer sentences whose content words appear in the retrieved context | Detects answers that drift away from the evidence — the precursor to hallucination. |
| `citation_alignment` | Share of answerable cases where the cited sources intersect the ground-truth documents | Citations users can verify are only useful if they point at the right evidence. |
| `refusal_correctness` | Refusing unanswerable questions and answering answerable ones | Confidently answering questions the corpus cannot support is the most damaging RAG failure mode; this metric makes abstention behaviour measurable. |
| `evidence_overlap` | Average sentence-level support score from the hallucination checker | Tracks how much of each generated answer is backed by retrieved evidence. |
| `hallucination_rate` | Share of answers marked high risk by overlap or judge checks | A direct regression signal for unsupported answer drift. |
| `judge_supported_rate` | Share of answers marked `supported` by the optional LLM judge | Useful when semantic judging is enabled; remains `0.0` for deterministic-only runs. |

## Report structure

The report is a typed Pydantic model (`EvaluationReport` in `evaluation.py`):

- `top_k`, `total_cases` — run configuration.
- `retrieval` — aggregate retrieval metrics over answerable cases.
- `answers` — aggregate answer-quality metrics.
- `cases` — per-case breakdown (retrieved IDs, per-metric scores, refusal flags) for debugging individual regressions.

## How this is used in practice

- **CI gate**: fail the build when `context_hit_rate`, `groundedness`, or
  `evidence_overlap` drops below a threshold after a retrieval or prompt change.
- **A/B comparison**: run the same dataset against two index or prompt configurations and diff the reports.
- **Drift monitoring**: re-run the suite on a schedule as the corpus grows; falling recall usually signals the evaluation set or the chunking strategy needs to evolve.
