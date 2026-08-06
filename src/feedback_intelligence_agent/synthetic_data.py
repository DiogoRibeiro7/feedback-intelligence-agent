"""Synthetic customer feedback dataset generation.

This module produces realistic, schema-compatible customer feedback records so
the repository is self-contained and can build larger demo datasets without any
external data or paid API. Generation uses a local, deterministically seeded
:class:`random.Random` instance (never the global ``random`` or NumPy state), so
the same seed and parameters always yield a byte-identical CSV.

Generated rows use the exact columns required by the feedback data contract
(:mod:`feedback_intelligence_agent.data_contracts`): ``feedback_id``,
``customer_segment``, ``channel``, ``rating``, ``text``, and ``created_at``, plus
optional ``tenant_id`` and ``sentiment`` columns. Ratings are derived from the
chosen sentiment so the dataset is internally consistent.
"""

from __future__ import annotations

import csv
import io
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

from feedback_intelligence_agent.schemas import FeedbackChannel

__all__ = [
    "DEFAULT_CUSTOMER_SEGMENTS",
    "DEFAULT_ISSUE_CATEGORIES",
    "DEFAULT_PRODUCT_AREAS",
    "DEFAULT_TENANT_IDS",
    "DEFAULT_SENTIMENT_DISTRIBUTION",
    "SyntheticDataConfig",
    "generate_feedback_rows",
    "write_feedback_csv",
]

#: Column order written to the generated CSV (matches the data contract plus
#: the optional ``sentiment`` column).
CSV_COLUMNS: tuple[str, ...] = (
    "tenant_id",
    "feedback_id",
    "customer_segment",
    "channel",
    "rating",
    "text",
    "created_at",
    "sentiment",
)

DEFAULT_PRODUCT_AREAS: tuple[str, ...] = (
    "onboarding",
    "dashboard",
    "integrations",
    "reporting",
    "pricing",
    "support",
    "performance",
    "documentation",
)

DEFAULT_CUSTOMER_SEGMENTS: tuple[str, ...] = ("enterprise", "mid_market", "startup")

DEFAULT_TENANT_IDS: tuple[str, ...] = ("default",)

DEFAULT_ISSUE_CATEGORIES: tuple[str, ...] = (
    "setup_friction",
    "missing_feature",
    "reliability",
    "value_for_money",
    "ease_of_use",
    "responsiveness",
)

#: Sentiment labels recognised by the data contract that this generator emits.
SENTIMENTS: tuple[str, ...] = ("positive", "neutral", "negative")

DEFAULT_SENTIMENT_DISTRIBUTION: dict[str, float] = {
    "positive": 0.4,
    "neutral": 0.25,
    "negative": 0.35,
}

#: Rating buckets each sentiment draws from, keeping rating and sentiment aligned.
_SENTIMENT_RATINGS: dict[str, tuple[int, ...]] = {
    "positive": (4, 5),
    "neutral": (3,),
    "negative": (1, 2),
}

# Sentence fragments keyed by sentiment then product area. Combined with issue
# categories and segments, these produce varied but grounded feedback text.
_PRODUCT_PHRASES: dict[str, dict[str, str]] = {
    "positive": {
        "onboarding": "Onboarding was smooth and the setup checklist made every step clear.",
        "dashboard": "The dashboard is genuinely useful and the alerts help the team react fast.",
        "integrations": "The integration worked on the first try and synced cleanly.",
        "reporting": "Reporting is reliable and exports finished quickly during month-end.",
        "pricing": "Pricing felt fair and the value was easy to justify internally.",
        "support": "Support responded fast and resolved the ticket on the first reply.",
        "performance": "Performance is snappy even when we load large customer lists.",
        "documentation": "The documentation is thorough and the examples saved us several hours.",
    },
    "neutral": {
        "onboarding": "Onboarding was acceptable, though a few setup steps could be clearer.",
        "dashboard": "The dashboard does the job but some views feel a little cluttered.",
        "integrations": "The integration works for now, but configuration took trial and error.",
        "reporting": "Reporting covers the basics, yet a few advanced filters are missing.",
        "pricing": "Pricing is reasonable, although the tiers are not always easy to compare.",
        "support": "Support was fine and answered our question after a short wait.",
        "performance": "Performance is generally okay, with occasional slowness on big queries.",
        "documentation": "The documentation is adequate but thin on advanced automations.",
    },
    "negative": {
        "onboarding": "Onboarding felt fragmented and we never knew who owned each setup step.",
        "dashboard": "The dashboard froze repeatedly and we lost work during a review.",
        "integrations": "The Salesforce integration broke after the last release.",
        "reporting": "Exports failed during month-end reporting and the error message was generic.",
        "pricing": "Pricing was hard to explain internally and the renewal became tense.",
        "support": "Support was slow and the workaround caused reporting delays for days.",
        "performance": "Performance degraded badly under load and pages timed out often.",
        "documentation": "Documentation for advanced automations is missing, so we kept guessing.",
    },
}

_ISSUE_CLAUSES: dict[str, str] = {
    "setup_friction": "Initial setup took longer than we expected.",
    "missing_feature": "We still need a capability the product does not yet offer.",
    "reliability": "Stability during critical workflows is our biggest concern.",
    "value_for_money": "We are weighing the cost against the value we actually use.",
    "ease_of_use": "Day-to-day usability shapes how the team feels about the tool.",
    "responsiveness": "How quickly issues get resolved matters a lot to us.",
}

_SEGMENT_CLAUSES: dict[str, str] = {
    "enterprise": "As an enterprise account, this affects several internal teams.",
    "mid_market": "For a mid-market team like ours, this has a direct impact on operations.",
    "startup": "As a small startup, our time and budget are limited.",
}


@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic feedback generation.

    Attributes:
        rows: Number of feedback records to generate (must be positive).
        seed: Seed for the local random generator; the same seed and parameters
            always produce a byte-identical CSV.
        product_areas: Product areas referenced by generated feedback text.
        tenant_ids: Tenant IDs assigned to records.
        customer_segments: Customer segments assigned to records.
        issue_categories: Issue categories woven into the feedback text.
        sentiment_distribution: Target proportion of each sentiment; values are
            normalised to sum to 1. Keys must be a subset of :data:`SENTIMENTS`.
        start_date: Earliest possible ``created_at`` timestamp.
        max_day_span: Records are spread across this many days after ``start_date``.
        include_sentiment_column: Whether to emit the optional ``sentiment`` column.
        include_tenant_column: Whether to emit the optional ``tenant_id`` column.
    """

    rows: int = 1000
    seed: int = 42
    product_areas: tuple[str, ...] = DEFAULT_PRODUCT_AREAS
    tenant_ids: tuple[str, ...] = DEFAULT_TENANT_IDS
    customer_segments: tuple[str, ...] = DEFAULT_CUSTOMER_SEGMENTS
    issue_categories: tuple[str, ...] = DEFAULT_ISSUE_CATEGORIES
    sentiment_distribution: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_SENTIMENT_DISTRIBUTION)
    )
    start_date: datetime = field(default_factory=lambda: datetime(2026, 1, 1, 8, 0, 0))
    max_day_span: int = 180
    include_sentiment_column: bool = True
    include_tenant_column: bool = False

    def __post_init__(self) -> None:
        """Validate configuration and normalise the sentiment distribution."""
        if self.rows <= 0:
            raise ValueError("rows must be a positive integer")
        if not self.product_areas:
            raise ValueError("product_areas must not be empty")
        if not self.tenant_ids:
            raise ValueError("tenant_ids must not be empty")
        if not self.customer_segments:
            raise ValueError("customer_segments must not be empty")
        if not self.issue_categories:
            raise ValueError("issue_categories must not be empty")
        if self.max_day_span < 0:
            raise ValueError("max_day_span must not be negative")

        unknown = set(self.sentiment_distribution) - set(SENTIMENTS)
        if unknown:
            raise ValueError(
                f"unknown sentiment(s) {sorted(unknown)}; valid options are {list(SENTIMENTS)}"
            )
        total = sum(self.sentiment_distribution.values())
        if total <= 0:
            raise ValueError("sentiment_distribution weights must sum to a positive value")
        self.sentiment_distribution = {
            sentiment: self.sentiment_distribution.get(sentiment, 0.0) / total
            for sentiment in SENTIMENTS
        }


def _build_text(
    rng: random.Random,
    *,
    sentiment: str,
    product_area: str,
    segment: str,
    issue_category: str,
) -> str:
    """Compose a grounded feedback sentence from the chosen attributes."""
    product_phrases = _PRODUCT_PHRASES[sentiment]
    # Fall back to a generic area if a custom product area has no phrase template.
    area_key = (
        product_area if product_area in product_phrases else rng.choice(DEFAULT_PRODUCT_AREAS)
    )
    parts = [
        product_phrases[area_key],
        _ISSUE_CLAUSES[issue_category],
        _SEGMENT_CLAUSES[segment],
    ]
    if product_area not in product_phrases:
        parts.append(f"This is specifically about {product_area}.")
    return " ".join(parts)


def generate_feedback_rows(config: SyntheticDataConfig) -> list[dict[str, str]]:
    """Generate synthetic feedback rows compatible with the data contract.

    Each returned mapping uses the columns in :data:`CSV_COLUMNS` (string
    values, as a CSV would carry them). Generation is fully deterministic for a
    given configuration.

    Args:
        config: Generation parameters.

    Returns:
        A list of ``config.rows`` feedback row mappings.
    """
    rng = random.Random(config.seed)
    channels = [channel.value for channel in FeedbackChannel]
    sentiments = list(SENTIMENTS)
    weights = [config.sentiment_distribution[sentiment] for sentiment in sentiments]

    rows: list[dict[str, str]] = []
    for index in range(config.rows):
        sentiment = rng.choices(sentiments, weights=weights, k=1)[0]
        rating = rng.choice(_SENTIMENT_RATINGS[sentiment])
        segment = rng.choice(list(config.customer_segments))
        tenant_id = rng.choice(list(config.tenant_ids)).strip().lower() or "default"
        channel = rng.choice(channels)
        product_area = rng.choice(list(config.product_areas))
        issue_category = rng.choice(list(config.issue_categories))
        text = _build_text(
            rng,
            sentiment=sentiment,
            product_area=product_area,
            segment=segment,
            issue_category=issue_category,
        )
        day_offset = rng.randint(0, config.max_day_span)
        minute_offset = rng.randint(0, 24 * 60 - 1)
        created_at = config.start_date + timedelta(days=day_offset, minutes=minute_offset)
        row = {
            "tenant_id": tenant_id,
            "feedback_id": f"syn-{index + 1:06d}",
            "customer_segment": segment,
            "channel": channel,
            "rating": str(rating),
            "text": text,
            "created_at": created_at.isoformat(),
            "sentiment": sentiment,
        }
        if not config.include_sentiment_column:
            row.pop("sentiment")
        if not config.include_tenant_column:
            row.pop("tenant_id")
        rows.append(row)
    return rows


def _rows_to_csv(rows: list[dict[str, str]], *, include_sentiment: bool) -> str:
    """Serialise rows to a CSV string with a stable column order.

    Uses ``\\n`` line terminators so output is byte-identical across platforms.
    """
    columns = [column for column in CSV_COLUMNS if column in rows[0]] if rows else []
    if not include_sentiment:
        columns = [column for column in columns if column != "sentiment"]
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def write_feedback_csv(config: SyntheticDataConfig, output: str | Path) -> Path:
    """Generate synthetic feedback and write it to a CSV file.

    Parent directories are created as needed. The file is written with UTF-8
    encoding and ``\\n`` newlines, so the same seed and parameters produce a
    byte-identical file on any platform.

    Args:
        config: Generation parameters.
        output: Destination CSV path.

    Returns:
        The path the CSV was written to.
    """
    rows = generate_feedback_rows(config)
    content = _rows_to_csv(rows, include_sentiment=config.include_sentiment_column)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8", newline="")
    return output_path
