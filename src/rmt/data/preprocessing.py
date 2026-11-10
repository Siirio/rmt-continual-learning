"""Deterministic preprocessing and dataset summaries."""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from .loader import DatasetRecord

LABELS = {
    1: ("World", "Sports", "Business", "Sci/Tech"),
    2: ("Negative", "Positive"),
    3: ("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"),
}
TASK_KEYS = {1: "ag_news", 2: "sst2", 3: "emotion"}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def dataset_summary(records: Iterable[DatasetRecord]) -> dict[str, object]:
    items = list(records)
    split_counts = Counter((TASK_KEYS[item.task], item.split) for item in items)
    class_counts = Counter(
        (TASK_KEYS[item.task], item.split, item.label_name) for item in items
    )
    return {
        "total_records": len(items),
        "tasks": sorted({TASK_KEYS[item.task] for item in items}),
        "split_counts": {"/".join(key): value for key, value in sorted(split_counts.items())},
        "class_counts": {"/".join(key): value for key, value in sorted(class_counts.items())},
        "mean_characters": round(sum(len(item.text) for item in items) / max(len(items), 1), 2),
    }

