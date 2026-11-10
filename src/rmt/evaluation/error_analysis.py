"""Confusion summaries and manually coded error categories."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence

from .metrics import confusion_matrix


@dataclass(frozen=True)
class CodedError:
    record_id: str
    task: str
    gold_label: str
    predicted_label: str
    category: str
    description: str = ""


def summarize_error_categories(errors: Iterable[CodedError]) -> list[dict[str, float | int | str]]:
    items = list(errors)
    counts = Counter(item.category for item in items)
    total = len(items)
    return [
        {
            "category": category,
            "count": count,
            "percent": 100 * count / total if total else 0.0,
        }
        for category, count in counts.most_common()
    ]


def confusion_summary(
    gold: Sequence[int], predicted: Sequence[int], labels: Sequence[str]
) -> dict[str, object]:
    matrix = confusion_matrix(gold, predicted, range(len(labels)))
    return {"labels": list(labels), "matrix": matrix.tolist(), "total": int(matrix.sum())}

