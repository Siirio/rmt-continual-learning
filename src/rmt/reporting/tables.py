"""Result aggregation and Markdown table formatting."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np


def aggregate_method_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    output = []
    metric_names = ("ACC", "macro_f1", "LA", "BWT", "FGT")
    for method, items in grouped.items():
        summary: dict[str, Any] = {"method": method, "seeds": len(items)}
        for metric in metric_names:
            values = [float(item[metric]) for item in items if item.get(metric) is not None]
            summary[f"{metric}_mean"] = float(np.mean(values)) if values else None
            summary[f"{metric}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None
        output.append(summary)
    return output


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    rendered.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(rendered)


def format_mean_sd(mean: float | None, sd: float | None, scale: float = 100.0) -> str:
    if mean is None or sd is None:
        return "N/A"
    return f"{mean * scale:.1f} +/- {sd * scale:.1f}"

