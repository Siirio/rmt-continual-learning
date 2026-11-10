"""Generate a compact Markdown results report from result tables."""

from __future__ import annotations

import csv
from pathlib import Path

from .tables import markdown_table


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def generate_results_markdown(results_dir: str | Path, output_path: str | Path) -> Path:
    root = Path(results_dir)
    comparison = _read_csv(root / "main_comparison.csv")
    rows = [
        [
            row["method_display"],
            row["ACC_mean_percent"],
            row["macro_f1_mean_percent"],
            row["FGT_mean_pp"],
            row["training_time_mean_minutes"],
        ]
        for row in comparison
    ]
    content = "\n".join(
        [
            "# Example Results Reference",
            "",
            "The values in this report come from the deterministic example package.",
            "",
            markdown_table(
                ["Method", "ACC (%)", "Macro-F1 (%)", "FGT (pp)", "Train time (min)"],
                rows,
            ),
            "",
        ]
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(content, encoding="utf-8")
    return destination

