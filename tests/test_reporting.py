from __future__ import annotations

import csv
import json

from rmt.reporting.example_data import generate_example_package
from rmt.reporting.report_generator import generate_results_markdown


def test_example_package_is_deterministic_and_internally_consistent(tmp_path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    generate_example_package(first, seed=12)
    generate_example_package(second, seed=12)
    assert (first / "per_seed_metrics.csv").read_bytes() == (second / "per_seed_metrics.csv").read_bytes()
    rows = list(csv.DictReader((first / "per_seed_metrics.csv").open()))
    assert len(rows) == 12
    matrices = json.loads((first / "accuracy_matrices.json").read_text())
    assert len(matrices["sequential_lora"]["0"]) == 4
    manifest = json.loads((first / "manifest.json").read_text())
    assert manifest["measured_model_runs"] is False
    assert len(list((first / "runs").glob("*.json"))) == 12


def test_results_markdown_uses_generated_table(tmp_path) -> None:
    generate_example_package(tmp_path)
    destination = generate_results_markdown(tmp_path, tmp_path / "RESULTS.md")
    content = destination.read_text()
    assert "Seq-LoRA" in content
    assert "deterministic example package" in content

