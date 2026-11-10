from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")

from rmt.reporting.example_data import generate_example_package
from rmt.reporting.figures import generate_all_figures


def test_all_required_figures_are_generated_as_png_and_svg(tmp_path) -> None:
    results = tmp_path / "results"
    figures = tmp_path / "figures"
    generate_example_package(results)
    paths = generate_all_figures(results, figures)
    assert len(paths) == 18
    assert len(list(figures.glob("*.png"))) == 9
    assert len(list(figures.glob("*.svg"))) == 9
    assert all(path.stat().st_size > 1000 for path in paths)
    assert (figures / "accuracy_comparison.png").read_bytes().startswith(b"\x89PNG")

