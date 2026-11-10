from __future__ import annotations

import pytest

from rmt.evaluation.metrics import accuracy, continual_learning_metrics, macro_f1


def test_classification_metrics_have_known_values() -> None:
    gold = [0, 0, 1, 1]
    predicted = [0, 1, 1, 1]
    assert accuracy(gold, predicted) == pytest.approx(0.75)
    assert macro_f1(gold, predicted) == pytest.approx((2 / 3 + 0.8) / 2)


def test_continual_learning_metrics_have_known_values() -> None:
    matrix = [
        [0.20, 0.50, 0.16],
        [0.80, 0.50, 0.16],
        [0.70, 0.90, 0.16],
        [0.60, 0.80, 0.75],
    ]
    metrics = continual_learning_metrics(matrix)
    assert metrics["ACC"] == pytest.approx((0.60 + 0.80 + 0.75) / 3)
    assert metrics["LA"] == pytest.approx((0.80 + 0.90 + 0.75) / 3)
    assert metrics["BWT"] == pytest.approx(-0.15)
    assert metrics["FGT"] == pytest.approx(0.15)

