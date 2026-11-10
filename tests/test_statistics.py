from __future__ import annotations

import pytest

from rmt.evaluation.statistics import (
    exact_mcnemar,
    holm_correction,
    paired_bootstrap_difference,
    paired_effects,
    resource_comparison,
)


def test_paired_bootstrap_difference_is_deterministic() -> None:
    method = [1, 1, 1, 0, 1, 0, 1, 1]
    baseline = [1, 0, 1, 0, 0, 0, 1, 0]
    result = paired_bootstrap_difference(method, baseline, resamples=2000, seed=4)
    assert result.estimate == pytest.approx(0.375)
    assert result.lower <= result.estimate <= result.upper


def test_exact_mcnemar_uses_discordant_pairs() -> None:
    baseline = [1] * 3 + [0] * 7
    method = [0] * 3 + [1] * 7
    result = exact_mcnemar(baseline, method)
    assert result["method_only_correct"] == 7
    assert result["baseline_only_correct"] == 3
    assert result["p_value"] == pytest.approx(0.34375)


def test_holm_correction_preserves_input_order() -> None:
    adjusted = holm_correction([0.04, 0.001, 0.02])
    assert adjusted[0]["adjusted_p_value"] == pytest.approx(0.04)
    assert adjusted[1]["adjusted_p_value"] == pytest.approx(0.003)
    assert adjusted[2]["adjusted_p_value"] == pytest.approx(0.04)


def test_effect_and_resource_differences_are_practical_units() -> None:
    effects = paired_effects([1, 1, 0, 0], [1, 0, 1, 1])
    resources = resource_comparison(120, 100)
    assert effects["absolute_improvement"] == pytest.approx(0.25)
    assert resources == {
        "absolute_difference": 20,
        "ratio_to_baseline": 1.2,
        "percent_change": pytest.approx(20),
    }

