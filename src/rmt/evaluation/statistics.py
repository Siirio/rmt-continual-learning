"""Paired statistical analysis for predictions and run summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float = 0.95


def mean_and_sample_sd(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        raise ValueError("At least one value is required")
    return float(array.mean()), float(array.std(ddof=1)) if array.size > 1 else 0.0


def mean_confidence_interval(
    values: Sequence[float], confidence: float = 0.95
) -> ConfidenceInterval:
    """Return a Student-t interval when SciPy is present, otherwise a normal interval."""
    mean, standard_deviation = mean_and_sample_sd(values)
    count = len(values)
    if count < 2:
        return ConfidenceInterval(mean, mean, mean, confidence)
    try:
        from scipy.stats import t

        critical = float(t.ppf((1 + confidence) / 2, df=count - 1))
    except ImportError:
        critical = 1.959963984540054
    half_width = critical * standard_deviation / math.sqrt(count)
    return ConfidenceInterval(mean, mean - half_width, mean + half_width, confidence)


def paired_bootstrap_difference(
    method_scores: Sequence[float],
    baseline_scores: Sequence[float],
    resamples: int = 10000,
    confidence: float = 0.95,
    seed: int = 2026,
) -> ConfidenceInterval:
    """Bootstrap the paired mean difference over identical test items."""
    method = np.asarray(method_scores, dtype=float)
    baseline = np.asarray(baseline_scores, dtype=float)
    if method.shape != baseline.shape or method.ndim != 1 or method.size == 0:
        raise ValueError("Paired score vectors must be non-empty and have equal shape")
    rng = np.random.default_rng(seed)
    differences = method - baseline
    indices = rng.integers(0, differences.size, size=(resamples, differences.size))
    estimates = differences[indices].mean(axis=1)
    alpha = 1 - confidence
    lower, upper = np.quantile(estimates, [alpha / 2, 1 - alpha / 2])
    return ConfidenceInterval(float(differences.mean()), float(lower), float(upper), confidence)


def mcnemar_discordant_pairs(
    baseline_correct: Sequence[int | bool], method_correct: Sequence[int | bool]
) -> tuple[int, int]:
    if len(baseline_correct) != len(method_correct):
        raise ValueError("Paired correctness vectors must have equal length")
    method_only = sum(not bool(base) and bool(method) for base, method in zip(baseline_correct, method_correct))
    baseline_only = sum(bool(base) and not bool(method) for base, method in zip(baseline_correct, method_correct))
    return method_only, baseline_only


def exact_mcnemar(
    baseline_correct: Sequence[int | bool], method_correct: Sequence[int | bool]
) -> dict[str, float | int]:
    """Two-sided exact McNemar test using the binomial distribution."""
    method_only, baseline_only = mcnemar_discordant_pairs(baseline_correct, method_correct)
    discordant = method_only + baseline_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(method_only, baseline_only)
        probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant)
        p_value = min(1.0, 2 * probability)
    return {
        "method_only_correct": method_only,
        "baseline_only_correct": baseline_only,
        "discordant_pairs": discordant,
        "p_value": p_value,
    }


def holm_correction(p_values: Sequence[float], alpha: float = 0.05) -> list[dict[str, float | bool]]:
    """Return Holm-adjusted p-values and step-down rejection decisions."""
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        candidate = (count - rank) * float(p_values[index])
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    rejected = [False] * count
    still_rejecting = True
    for rank, index in enumerate(order):
        threshold = alpha / (count - rank)
        still_rejecting = still_rejecting and p_values[index] <= threshold
        rejected[index] = still_rejecting
    return [
        {"p_value": float(value), "adjusted_p_value": adjusted[index], "reject": rejected[index]}
        for index, value in enumerate(p_values)
    ]


def paired_effects(
    baseline_correct: Sequence[int | bool], method_correct: Sequence[int | bool]
) -> dict[str, float]:
    method_only, baseline_only = mcnemar_discordant_pairs(baseline_correct, method_correct)
    count = len(baseline_correct)
    baseline_accuracy = float(np.mean(baseline_correct))
    method_accuracy = float(np.mean(method_correct))
    return {
        "absolute_improvement": method_accuracy - baseline_accuracy,
        "relative_error_reduction": (
            (method_accuracy - baseline_accuracy) / (1 - baseline_accuracy)
            if baseline_accuracy < 1
            else 0.0
        ),
        "matched_odds_ratio": (method_only + 0.5) / (baseline_only + 0.5),
        "net_discordant_fraction": (method_only - baseline_only) / count if count else 0.0,
    }


def resource_comparison(method: float, baseline: float) -> dict[str, float]:
    if baseline == 0:
        raise ValueError("Resource baseline must be non-zero")
    return {
        "absolute_difference": method - baseline,
        "ratio_to_baseline": method / baseline,
        "percent_change": (method / baseline - 1) * 100,
    }

