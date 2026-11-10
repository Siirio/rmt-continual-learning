"""Classification and continual-learning metrics."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def accuracy(gold: Sequence[int], predicted: Sequence[int]) -> float:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    return float(np.mean(np.asarray(gold) == np.asarray(predicted))) if gold else 0.0


def per_class_f1(gold: Sequence[int], predicted: Sequence[int]) -> dict[int, float]:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    labels = sorted(set(gold) | set(predicted))
    scores = {}
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        scores[label] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return scores


def macro_f1(gold: Sequence[int], predicted: Sequence[int]) -> float:
    scores = per_class_f1(gold, predicted)
    return float(np.mean(list(scores.values()))) if scores else 0.0


def confusion_matrix(
    gold: Sequence[int], predicted: Sequence[int], labels: Sequence[int] | None = None
) -> np.ndarray:
    if len(gold) != len(predicted):
        raise ValueError("gold and predicted must have the same length")
    ordered = list(labels) if labels is not None else sorted(set(gold) | set(predicted))
    positions = {label: index for index, label in enumerate(ordered)}
    matrix = np.zeros((len(ordered), len(ordered)), dtype=int)
    for actual, guess in zip(gold, predicted):
        matrix[positions[actual], positions[guess]] += 1
    return matrix


def continual_learning_metrics(matrix: Sequence[Sequence[float]]) -> dict[str, float]:
    """Calculate ACC, LA, BWT, and FGT from a stage-by-task accuracy matrix.

    A matrix may include the zero-shot row. If it has T + 1 rows for T tasks,
    the first row is removed before the standard definitions are applied.
    """
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2:
        raise ValueError("Accuracy matrix must be two-dimensional")
    task_count = values.shape[1]
    if values.shape[0] == task_count + 1:
        values = values[1:]
    if values.shape != (task_count, task_count):
        raise ValueError("Expected a T x T matrix, with one optional zero-shot row")
    learned = np.diag(values)
    final = values[-1]
    earlier = range(task_count - 1)
    bwt = np.mean([final[index] - learned[index] for index in earlier])
    forgetting = np.mean(
        [np.max(values[index : task_count - 1, index]) - final[index] for index in earlier]
    )
    return {
        "ACC": float(np.mean(final)),
        "LA": float(np.mean(learned)),
        "BWT": float(bwt),
        "FGT": float(forgetting),
    }


def task_accuracies(matrix: Sequence[Sequence[float]]) -> list[float]:
    values = np.asarray(matrix, dtype=float)
    return [float(value) for value in values[-1]]

