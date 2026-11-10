"""Shared strategy contracts and serializable results."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, MutableMapping, Sequence

Example = Any
TaskData = Mapping[str, Mapping[str, Sequence[Example]]]
TrainFn = Callable[[Any, Sequence[Example], int, Any | None], Mapping[str, Any]]
EvaluationFn = Callable[[Any, str, Sequence[Example]], Mapping[str, Any]]


@dataclass
class StageResult:
    stage: str
    training: dict[str, Any]
    evaluation: dict[str, dict[str, Any]]
    method_state: dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyResult:
    method: str
    seed: int
    stages: list[StageResult]
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_tasks(
    model: Any,
    task_order: Sequence[str],
    datasets: TaskData,
    evaluator: EvaluationFn,
    split: str = "test",
) -> dict[str, dict[str, Any]]:
    return {
        task: dict(evaluator(model, task, datasets[task][split])) for task in task_order
    }


def combine_training_sets(
    task_order: Sequence[str], datasets: TaskData, split: str = "train"
) -> list[Example]:
    combined: list[Example] = []
    for task in task_order:
        combined.extend(datasets[task][split])
    return combined

