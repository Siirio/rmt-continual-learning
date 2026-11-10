"""Class-stratified experience replay and sequential integration."""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from typing import Any, Sequence

from .base import EvaluationFn, StageResult, StrategyResult, TaskData, TrainFn, evaluate_tasks


def _value(example: Any, name: str) -> Any:
    if isinstance(example, dict):
        return example[name]
    return getattr(example, name)


def _serializable(example: Any) -> Any:
    if is_dataclass(example):
        return asdict(example)
    if isinstance(example, dict):
        return example
    return {"label": _value(example, "label"), "text": str(_value(example, "text"))}


class ReplayMemory:
    """A task-aware, class-balanced replay memory."""

    def __init__(self, fraction: float = 0.10) -> None:
        if not 0 < fraction <= 1:
            raise ValueError("Replay fraction must be in (0, 1]")
        self.fraction = fraction
        self._by_task: dict[str, list[Any]] = {}

    def add_task(self, task: str, examples: Sequence[Any], seed: int) -> list[Any]:
        if task in self._by_task:
            raise ValueError(f"Replay task {task!r} has already been added")
        if not examples:
            self._by_task[task] = []
            return []
        groups: dict[int, list[Any]] = defaultdict(list)
        for example in examples:
            groups[int(_value(example, "label"))].append(example)
        rng = random.Random(seed)
        labels = sorted(groups)
        target = max(1, round(len(examples) * self.fraction))
        base, remainder = divmod(target, len(labels))
        selected: list[Any] = []
        leftovers: list[Any] = []
        for position, label in enumerate(labels):
            candidates = list(groups[label])
            rng.shuffle(candidates)
            allocation = base + (1 if position < remainder else 0)
            selected.extend(candidates[:allocation])
            leftovers.extend(candidates[allocation:])
        if len(selected) < target:
            rng.shuffle(leftovers)
            selected.extend(leftovers[: target - len(selected)])
        rng.shuffle(selected)
        self._by_task[task] = selected[:target]
        return list(self._by_task[task])

    def sample_all(self, seed: int | None = None) -> list[Any]:
        items = [example for task in self._by_task.values() for example in task]
        if seed is not None:
            random.Random(seed).shuffle(items)
        return items

    @property
    def example_count(self) -> int:
        return sum(len(items) for items in self._by_task.values())

    @property
    def storage_bytes(self) -> int:
        payload = [_serializable(item) for item in self.sample_all()]
        return len(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))

    def inspect(self) -> dict[str, Any]:
        return {
            "fraction": self.fraction,
            "examples": self.example_count,
            "storage_bytes": self.storage_bytes,
            "tasks": {
                task: {
                    "examples": len(items),
                    "class_counts": {
                        str(label): sum(int(_value(item, "label")) == label for item in items)
                        for label in sorted({int(_value(item, "label")) for item in items})
                    },
                }
                for task, items in self._by_task.items()
            },
        }


class ExperienceReplay:
    name = "experience_replay"

    def __init__(self, buffer_fraction: float = 0.10) -> None:
        self.buffer_fraction = buffer_fraction

    def run(
        self,
        model: Any,
        datasets: TaskData,
        task_order: Sequence[str],
        seed: int,
        trainer: TrainFn,
        evaluator: EvaluationFn,
    ) -> StrategyResult:
        memory = ReplayMemory(self.buffer_fraction)
        stages: list[StageResult] = []
        for stage_index, task in enumerate(task_order):
            replay = memory.sample_all(seed * 100 + stage_index)
            training_examples = list(datasets[task]["train"]) + replay
            training = dict(
                trainer(model, training_examples, seed * 100 + stage_index, None)
            )
            memory.add_task(task, datasets[task]["train"], seed * 100 + stage_index)
            stages.append(
                StageResult(
                    task,
                    training,
                    evaluate_tasks(model, task_order, datasets, evaluator),
                    {
                        "replay_examples_used": len(replay),
                        "replay_memory": memory.inspect(),
                    },
                )
            )
        return StrategyResult(
            self.name,
            seed,
            stages,
            {"buffer_fraction": self.buffer_fraction},
        )

