"""Standard sequential LoRA baseline."""

from __future__ import annotations

from typing import Any, Sequence

from .base import EvaluationFn, StageResult, StrategyResult, TaskData, TrainFn, evaluate_tasks


class SequentialLoRA:
    name = "sequential_lora"

    def run(
        self,
        model: Any,
        datasets: TaskData,
        task_order: Sequence[str],
        seed: int,
        trainer: TrainFn,
        evaluator: EvaluationFn,
    ) -> StrategyResult:
        stages: list[StageResult] = []
        for stage_index, task in enumerate(task_order):
            training = dict(
                trainer(model, datasets[task]["train"], seed * 100 + stage_index, None)
            )
            evaluation = evaluate_tasks(model, task_order, datasets, evaluator)
            stages.append(StageResult(task, training, evaluation))
        return StrategyResult(self.name, seed, stages)

