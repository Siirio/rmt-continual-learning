"""Joint multi-task LoRA reference condition."""

from __future__ import annotations

from typing import Any, Sequence

from .base import (
    EvaluationFn,
    StageResult,
    StrategyResult,
    TaskData,
    TrainFn,
    combine_training_sets,
    evaluate_tasks,
)


class JointTraining:
    name = "joint_training"

    def run(
        self,
        model: Any,
        datasets: TaskData,
        task_order: Sequence[str],
        seed: int,
        trainer: TrainFn,
        evaluator: EvaluationFn,
    ) -> StrategyResult:
        combined = combine_training_sets(task_order, datasets)
        training = dict(trainer(model, combined, seed, None))
        evaluation = evaluate_tasks(model, task_order, datasets, evaluator)
        stage = StageResult(
            stage="joint",
            training=training,
            evaluation=evaluation,
            method_state={"training_examples": len(combined)},
        )
        return StrategyResult(self.name, seed, [stage], {"task_order": list(task_order)})

