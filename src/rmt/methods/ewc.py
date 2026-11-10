"""Online Elastic Weight Consolidation for trainable LoRA parameters."""

from __future__ import annotations

import random
import time
from typing import Any, Callable, Mapping, Sequence

from .base import EvaluationFn, StageResult, StrategyResult, TaskData, TrainFn, evaluate_tasks

LossFn = Callable[[Any, Sequence[Any]], Any]


class OnlineEWC:
    """Diagonal empirical Fisher with online accumulation and parameter anchors."""

    def __init__(self, lambda_: float, gamma: float = 1.0, loss_scale: float = 1024.0) -> None:
        if lambda_ < 0:
            raise ValueError("EWC lambda must be non-negative")
        if not 0 <= gamma <= 1:
            raise ValueError("EWC gamma must be in [0, 1]")
        self.lambda_ = float(lambda_)
        self.gamma = float(gamma)
        self.loss_scale = float(loss_scale)
        self.fisher: dict[str, Any] = {}
        self.anchor: dict[str, Any] = {}

    @staticmethod
    def trainable_parameters(model: Any) -> dict[str, Any]:
        parameters = {
            name: parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not parameters:
            raise ValueError("EWC requires at least one trainable parameter")
        return parameters

    def consolidate(
        self,
        model: Any,
        examples: Sequence[Any],
        loss_fn: LossFn,
        sample_count: int = 200,
        seed: int = 0,
    ) -> dict[str, Any]:
        """Estimate and normalize the empirical Fisher from per-example gradients."""
        import torch

        if not examples:
            raise ValueError("Cannot estimate Fisher information without examples")
        started = time.perf_counter()
        parameters = self.trainable_parameters(model)
        selected = random.Random(seed).sample(list(examples), min(sample_count, len(examples)))
        estimate = {
            name: torch.zeros_like(parameter, dtype=torch.float32)
            for name, parameter in parameters.items()
        }
        model.eval()
        for example in selected:
            model.zero_grad(set_to_none=True)
            scaled = loss_fn(model, [example]) * self.loss_scale
            scaled.backward()
            for name, parameter in parameters.items():
                if parameter.grad is not None:
                    gradient = parameter.grad.detach().float() / self.loss_scale
                    estimate[name].add_(gradient.square())
        model.zero_grad(set_to_none=True)
        for value in estimate.values():
            value.div_(len(selected))
        total = sum(value.sum() for value in estimate.values())
        elements = sum(value.numel() for value in estimate.values())
        mean = (total / max(elements, 1)).clamp_min(1e-30)
        estimate = {name: value / mean for name, value in estimate.items()}
        if self.fisher:
            self.fisher = {
                name: self.gamma * self.fisher[name] + estimate[name]
                for name in estimate
            }
        else:
            self.fisher = estimate
        self.anchor = {
            name: parameter.detach().clone().float() for name, parameter in parameters.items()
        }
        return {
            "fisher_samples": len(selected),
            "fisher_mean": self.fisher_mean,
            "fisher_time_seconds": time.perf_counter() - started,
            "ewc_state_bytes": self.state_bytes,
        }

    @property
    def fisher_mean(self) -> float:
        if not self.fisher:
            return 0.0
        total = sum(value.sum().item() for value in self.fisher.values())
        elements = sum(value.numel() for value in self.fisher.values())
        return total / elements

    @property
    def state_bytes(self) -> int:
        return sum(
            value.numel() * value.element_size()
            for mapping in (self.fisher, self.anchor)
            for value in mapping.values()
        )

    def penalty(self, model: Any) -> Any:
        import torch

        parameters = self.trainable_parameters(model)
        if not self.fisher:
            sample = next(iter(parameters.values()))
            return torch.zeros((), dtype=torch.float32, device=sample.device)
        total = None
        for name, parameter in parameters.items():
            fisher = self.fisher[name].to(parameter.device)
            anchor = self.anchor[name].to(parameter.device)
            term = (fisher * (parameter.float() - anchor).square()).sum()
            total = term if total is None else total + term
        return 0.5 * self.lambda_ * total

    def regularized_loss(self, model: Any, examples: Sequence[Any], loss_fn: LossFn) -> Any:
        return loss_fn(model, examples) + self.penalty(model)


class EWCStrategy:
    name = "ewc"

    def __init__(
        self,
        lambda_: float,
        loss_fn: LossFn,
        fisher_samples: int = 200,
        gamma: float = 1.0,
    ) -> None:
        self.lambda_ = lambda_
        self.loss_fn = loss_fn
        self.fisher_samples = fisher_samples
        self.gamma = gamma

    def run(
        self,
        model: Any,
        datasets: TaskData,
        task_order: Sequence[str],
        seed: int,
        trainer: TrainFn,
        evaluator: EvaluationFn,
    ) -> StrategyResult:
        ewc = OnlineEWC(self.lambda_, self.gamma)
        stages: list[StageResult] = []
        for stage_index, task in enumerate(task_order):
            stage_seed = seed * 100 + stage_index
            training = dict(trainer(model, datasets[task]["train"], stage_seed, ewc))
            state = ewc.consolidate(
                model,
                datasets[task]["train"],
                self.loss_fn,
                self.fisher_samples,
                stage_seed,
            )
            stages.append(
                StageResult(
                    task,
                    training,
                    evaluate_tasks(model, task_order, datasets, evaluator),
                    state,
                )
            )
        return StrategyResult(
            self.name,
            seed,
            stages,
            {"lambda": self.lambda_, "fisher_samples": self.fisher_samples},
        )


def select_ewc_lambda(results: Sequence[Mapping[str, float]]) -> float:
    """Select highest final validation ACC, then lower FGT, then lower lambda."""
    if not results:
        raise ValueError("No EWC calibration results were supplied")
    required = {"lambda", "validation_acc", "validation_fgt"}
    for result in results:
        if not required <= result.keys():
            raise ValueError("Calibration results require lambda, validation_acc, and validation_fgt")
    best = max(
        results,
        key=lambda item: (
            float(item["validation_acc"]),
            -float(item["validation_fgt"]),
            -float(item["lambda"]),
        ),
    )
    return float(best["lambda"])

