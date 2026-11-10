"""Shared training and evaluation engine for causal label scoring."""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from rmt.config import TrainingSettings
from rmt.data.prompts import encode_training_record, pad_batch
from rmt.models.qwen import ModelBundle, predict_label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def macro_f1(gold: Sequence[int], predicted: Sequence[int], labels: Sequence[int]) -> float:
    scores = []
    for label in labels:
        true_positive = sum(g == label and p == label for g, p in zip(gold, predicted))
        false_positive = sum(g != label and p == label for g, p in zip(gold, predicted))
        false_negative = sum(g == label and p != label for g, p in zip(gold, predicted))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0
        scores.append(2 * precision * recall / (precision + recall) if precision + recall else 0)
    return float(np.mean(scores)) if scores else 0.0


@dataclass
class TrainingEngine:
    bundle: ModelBundle
    settings: TrainingSettings
    maximum_length: int = 160
    text_token_limit: int = 96

    def _encoded(self, examples: Sequence[Any]) -> list[dict[str, Any]]:
        return [
            encode_training_record(
                example,
                self.bundle.tokenizer,
                self.text_token_limit,
                self.maximum_length,
            )
            for example in examples
        ]

    def loss(self, model: Any, examples: Sequence[Any]) -> Any:
        encoded = self._encoded(examples)
        pad_id = self.bundle.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer pad token is not configured")
        batch = pad_batch(encoded, pad_id)
        batch = {key: value.to(self.bundle.device) for key, value in batch.items()}
        return model(**batch).loss

    def train(
        self,
        model: Any,
        examples: Sequence[Any],
        seed: int,
        regularizer: Any | None = None,
    ) -> dict[str, Any]:
        import torch

        seed_everything(seed)
        if not examples:
            raise ValueError("Training requires at least one example")
        model.train()
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.settings.learning_rate,
            weight_decay=self.settings.weight_decay,
        )
        batch_size = self.settings.effective_batch_size
        batches_per_epoch = math.ceil(len(examples) / batch_size)
        total_steps = batches_per_epoch * self.settings.epochs_per_task
        warmup_steps = round(total_steps * self.settings.warmup_ratio)

        def learning_rate_factor(step: int) -> float:
            if warmup_steps and step < warmup_steps:
                return (step + 1) / warmup_steps
            remaining = max(total_steps - warmup_steps, 1)
            return max(0.0, (total_steps - step) / remaining)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, learning_rate_factor)
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        losses: list[float] = []
        order = list(range(len(examples)))
        rng = random.Random(seed)
        for _ in range(self.settings.epochs_per_task):
            rng.shuffle(order)
            for start in range(0, len(order), batch_size):
                batch_examples = [examples[index] for index in order[start : start + batch_size]]
                optimizer.zero_grad(set_to_none=True)
                base_loss = self.loss(model, batch_examples)
                loss = base_loss + regularizer.penalty(model) if regularizer is not None else base_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, self.settings.gradient_clip_norm)
                optimizer.step()
                scheduler.step()
                losses.append(float(loss.detach().cpu().item()))
        peak = (
            int(torch.cuda.max_memory_allocated() / (1024**2))
            if torch.cuda.is_available()
            else None
        )
        return {
            "examples": len(examples),
            "optimizer_steps": len(losses),
            "training_time_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mb": peak,
            "initial_loss": losses[0],
            "final_loss": losses[-1],
        }

    def evaluate(self, model: Any, task: str, examples: Sequence[Any]) -> dict[str, Any]:
        predicted = [predict_label(self.bundle, example) for example in examples]
        gold = [int(example.label) for example in examples]
        labels = sorted(set(gold))
        correct = [int(g == p) for g, p in zip(gold, predicted)]
        return {
            "task": task,
            "count": len(examples),
            "accuracy": sum(correct) / len(correct) if correct else 0.0,
            "macro_f1": macro_f1(gold, predicted, labels),
            "predictions": predicted,
            "gold": gold,
            "correct": correct,
            "predicted_distribution": dict(Counter(predicted)),
        }

