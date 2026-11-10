"""PEFT LoRA configuration for the frozen protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LoRASettings:
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "k_proj", "v_proj", "o_proj")
    bias: str = "none"
    task_type: str = "CAUSAL_LM"

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "LoRASettings":
        return cls(
            rank=int(values.get("rank", 8)),
            alpha=int(values.get("alpha", 16)),
            dropout=float(values.get("dropout", 0.05)),
            target_modules=tuple(values.get("target_modules", cls.target_modules)),
            bias=str(values.get("bias", "none")),
            task_type=str(values.get("task_type", "CAUSAL_LM")),
        )


def build_lora_config(settings: LoRASettings) -> Any:
    try:
        from peft import LoraConfig
    except ImportError as error:
        raise RuntimeError("Install the experiment dependencies to configure LoRA") from error
    return LoraConfig(
        r=settings.rank,
        lora_alpha=settings.alpha,
        lora_dropout=settings.dropout,
        target_modules=list(settings.target_modules),
        bias=settings.bias,
        task_type=settings.task_type,
    )


def attach_lora(model: Any, settings: LoRASettings | None = None) -> Any:
    """Attach the single trainable LoRA adapter used by all conditions."""
    try:
        from peft import get_peft_model
    except ImportError as error:
        raise RuntimeError("Install the experiment dependencies to attach LoRA") from error
    return get_peft_model(model, build_lora_config(settings or LoRASettings()))


def trainable_parameter_count(model: Any) -> tuple[int, int]:
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return trainable, total

