"""Configuration loading and validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when a configuration file is missing required values."""


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping and reject empty or non-mapping documents."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    with config_path.open(encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, dict):
        raise ConfigurationError(f"Expected a mapping in {config_path}")
    return payload


@dataclass(frozen=True)
class TrainingSettings:
    """Validated settings shared by all training strategies."""

    learning_rate: float
    epochs_per_task: int
    effective_batch_size: int
    warmup_ratio: float
    gradient_clip_norm: float
    weight_decay: float = 0.0
    optimizer: str = "AdamW"

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "TrainingSettings":
        required = {
            "learning_rate",
            "epochs_per_task",
            "effective_batch_size",
            "warmup_ratio",
            "gradient_clip_norm",
        }
        missing = sorted(required - values.keys())
        if missing:
            raise ConfigurationError(f"Missing training settings: {', '.join(missing)}")
        settings = cls(**values)
        if settings.learning_rate <= 0 or settings.epochs_per_task <= 0:
            raise ConfigurationError("Learning rate and epochs must be positive")
        if not 0 <= settings.warmup_ratio < 1:
            raise ConfigurationError("warmup_ratio must be in [0, 1)")
        return settings

