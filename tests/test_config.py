from __future__ import annotations

from pathlib import Path

from rmt.config import TrainingSettings, load_yaml
from rmt.models.lora import LoRASettings


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_training_configuration() -> None:
    experiment = load_yaml(ROOT / "configs/experiment.yaml")
    settings = TrainingSettings.from_mapping(experiment["training"])
    assert settings.learning_rate == 2e-4
    assert settings.effective_batch_size == 16
    assert experiment["task_order"] == ["ag_news", "sst2", "emotion"]


def test_frozen_lora_configuration() -> None:
    model = load_yaml(ROOT / "configs/model.yaml")
    settings = LoRASettings.from_mapping(model["lora"])
    assert settings.rank == 8
    assert settings.alpha == 16
    assert settings.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj")

