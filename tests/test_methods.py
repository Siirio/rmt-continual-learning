from __future__ import annotations

from rmt.methods.joint_training import JointTraining
from rmt.methods.sequential_lora import SequentialLoRA


def datasets():
    return {
        "a": {"train": ["a1", "a2"], "test": ["at"]},
        "b": {"train": ["b1"], "test": ["bt"]},
        "c": {"train": ["c1", "c2"], "test": ["ct"]},
    }


def trainer(model, examples, seed, regularizer):
    model["seen"].extend(examples)
    return {"examples": len(examples), "seed": seed}


def evaluator(model, task, examples):
    return {"accuracy": len(model["seen"]) / 10, "count": len(examples)}


def test_sequential_lora_updates_one_adapter_in_order() -> None:
    model = {"seen": []}
    result = SequentialLoRA().run(model, datasets(), ["a", "b", "c"], 2, trainer, evaluator)
    assert [stage.stage for stage in result.stages] == ["a", "b", "c"]
    assert [stage.training["seed"] for stage in result.stages] == [200, 201, 202]
    assert model["seen"] == ["a1", "a2", "b1", "c1", "c2"]


def test_joint_training_uses_all_examples_once() -> None:
    model = {"seen": []}
    result = JointTraining().run(model, datasets(), ["a", "b", "c"], 1, trainer, evaluator)
    assert result.stages[0].method_state["training_examples"] == 5
    assert result.stages[0].training["examples"] == 5

