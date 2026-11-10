from __future__ import annotations

from collections import Counter

from rmt.methods.experience_replay import ExperienceReplay, ReplayMemory


def examples(task: str, count: int = 1000, classes: int = 4):
    return [
        {"task": task, "label": index % classes, "text": f"{task} text {index}"}
        for index in range(count)
    ]


def test_replay_memory_selects_ten_percent_and_balances_classes() -> None:
    memory = ReplayMemory(0.10)
    selected = memory.add_task("ag_news", examples("ag_news"), seed=7)
    counts = Counter(item["label"] for item in selected)
    assert len(selected) == 100
    assert counts == Counter({0: 25, 1: 25, 2: 25, 3: 25})
    assert memory.storage_bytes > 0


def test_replay_selection_is_deterministic() -> None:
    first = ReplayMemory(0.10).add_task("a", examples("a"), seed=11)
    second = ReplayMemory(0.10).add_task("a", examples("a"), seed=11)
    assert first == second


def test_replay_is_added_only_after_task_completion() -> None:
    data = {
        "a": {"train": examples("a", 20, 2), "test": []},
        "b": {"train": examples("b", 20, 2), "test": []},
        "c": {"train": examples("c", 20, 2), "test": []},
    }
    trained_sizes = []

    def trainer(model, batch, seed, regularizer):
        trained_sizes.append(len(batch))
        return {"examples": len(batch)}

    def evaluator(model, task, batch):
        return {"accuracy": 0.0}

    result = ExperienceReplay(0.10).run({}, data, ["a", "b", "c"], 0, trainer, evaluator)
    assert trained_sizes == [20, 22, 24]
    assert result.stages[-1].method_state["replay_memory"]["examples"] == 6

