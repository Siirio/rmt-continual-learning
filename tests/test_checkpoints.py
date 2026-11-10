from __future__ import annotations

import torch

from rmt.training.checkpoints import CheckpointManager


def test_checkpoint_round_trip(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    manager = CheckpointManager(tmp_path)
    original = model.weight.detach().clone()
    manager.save("run", model, {"stage": 2})
    with torch.no_grad():
        model.weight.add_(10)
    metadata = manager.load("run", model)
    assert torch.equal(model.weight, original)
    assert metadata == {"stage": 2}
    assert manager.exists("run")

