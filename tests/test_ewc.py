from __future__ import annotations

import pytest
import torch

from rmt.methods.ewc import OnlineEWC, select_ewc_lambda


def loss_fn(model, examples):
    losses = []
    for x, y in examples:
        prediction = model(torch.tensor([[x]], dtype=torch.float32))
        losses.append((prediction.squeeze() - y) ** 2)
    return torch.stack(losses).mean()


def test_empirical_fisher_is_normalized_and_penalty_uses_anchor() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    data = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]
    ewc = OnlineEWC(lambda_=10.0)
    state = ewc.consolidate(model, data, loss_fn, sample_count=3, seed=0)
    assert state["fisher_mean"] == pytest.approx(1.0)
    assert ewc.penalty(model).item() == pytest.approx(0.0)
    with torch.no_grad():
        model.weight.add_(0.5)
    assert ewc.penalty(model).item() == pytest.approx(1.25)
    assert ewc.state_bytes == 8


def test_online_fisher_accumulates_with_gamma_one() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    data = [(1.0, 2.0), (2.0, 4.0)]
    ewc = OnlineEWC(lambda_=1.0, gamma=1.0)
    ewc.consolidate(model, data, loss_fn, sample_count=2, seed=1)
    ewc.consolidate(model, data, loss_fn, sample_count=2, seed=1)
    assert ewc.fisher_mean == pytest.approx(2.0)


def test_lambda_selection_uses_validation_only_and_declared_ties() -> None:
    results = [
        {"lambda": 0.1, "validation_acc": 0.70, "validation_fgt": 0.10},
        {"lambda": 1.0, "validation_acc": 0.72, "validation_fgt": 0.08},
        {"lambda": 10.0, "validation_acc": 0.72, "validation_fgt": 0.05},
    ]
    assert select_ewc_lambda(results) == 10.0

