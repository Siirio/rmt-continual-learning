"""Prediction records and JSON Lines persistence."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class PredictionRecord:
    run_id: str
    stage: str
    task: str
    record_id: str
    gold: int
    predicted: int
    correct: bool


def write_predictions(path: str | Path, records: Iterable[PredictionRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


def read_predictions(path: str | Path) -> list[PredictionRecord]:
    with Path(path).open(encoding="utf-8") as stream:
        return [PredictionRecord(**json.loads(line)) for line in stream if line.strip()]

