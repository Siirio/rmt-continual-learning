"""Create and execute the complete 19-run experiment schedule."""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    method: str
    seed: int
    mode: str = "main"
    parameters: dict[str, Any] = field(default_factory=dict)


def create_run_schedule() -> list[RunSpec]:
    schedule: list[RunSpec] = []
    for method in ("sequential_lora", "experience_replay", "ewc", "joint_training"):
        for seed in (0, 1, 2):
            schedule.append(RunSpec(f"{method}_seed{seed}", method, seed))
    for value in (0.01, 0.1, 1.0, 10.0, 100.0):
        token = str(value).replace(".", "p")
        schedule.append(
            RunSpec(
                f"ewc_calibration_lambda_{token}",
                "ewc",
                1000,
                "calibration",
                {"lambda": value},
            )
        )
    for fraction in (0.05, 0.20):
        schedule.append(
            RunSpec(
                f"replay_sensitivity_{round(fraction * 100)}pct",
                "experience_replay",
                1000,
                "sensitivity",
                {"buffer_fraction": fraction},
            )
        )
    assert len(schedule) == 19
    return schedule


Executor = Callable[[RunSpec], dict[str, Any]]


class ExperimentRunner:
    def __init__(self, output_dir: str | Path) -> None:
        self.output_dir = Path(output_dir)

    def result_path(self, run: RunSpec) -> Path:
        return self.output_dir / f"{run.run_id}.json"

    def _metadata(self) -> dict[str, Any]:
        return {
            "python": sys.version,
            "platform": platform.platform(),
            "created_utc": datetime.now(timezone.utc).isoformat(),
        }

    def run(
        self,
        schedule: Iterable[RunSpec],
        executor: Executor,
        resume: bool = True,
        fail_fast: bool = False,
    ) -> list[dict[str, Any]]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for specification in schedule:
            path = self.result_path(specification)
            if resume and path.exists():
                results.append(json.loads(path.read_text(encoding="utf-8")))
                continue
            started = time.perf_counter()
            payload: dict[str, Any] = {
                "specification": asdict(specification),
                "environment": self._metadata(),
            }
            try:
                payload["result"] = executor(specification)
                payload["status"] = "completed"
            except Exception as error:
                payload.update(
                    status="failed",
                    error_type=type(error).__name__,
                    error_message=str(error),
                    traceback=traceback.format_exc(),
                )
                if fail_fast:
                    self._write(path, payload, started)
                    raise
            self._write(path, payload, started)
            results.append(payload)
        return results

    @staticmethod
    def _write(path: Path, payload: dict[str, Any], started: float) -> None:
        payload["runner_elapsed_seconds"] = time.perf_counter() - started
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)

