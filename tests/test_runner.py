from __future__ import annotations

import json

from rmt.training.runner import ExperimentRunner, create_run_schedule


def test_schedule_contains_all_nineteen_planned_runs() -> None:
    schedule = create_run_schedule()
    assert len(schedule) == 19
    assert sum(run.mode == "main" for run in schedule) == 12
    assert sum(run.mode == "calibration" for run in schedule) == 5
    assert sum(run.mode == "sensitivity" for run in schedule) == 2
    assert len({run.run_id for run in schedule}) == 19


def test_runner_persists_failures_and_resumes_completed_files(tmp_path) -> None:
    runner = ExperimentRunner(tmp_path)
    schedule = create_run_schedule()[:2]
    calls = []

    def executor(spec):
        calls.append(spec.run_id)
        if spec.seed == 1:
            raise RuntimeError("expected failure")
        return {"value": spec.seed}

    first = runner.run(schedule, executor)
    second = runner.run(schedule, executor, resume=True)
    assert [item["status"] for item in first] == ["completed", "failed"]
    assert calls == [run.run_id for run in schedule]
    assert second[0]["result"] == {"value": 0}
    failed = json.loads(runner.result_path(schedule[1]).read_text())
    assert failed["error_type"] == "RuntimeError"

