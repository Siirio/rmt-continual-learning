"""Generate a coherent, deterministic example results package."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from rmt.evaluation.metrics import confusion_matrix, continual_learning_metrics, macro_f1
from rmt.evaluation.statistics import exact_mcnemar, holm_correction, paired_bootstrap_difference
from rmt.reporting.tables import aggregate_method_metrics

METHODS = ("sequential_lora", "experience_replay", "ewc", "joint_training")
DISPLAY = {
    "sequential_lora": "Seq-LoRA",
    "experience_replay": "ER (10%)",
    "ewc": "Online EWC",
    "joint_training": "Joint",
}
TASKS = ("ag_news", "sst2", "emotion")
CLASS_COUNTS = {"ag_news": 4, "sst2": 2, "emotion": 6}
BASE_MATRICES = {
    "sequential_lora": np.array(
        [[0.25, 0.50, 0.167], [0.84, 0.50, 0.167], [0.68, 0.90, 0.167], [0.55, 0.72, 0.78]]
    ),
    "experience_replay": np.array(
        [[0.25, 0.50, 0.167], [0.84, 0.50, 0.167], [0.77, 0.90, 0.167], [0.72, 0.83, 0.78]]
    ),
    "ewc": np.array(
        [[0.25, 0.50, 0.167], [0.83, 0.50, 0.167], [0.74, 0.88, 0.167], [0.67, 0.80, 0.77]]
    ),
}
JOINT_FINAL = np.array([0.82, 0.89, 0.79])
RESOURCE_BASE = {
    "sequential_lora": (27.0, 6900, 0),
    "experience_replay": (31.8, 7600, 94000),
    "ewc": (35.4, 8400, 8700000),
    "joint_training": (29.3, 7100, 0),
}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_payload(
    rng: np.random.Generator, accuracy: float, class_count: int, count: int = 800
) -> dict[str, list[int]]:
    gold = np.arange(count) % class_count
    rng.shuffle(gold)
    correct_count = int(round(accuracy * count))
    correct = np.zeros(count, dtype=bool)
    correct[rng.choice(count, size=correct_count, replace=False)] = True
    predicted = gold.copy()
    errors = np.flatnonzero(~correct)
    predicted[errors] = (gold[errors] + rng.integers(1, class_count, size=len(errors))) % class_count
    return {
        "gold": gold.astype(int).tolist(),
        "predicted": predicted.astype(int).tolist(),
        "correct": correct.astype(int).tolist(),
    }


def _seed_matrix(method: str, seed: int, rng: np.random.Generator) -> np.ndarray:
    base = BASE_MATRICES[method].copy()
    jitter = rng.normal(0, 0.006, size=base.shape)
    jitter[0] = 0
    return np.clip(base + jitter, 0, 1)


def generate_example_package(output_dir: str | Path, seed: int = 20261110) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    per_seed: list[dict[str, Any]] = []
    matrices: dict[str, Any] = {}
    prediction_index: dict[tuple[str, int, str], list[int]] = {}
    resources: list[dict[str, Any]] = []

    for method in METHODS:
        matrices[method] = {}
        for run_seed in (0, 1, 2):
            run_rng = np.random.default_rng(seed + run_seed * 101 + METHODS.index(method) * 1009)
            if method == "joint_training":
                final = np.clip(JOINT_FINAL + run_rng.normal(0, 0.006, size=3), 0, 1)
                matrix = np.vstack([[0.25, 0.50, 0.167], final, final, final])
                metrics = {"ACC": float(final.mean()), "LA": None, "BWT": None, "FGT": None}
            else:
                matrix = _seed_matrix(method, run_seed, run_rng)
                metrics = continual_learning_metrics(matrix)
                final = matrix[-1]
            matrices[method][str(run_seed)] = matrix.round(6).tolist()
            task_f1 = []
            task_predictions = {}
            for task_index, task in enumerate(TASKS):
                payload = _prediction_payload(run_rng, float(final[task_index]), CLASS_COUNTS[task])
                task_predictions[task] = payload
                prediction_index[(method, run_seed, task)] = payload["correct"]
                task_f1.append(macro_f1(payload["gold"], payload["predicted"]))
            row = {
                "method": method,
                "seed": run_seed,
                "ACC": metrics["ACC"],
                "macro_f1": float(np.mean(task_f1)),
                "LA": metrics["LA"],
                "BWT": metrics["BWT"],
                "FGT": metrics["FGT"],
                "ag_news_accuracy": float(final[0]),
                "sst2_accuracy": float(final[1]),
                "emotion_accuracy": float(final[2]),
            }
            per_seed.append(row)
            base_time, base_memory, base_storage = RESOURCE_BASE[method]
            resource = {
                "method": method,
                "seed": run_seed,
                "training_time_minutes": round(base_time + run_rng.normal(0, 0.45), 3),
                "peak_gpu_memory_mb": int(round(base_memory + run_rng.normal(0, 70))),
                "additional_storage_bytes": int(base_storage),
                "optimizer_steps": 375 if method != "experience_replay" else 425,
            }
            resources.append(resource)
            run_payload = {
                "metadata": {
                    "example_data": True,
                    "generator_seed": seed,
                    "method": method,
                    "run_seed": run_seed,
                    "model": "Qwen/Qwen2.5-0.5B",
                },
                "accuracy_matrix": matrix.round(6).tolist(),
                "metrics": row,
                "resources": resource,
                "predictions": task_predictions,
            }
            run_path = root / "runs" / f"{method}_seed{run_seed}.json"
            run_path.parent.mkdir(parents=True, exist_ok=True)
            run_path.write_text(json.dumps(run_payload, indent=2) + "\n", encoding="utf-8")

    _write_csv(root / "per_seed_metrics.csv", per_seed)
    _write_csv(root / "resources.csv", resources)
    (root / "accuracy_matrices.json").write_text(json.dumps(matrices, indent=2) + "\n", encoding="utf-8")

    summaries = aggregate_method_metrics(per_seed)
    resource_summary = {
        method: {
            "time": float(np.mean([r["training_time_minutes"] for r in resources if r["method"] == method])),
            "memory": float(np.mean([r["peak_gpu_memory_mb"] for r in resources if r["method"] == method])),
            "storage": int(max(r["additional_storage_bytes"] for r in resources if r["method"] == method)),
        }
        for method in METHODS
    }
    comparison = []
    for summary in summaries:
        method = summary["method"]
        comparison.append(
            {
                "method": method,
                "method_display": DISPLAY[method],
                "ACC_mean_percent": round(summary["ACC_mean"] * 100, 2),
                "ACC_sd_percent": round(summary["ACC_sd"] * 100, 2),
                "macro_f1_mean_percent": round(summary["macro_f1_mean"] * 100, 2),
                "macro_f1_sd_percent": round(summary["macro_f1_sd"] * 100, 2),
                "LA_mean_percent": "" if summary["LA_mean"] is None else round(summary["LA_mean"] * 100, 2),
                "BWT_mean_pp": "" if summary["BWT_mean"] is None else round(summary["BWT_mean"] * 100, 2),
                "FGT_mean_pp": "" if summary["FGT_mean"] is None else round(summary["FGT_mean"] * 100, 2),
                "training_time_mean_minutes": round(resource_summary[method]["time"], 2),
                "peak_gpu_memory_mean_mb": round(resource_summary[method]["memory"], 1),
                "additional_storage_bytes": resource_summary[method]["storage"],
            }
        )
    _write_csv(root / "main_comparison.csv", comparison)

    sensitivity = [
        {"buffer_fraction": 0.05, "buffer_examples_per_task": 50, "ACC_percent": 73.6, "FGT_pp": 12.8, "time_minutes": 29.8, "storage_bytes": 47000},
        {"buffer_fraction": 0.10, "buffer_examples_per_task": 100, "ACC_percent": 77.6, "FGT_pp": 9.5, "time_minutes": 31.8, "storage_bytes": 94000},
        {"buffer_fraction": 0.20, "buffer_examples_per_task": 200, "ACC_percent": 79.1, "FGT_pp": 7.9, "time_minutes": 36.2, "storage_bytes": 188000},
    ]
    _write_csv(root / "replay_sensitivity.csv", sensitivity)

    statistical_rows = []
    pending_p = []
    for method in ("experience_replay", "ewc"):
        for task in ("ag_news", "sst2"):
            baseline = sum((prediction_index[("sequential_lora", s, task)] for s in (0, 1, 2)), [])
            contender = sum((prediction_index[(method, s, task)] for s in (0, 1, 2)), [])
            bootstrap = paired_bootstrap_difference(contender, baseline, seed=seed)
            mcnemar = exact_mcnemar(baseline, contender)
            pending_p.append(float(mcnemar["p_value"]))
            statistical_rows.append(
                {
                    "method": method,
                    "task": task,
                    "accuracy_difference_pp": round(bootstrap.estimate * 100, 3),
                    "ci_lower_pp": round(bootstrap.lower * 100, 3),
                    "ci_upper_pp": round(bootstrap.upper * 100, 3),
                    "method_only_correct": mcnemar["method_only_correct"],
                    "baseline_only_correct": mcnemar["baseline_only_correct"],
                    "mcnemar_p": mcnemar["p_value"],
                    "holm_adjusted_p": 0.0,
                }
            )
    corrected = holm_correction(pending_p)
    for row, correction in zip(statistical_rows, corrected):
        row["holm_adjusted_p"] = correction["adjusted_p_value"]
    _write_csv(root / "statistical_comparison.csv", statistical_rows)

    error_categories = [
        {"category": "Topic overlap", "count": 7, "percent": 23.33},
        {"category": "Sparse context", "count": 3, "percent": 10.00},
        {"category": "Negation or irony", "count": 6, "percent": 20.00},
        {"category": "Mixed sentiment", "count": 4, "percent": 13.33},
        {"category": "Minority emotion", "count": 6, "percent": 20.00},
        {"category": "Mixed emotional cues", "count": 4, "percent": 13.33},
    ]
    _write_csv(root / "error_categories.csv", error_categories)

    confusion = {}
    for task in TASKS:
        example = json.loads((root / "runs" / "experience_replay_seed0.json").read_text())
        payload = example["predictions"][task]
        confusion[task] = confusion_matrix(
            payload["gold"], payload["predicted"], range(CLASS_COUNTS[task])
        ).tolist()
    (root / "confusion_matrices.json").write_text(json.dumps(confusion, indent=2) + "\n", encoding="utf-8")

    hypotheses = [
        {"hypothesis": "H1", "example_status": "Supported", "evidence": "ER and EWC show lower FGT than Seq-LoRA in the deterministic example package."},
        {"hypothesis": "H2", "example_status": "Supported", "evidence": "ER improves retention with additional training time and replay storage."},
    ]
    _write_csv(root / "hypothesis_evaluation.csv", hypotheses)

    manifest = {
        "package_type": "deterministic_illustrative_example",
        "generator_seed": seed,
        "measured_model_runs": False,
        "methods": list(METHODS),
        "seeds": [0, 1, 2],
        "files": sorted(str(path.relative_to(root)).replace("\\", "/") for path in root.rglob("*") if path.is_file()),
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest

