"""Scientific figures for example and measured result packages."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

METHOD_ORDER = ["sequential_lora", "experience_replay", "ewc", "joint_training"]
DISPLAY = {
    "sequential_lora": "Seq-LoRA",
    "experience_replay": "ER (10%)",
    "ewc": "Online EWC",
    "joint_training": "Joint",
}
COLORS = {
    "sequential_lora": "#4B5563",
    "experience_replay": "#17823B",
    "ewc": "#76A83B",
    "joint_training": "#B7D68A",
}
TASK_LABELS = ["AG News", "SST-2", "Emotion"]


def _pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("Install the reporting dependencies to generate figures") from error
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#4B5563",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.9,
            "figure.facecolor": "white",
            "axes.facecolor": "#FCFDFC",
            "savefig.facecolor": "white",
        }
    )
    return plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _save(fig: Any, output: Path, name: str) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"{name}.png", output / f"{name}.svg"]
    fig.savefig(paths[0], dpi=220, bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    return paths


def accuracy_comparison(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    rows = _read_csv(results / "main_comparison.csv")
    methods = [row["method"] for row in rows]
    means = [float(row["ACC_mean_percent"]) for row in rows]
    errors = [float(row["ACC_sd_percent"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8.2, 4.8))
    bars = axis.bar(
        [DISPLAY[item] for item in methods],
        means,
        yerr=errors,
        capsize=4,
        color=[COLORS[item] for item in methods],
        edgecolor="white",
    )
    axis.set(title="Final average accuracy", ylabel="Accuracy (%)", ylim=(0, 100))
    axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, output, "accuracy_comparison")


def forgetting_comparison(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    rows = [row for row in _read_csv(results / "main_comparison.csv") if row["FGT_mean_pp"]]
    methods = [row["method"] for row in rows]
    values = [float(row["FGT_mean_pp"]) for row in rows]
    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    bars = axis.bar(
        [DISPLAY[item] for item in methods], values, color=[COLORS[item] for item in methods]
    )
    axis.set(title="Average forgetting after the final update", ylabel="FGT (percentage points)")
    axis.bar_label(bars, fmt="%.1f", padding=4)
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, output, "forgetting_comparison")


def retention_curves(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    matrices = json.loads((results / "accuracy_matrices.json").read_text(encoding="utf-8"))
    stages = ["Base", "After T1", "After T2", "After T3"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    for axis, task_index, task_name in zip(axes, (0, 1), TASK_LABELS[:2]):
        for method in METHOD_ORDER[:3]:
            arrays = np.asarray(list(matrices[method].values()), dtype=float)
            mean = arrays[:, :, task_index].mean(axis=0) * 100
            standard_deviation = arrays[:, :, task_index].std(axis=0, ddof=1) * 100
            axis.plot(stages, mean, marker="o", linewidth=2, label=DISPLAY[method], color=COLORS[method])
            axis.fill_between(stages, mean - standard_deviation, mean + standard_deviation, color=COLORS[method], alpha=0.12)
        axis.set(title=f"{task_name} retention", xlabel="Evaluation stage")
        axis.tick_params(axis="x", rotation=20)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.tight_layout()
    return _save(fig, output, "retention_curves")


def accuracy_matrix_heatmaps(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    matrices = json.loads((results / "accuracy_matrices.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.2), constrained_layout=True)
    image = None
    for axis, method in zip(axes, METHOD_ORDER[:3]):
        matrix = np.asarray(list(matrices[method].values()), dtype=float).mean(axis=0) * 100
        image = axis.imshow(matrix, cmap="YlGn", vmin=0, vmax=100, aspect="auto")
        axis.set_title(DISPLAY[method])
        axis.set_xticks(range(3), TASK_LABELS, rotation=25, ha="right")
        axis.set_yticks(range(4), ["Base", "After T1", "After T2", "After T3"])
        for row in range(4):
            for column in range(3):
                color = "white" if matrix[row, column] > 62 else "#172554"
                axis.text(column, row, f"{matrix[row, column]:.1f}", ha="center", va="center", color=color, fontsize=8)
        axis.grid(False)
    fig.colorbar(image, ax=axes, label="Accuracy (%)", shrink=0.78)
    fig.suptitle("Accuracy matrices across sequential updates", fontsize=14, fontweight="bold")
    return _save(fig, output, "accuracy_matrix_heatmaps")


def _resource_bar(results: Path, output: Path, field: str, title: str, ylabel: str, name: str) -> list[Path]:
    plt = _pyplot()
    rows = _read_csv(results / "main_comparison.csv")
    methods = [row["method"] for row in rows]
    values = [float(row[field]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    bars = axis.bar([DISPLAY[item] for item in methods], values, color=[COLORS[item] for item in methods])
    axis.set(title=title, ylabel=ylabel)
    axis.bar_label(bars, fmt="%.1f", padding=4, fontsize=9)
    axis.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, output, name)


def training_time_comparison(results: Path, output: Path) -> list[Path]:
    return _resource_bar(results, output, "training_time_mean_minutes", "Training time per run", "Minutes", "training_time_comparison")


def gpu_memory_comparison(results: Path, output: Path) -> list[Path]:
    return _resource_bar(results, output, "peak_gpu_memory_mean_mb", "Peak GPU memory", "Memory (MB)", "gpu_memory_comparison")


def replay_sensitivity(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    rows = _read_csv(results / "replay_sensitivity.csv")
    fractions = [float(row["buffer_fraction"]) * 100 for row in rows]
    accuracy = [float(row["ACC_percent"]) for row in rows]
    forgetting = [float(row["FGT_pp"]) for row in rows]
    fig, axis = plt.subplots(figsize=(7.6, 4.7))
    axis.plot(fractions, accuracy, marker="o", linewidth=2.2, color=COLORS["experience_replay"], label="ACC")
    axis.plot(fractions, forgetting, marker="s", linewidth=2.2, color="#76A83B", label="FGT")
    axis.set(title="Replay-buffer sensitivity", xlabel="Buffer per completed task (%)", ylabel="Percentage points", xticks=fractions)
    axis.legend(frameon=False)
    fig.tight_layout()
    return _save(fig, output, "replay_buffer_sensitivity")


def confusion_matrices(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    payload = json.loads((results / "confusion_matrices.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.1), constrained_layout=True)
    image = None
    for axis, task, title in zip(axes, ("ag_news", "sst2", "emotion"), TASK_LABELS):
        matrix = np.asarray(payload[task], dtype=float)
        normalized = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
        image = axis.imshow(normalized, cmap="YlGn", vmin=0, vmax=1)
        axis.set(title=title, xlabel="Predicted class", ylabel="True class")
        axis.set_xticks(range(matrix.shape[1]))
        axis.set_yticks(range(matrix.shape[0]))
        axis.grid(False)
    fig.colorbar(image, ax=axes, label="Row proportion", shrink=0.78)
    fig.suptitle("Example final confusion matrices for ER (seed 0)", fontsize=14, fontweight="bold")
    return _save(fig, output, "confusion_matrices")


def retention_vs_cost(results: Path, output: Path) -> list[Path]:
    plt = _pyplot()
    rows = [row for row in _read_csv(results / "main_comparison.csv") if row["FGT_mean_pp"]]
    fig, axis = plt.subplots(figsize=(7.5, 5))
    for row in rows:
        method = row["method"]
        x = float(row["training_time_mean_minutes"])
        y = float(row["FGT_mean_pp"])
        axis.scatter(x, y, s=95, color=COLORS[method], edgecolor="white", linewidth=1.2)
        axis.annotate(DISPLAY[method], (x, y), xytext=(7, 5), textcoords="offset points", fontsize=9, fontweight="bold")
    axis.set(title="Forgetting versus training cost", xlabel="Training time per run (minutes)", ylabel="FGT (percentage points)")
    fig.tight_layout()
    return _save(fig, output, "retention_vs_computational_cost")


FIGURES: tuple[Callable[[Path, Path], list[Path]], ...] = (
    accuracy_comparison,
    forgetting_comparison,
    retention_curves,
    accuracy_matrix_heatmaps,
    training_time_comparison,
    gpu_memory_comparison,
    replay_sensitivity,
    confusion_matrices,
    retention_vs_cost,
)


def generate_all_figures(results_dir: str | Path, output_dir: str | Path) -> list[Path]:
    results, output = Path(results_dir), Path(output_dir)
    generated = []
    for generator in FIGURES:
        generated.extend(generator(results, output))
        _pyplot().close("all")
    return generated

