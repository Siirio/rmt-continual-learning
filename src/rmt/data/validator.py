"""Integrity checks for the frozen Assignment 4 dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .loader import CSV_NAME, DatasetRecord, load_records
from .preprocessing import LABELS, normalize_text

EXPECTED_SPLITS = {"train": 1000, "validation": 200, "test": 800}


@dataclass
class ValidationReport:
    valid: bool
    total_records: int
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    split_counts: dict[str, int] = field(default_factory=dict)
    class_counts: dict[str, int] = field(default_factory=dict)

    def require_valid(self) -> "ValidationReport":
        if not self.valid:
            raise ValueError("Dataset validation failed: " + "; ".join(self.errors))
        return self


def _text_hash(text: str) -> str:
    normalized = normalize_text(text).casefold()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _split_hash(records: Iterable[DatasetRecord]) -> str:
    content = "\n".join(f"{item.record_id}\t{item.text_sha256}" for item in records)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def validate_records(
    records: Iterable[DatasetRecord], manifest: dict[str, object] | None = None
) -> ValidationReport:
    items = list(records)
    errors: list[str] = []
    counts = Counter((item.task, item.split) for item in items)
    record_ids = [item.record_id for item in items]
    hashes = [item.text_sha256 for item in items]

    if len(items) != 6000:
        errors.append(f"Expected 6000 records, found {len(items)}")
    if len(set(record_ids)) != len(record_ids):
        errors.append("Duplicate record identifiers detected")
    if len(set(hashes)) != len(hashes):
        errors.append("Duplicate normalized texts or cross-split leakage detected")

    for item in items:
        if item.split not in EXPECTED_SPLITS:
            errors.append(f"Invalid split {item.split!r} in {item.record_id}")
        if item.task not in LABELS:
            errors.append(f"Invalid task {item.task!r} in {item.record_id}")
            continue
        labels = LABELS[item.task]
        if not 0 <= item.label < len(labels) or item.label_name != labels[item.label]:
            errors.append(f"Label mismatch in {item.record_id}")
        if not normalize_text(item.text):
            errors.append(f"Empty text in {item.record_id}")
        if _text_hash(item.text) != item.text_sha256:
            errors.append(f"Text hash mismatch in {item.record_id}")

    for task in LABELS:
        for split, expected in EXPECTED_SPLITS.items():
            actual = counts[(task, split)]
            if actual != expected:
                errors.append(f"Task {task} {split}: expected {expected}, found {actual}")

    if manifest:
        for expected in manifest.get("split_hashes", []):
            task = int(expected["task"])
            split = str(expected["split"])
            subset = [item for item in items if item.task == task and item.split == split]
            actual = _split_hash(subset)
            if actual != expected["sha256_record_ids_and_text"]:
                errors.append(f"Task {task} {split}: split hash mismatch")

    class_counts = Counter((item.task_name, item.split, item.label_name) for item in items)
    return ValidationReport(
        valid=not errors,
        total_records=len(items),
        errors=errors,
        split_counts={f"task_{task}/{split}": count for (task, split), count in sorted(counts.items())},
        class_counts={"/".join(key): count for key, count in sorted(class_counts.items())},
    )


def validate_dataset(
    dataset_path: str | Path, manifest_path: str | Path | None = None
) -> ValidationReport:
    manifest = None
    if manifest_path is not None:
        with Path(manifest_path).open(encoding="utf-8") as stream:
            manifest = json.load(stream)
    return validate_records(load_records(dataset_path), manifest)

