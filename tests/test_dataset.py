from __future__ import annotations

import hashlib
from collections import Counter

from rmt.data.loader import DatasetRecord, group_records
from rmt.data.preprocessing import LABELS, dataset_summary, normalize_text
from rmt.data.validator import validate_records


def make_records() -> list[DatasetRecord]:
    records = []
    task_names = {1: "AG News", 2: "SST-2", 3: "Emotion"}
    datasets = {1: "fancyzhx/ag_news", 2: "nyu-mll/glue", 3: "dair-ai/emotion"}
    expected = {"train": 1000, "validation": 200, "test": 800}
    for task, labels in LABELS.items():
        for split, count in expected.items():
            for index in range(count):
                text = f"Unique task {task} {split} example {index}"
                normalized = normalize_text(text).casefold()
                label = index % len(labels)
                records.append(
                    DatasetRecord(
                        split=split,
                        task=task,
                        source_index=index,
                        label=label,
                        text=text,
                        task_name=task_names[task],
                        dataset=datasets[task],
                        source_split=split,
                        label_name=labels[label],
                        record_id=f"T{task}_{split}_{index:07d}",
                        text_sha256=hashlib.sha256(normalized.encode()).hexdigest(),
                    )
                )
    return records


def test_complete_dataset_passes_integrity_checks() -> None:
    report = validate_records(make_records())
    assert report.valid
    assert report.total_records == 6000
    assert report.split_counts["task_3/test"] == 800


def test_duplicate_text_is_rejected() -> None:
    records = make_records()
    records[-1] = DatasetRecord(**{**records[-1].__dict__, "text_sha256": records[0].text_sha256})
    report = validate_records(records)
    assert not report.valid
    assert any("Duplicate normalized texts" in error for error in report.errors)


def test_grouping_and_summary_are_stable() -> None:
    records = make_records()
    grouped = group_records(records)
    summary = dataset_summary(records)
    assert len(grouped["ag_news"]["train"]) == 1000
    assert summary["split_counts"]["emotion/validation"] == 200


def test_frozen_label_spaces() -> None:
    assert LABELS[1] == ("World", "Sports", "Business", "Sci/Tech")
    assert LABELS[2] == ("Negative", "Positive")
    assert LABELS[3] == ("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise")

