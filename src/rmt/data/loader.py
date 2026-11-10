"""Load the frozen CSV from a file or the supplied dataset package."""

from __future__ import annotations

import csv
import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, TextIO

CSV_NAME = "assignment4_real_6000.csv"
REQUIRED_COLUMNS = {
    "split",
    "task",
    "source_index",
    "label",
    "text",
    "task_name",
    "dataset",
    "source_split",
    "label_name",
    "record_id",
    "text_sha256",
}


@dataclass(frozen=True)
class DatasetRecord:
    split: str
    task: int
    source_index: int
    label: int
    text: str
    task_name: str
    dataset: str
    source_split: str
    label_name: str
    record_id: str
    text_sha256: str

    @classmethod
    def from_row(cls, row: dict[str, str]) -> "DatasetRecord":
        return cls(
            split=row["split"],
            task=int(row["task"]),
            source_index=int(row["source_index"]),
            label=int(row["label"]),
            text=row["text"],
            task_name=row["task_name"],
            dataset=row["dataset"],
            source_split=row["source_split"],
            label_name=row["label_name"],
            record_id=row["record_id"],
            text_sha256=row["text_sha256"],
        )


def _records_from_stream(stream: TextIO) -> Iterator[DatasetRecord]:
    reader = csv.DictReader(stream)
    columns = set(reader.fieldnames or [])
    missing = REQUIRED_COLUMNS - columns
    if missing:
        raise ValueError(f"Dataset columns are missing: {', '.join(sorted(missing))}")
    for line_number, row in enumerate(reader, start=2):
        try:
            yield DatasetRecord.from_row(row)
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid dataset row at line {line_number}: {error}") from error


def load_records(path: str | Path) -> list[DatasetRecord]:
    """Load records from a CSV or ZIP without changing their order."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as archive:
            if CSV_NAME not in archive.namelist():
                raise ValueError(f"{source} does not contain {CSV_NAME}")
            payload = archive.read(CSV_NAME).decode("utf-8")
        return list(_records_from_stream(io.StringIO(payload)))
    with source.open(encoding="utf-8", newline="") as stream:
        return list(_records_from_stream(stream))


def group_records(
    records: Iterable[DatasetRecord],
) -> dict[str, dict[str, list[DatasetRecord]]]:
    """Group records by stable task key and split."""
    task_keys = {1: "ag_news", 2: "sst2", 3: "emotion"}
    grouped: dict[str, dict[str, list[DatasetRecord]]] = {}
    for record in records:
        task = task_keys[record.task]
        grouped.setdefault(task, {}).setdefault(record.split, []).append(record)
    return grouped

