"""Prompt templates and target-only causal-language-model encoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .loader import DatasetRecord


class TokenizerLike(Protocol):
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...


@dataclass(frozen=True)
class TaskPrompt:
    key: str
    instruction: str
    labels: tuple[str, ...]


TASK_PROMPTS = {
    1: TaskPrompt(
        "ag_news",
        "Classify the news topic.",
        ("World", "Sports", "Business", "Technology"),
    ),
    2: TaskPrompt(
        "sst2",
        "Classify the sentiment.",
        ("Negative", "Positive"),
    ),
    3: TaskPrompt(
        "emotion",
        "Classify the expressed emotion.",
        ("Sadness", "Joy", "Love", "Anger", "Fear", "Surprise"),
    ),
}


def label_word(record: DatasetRecord) -> str:
    """Return the protocol label word, including the AG News wording change."""
    return TASK_PROMPTS[record.task].labels[record.label]


def truncate_text(tokenizer: TokenizerLike, text: str, limit: int = 96) -> str:
    token_ids = tokenizer.encode(text, add_special_tokens=False)[:limit]
    return tokenizer.decode(token_ids, skip_special_tokens=True).strip()


def build_prompt(record: DatasetRecord, text: str | None = None) -> str:
    spec = TASK_PROMPTS[record.task]
    return (
        f"Task: {spec.instruction}\n"
        f"Options: {', '.join(spec.labels)}\n"
        f"Text: {record.text if text is None else text}\n"
        "Answer:"
    )


def encode_training_record(
    record: DatasetRecord,
    tokenizer: TokenizerLike,
    text_token_limit: int = 96,
    maximum_length: int = 160,
) -> dict[str, Any]:
    """Encode one record and mask loss outside the answer tokens."""
    shortened = truncate_text(tokenizer, record.text, text_token_limit)
    prompt_ids = tokenizer.encode(build_prompt(record, shortened), add_special_tokens=True)
    answer_ids = tokenizer.encode(" " + label_word(record), add_special_tokens=False)
    if tokenizer.eos_token_id is not None:
        answer_ids.append(tokenizer.eos_token_id)
    available = maximum_length - len(answer_ids)
    if available <= 0:
        raise ValueError("maximum_length is too small for the answer tokens")
    prompt_ids = prompt_ids[:available]
    input_ids = prompt_ids + answer_ids
    return {
        "input_ids": input_ids,
        "attention_mask": [1] * len(input_ids),
        "labels": [-100] * len(prompt_ids) + answer_ids,
        "task": record.task,
        "record_id": record.record_id,
    }


def pad_batch(examples: Sequence[dict[str, Any]], pad_token_id: int) -> dict[str, Any]:
    """Pad encoded examples and return PyTorch tensors."""
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is required to create a training batch") from error
    width = max(len(item["input_ids"]) for item in examples)
    input_ids, attention_mask, labels = [], [], []
    for item in examples:
        padding = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_token_id] * padding)
        attention_mask.append(item["attention_mask"] + [0] * padding)
        labels.append(item["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

