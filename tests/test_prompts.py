from __future__ import annotations

from rmt.data.loader import DatasetRecord
from rmt.data.prompts import build_prompt, encode_training_record, label_word


class CharacterTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        prefix = [1] if add_special_tokens else []
        return prefix + [ord(character) for character in text]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        return "".join(chr(token) for token in token_ids if token > 1)


def record() -> DatasetRecord:
    return DatasetRecord(
        split="train",
        task=1,
        source_index=1,
        label=3,
        text="A technology story",
        task_name="AG News",
        dataset="fancyzhx/ag_news",
        source_split="train",
        label_name="Sci/Tech",
        record_id="T1_train_0000001",
        text_sha256="unused",
    )


def test_prompt_uses_protocol_label_word() -> None:
    item = record()
    assert label_word(item) == "Technology"
    prompt = build_prompt(item)
    assert prompt.startswith("Task: Classify the news topic.")
    assert prompt.endswith("Answer:")


def test_training_encoding_masks_prompt_tokens() -> None:
    encoded = encode_training_record(record(), CharacterTokenizer(), maximum_length=160)
    first_target = next(index for index, value in enumerate(encoded["labels"]) if value != -100)
    assert all(value == -100 for value in encoded["labels"][:first_target])
    assert encoded["labels"][-1] == 0
    assert len(encoded["input_ids"]) <= 160

