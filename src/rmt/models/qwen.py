"""Qwen model construction and constrained label-likelihood scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from rmt.data.loader import DatasetRecord
from rmt.data.prompts import TASK_PROMPTS, build_prompt, truncate_text


@dataclass
class ModelBundle:
    model: Any
    tokenizer: Any
    device: str


def initialize_qwen(
    model_name: str = "Qwen/Qwen2.5-0.5B",
    use_fp16_on_cuda: bool = True,
    attach_adapter: bool = True,
    lora_settings: Any | None = None,
) -> ModelBundle:
    """Initialize the frozen base model and its LoRA adapter.

    This function performs model download and allocation only when called by an
    experiment command. Importing the package or running ordinary tests does not
    load model weights.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("Install the experiment dependencies to load Qwen") from error

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" and use_fp16_on_cuda else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    if attach_adapter:
        from .lora import LoRASettings, attach_lora

        model = attach_lora(model, lora_settings or LoRASettings())
    model.to(device)
    return ModelBundle(model=model, tokenizer=tokenizer, device=device)


def score_label_options(
    bundle: ModelBundle,
    record: DatasetRecord,
    maximum_length: int = 160,
    text_token_limit: int = 96,
) -> list[float]:
    """Return mean target-token log probabilities for every valid label."""
    import torch
    import torch.nn.functional as functional

    model, tokenizer = bundle.model, bundle.tokenizer
    shortened = truncate_text(tokenizer, record.text, text_token_limit)
    prompt_ids = tokenizer.encode(build_prompt(record, shortened), add_special_tokens=True)
    scores: list[float] = []
    model.eval()
    with torch.no_grad():
        for option in TASK_PROMPTS[record.task].labels:
            target = tokenizer.encode(" " + option, add_special_tokens=False)
            if tokenizer.eos_token_id is not None:
                target.append(tokenizer.eos_token_id)
            prefix = prompt_ids[: maximum_length - len(target)]
            ids = torch.tensor([prefix + target], dtype=torch.long, device=bundle.device)
            logits = model(input_ids=ids).logits[:, :-1]
            predicted = ids[:, 1:]
            log_probs = functional.log_softmax(logits.float(), dim=-1)
            token_scores = log_probs.gather(-1, predicted.unsqueeze(-1)).squeeze(-1)
            target_scores = token_scores[:, len(prefix) - 1 :]
            scores.append(float(target_scores.mean().item()))
    return scores


def predict_label(bundle: ModelBundle, record: DatasetRecord) -> int:
    scores = score_label_options(bundle, record)
    return max(range(len(scores)), key=scores.__getitem__)

