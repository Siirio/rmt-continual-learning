"""Checkpoint serialization for resumable experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class CheckpointManager:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def run_directory(self, run_id: str) -> Path:
        return self.root / run_id

    def save(self, run_id: str, model: Any, metadata: dict[str, Any]) -> Path:
        """Save trainable state and stage metadata atomically at file level."""
        try:
            import torch
        except ImportError as error:
            raise RuntimeError("PyTorch is required for checkpoint serialization") from error
        directory = self.run_directory(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        state = {
            name: parameter.detach().cpu()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        temporary = directory / "adapter_state.pt.tmp"
        final = directory / "adapter_state.pt"
        torch.save(state, temporary)
        temporary.replace(final)
        metadata_path = directory / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return directory

    def load(self, run_id: str, model: Any) -> dict[str, Any]:
        import torch

        directory = self.run_directory(run_id)
        state = torch.load(directory / "adapter_state.pt", map_location="cpu", weights_only=True)
        parameters = dict(model.named_parameters())
        missing = set(state) - set(parameters)
        if missing:
            raise ValueError(f"Checkpoint parameters are not present in the model: {sorted(missing)}")
        with torch.no_grad():
            for name, value in state.items():
                parameters[name].copy_(value.to(parameters[name].device))
        return json.loads((directory / "metadata.json").read_text(encoding="utf-8"))

    def exists(self, run_id: str) -> bool:
        directory = self.run_directory(run_id)
        return (directory / "adapter_state.pt").exists() and (directory / "metadata.json").exists()

