"""Checkpoint helpers for NovaRL experiments."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


@dataclass(slots=True)
class CheckpointState:
    """Represents the serialized state of an experiment."""

    step: int
    payload: Mapping[str, Any]


class CheckpointManager:
    """Handles atomic save/load cycles for experiment checkpoints."""

    def __init__(self, directory: os.PathLike[str] | str, *, filename: str = "latest.pt") -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._filename = filename

    @property
    def path(self) -> Path:
        return self._directory / self._filename

    def save(self, state: CheckpointState) -> Path:
        """Atomically persist ``state`` to disk and return the final path."""

        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".tmp", dir=str(self._directory))
        os.close(tmp_fd)
        torch.save({"step": state.step, "payload": dict(state.payload)}, tmp_path)
        final_path = self.path
        os.replace(tmp_path, final_path)
        return final_path

    def load(self, path: os.PathLike[str] | str | None = None) -> CheckpointState:
        """Load a previously saved checkpoint."""

        target = Path(path) if path is not None else self.path
        data = torch.load(target, map_location="cpu")
        return CheckpointState(step=int(data["step"]), payload=data["payload"])


__all__ = ["CheckpointManager", "CheckpointState"]

