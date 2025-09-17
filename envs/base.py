from __future__ import annotations

from typing import Optional

import torch

from core.interfaces import Environment, EnvStep


class BatchedEnvironment(Environment):
    """Base class that handles device placement for batched environments."""

    def __init__(self, batch_size: int, device: Optional[torch.device] = None) -> None:
        self.batch_size = batch_size
        self.device = device or torch.device("cpu")

    def _zeros(self) -> EnvStep:
        obs = torch.zeros((self.batch_size, 1), device=self.device)
        zeros = torch.zeros((self.batch_size,), device=self.device)
        infos = tuple({} for _ in range(self.batch_size))
        return EnvStep(observations=obs, rewards=zeros, dones=zeros, infos=infos)


__all__ = ["BatchedEnvironment"]
