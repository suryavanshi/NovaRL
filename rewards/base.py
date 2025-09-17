from __future__ import annotations

import abc
from typing import Any

import torch

from core.interfaces import RewardManager


class RewardManagerBase(RewardManager, abc.ABC):
    """Base class for reward managers."""

    @abc.abstractmethod
    def score(self, samples: Any) -> torch.Tensor: ...


__all__ = ["RewardManagerBase"]
