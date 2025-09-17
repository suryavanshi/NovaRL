from __future__ import annotations

from typing import Any

import torch

from core.interfaces import EnvStep
from rewards.base import RewardManagerBase


class IdentityRewardManager(RewardManagerBase):
    """Returns rewards directly from environment steps."""

    def score(self, samples: Any) -> torch.Tensor:
        if isinstance(samples, EnvStep):
            return samples.rewards
        if isinstance(samples, torch.Tensor):
            return samples
        raise TypeError(f"Unsupported sample type: {type(samples)!r}")


class NoisyRewardManager(RewardManagerBase):
    """Adds small Gaussian noise to the environment reward for testing."""

    def __init__(self, noise_std: float = 0.1) -> None:
        self.noise_std = noise_std

    def score(self, samples: Any) -> torch.Tensor:
        base = IdentityRewardManager().score(samples)
        if self.noise_std == 0:
            return base
        noise = torch.randn_like(base) * self.noise_std
        return base + noise


__all__ = ["IdentityRewardManager", "NoisyRewardManager"]
