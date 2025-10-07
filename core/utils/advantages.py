"""Reusable advantage and return estimation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch

from .stats import compute_gae


@dataclass(slots=True)
class AdvantageEstimate:
    """Container bundling raw advantages with bootstrapped returns."""

    advantages: torch.Tensor
    returns: torch.Tensor


class AdvantageEstimator(Protocol):
    """Protocol describing advantage estimation strategies."""

    def estimate(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> AdvantageEstimate:
        """Return estimated advantages and bootstrapped returns."""


class GAEAdvantageEstimator:
    """Generalised Advantage Estimation helper.

    The estimator expects time-major tensors with leading shape ``(T, B, ...)``.
    ``values`` should contain the critic values for each transition and must
    therefore have shape ``(T, B)`` while ``bootstrap_value`` corresponds to the
    value prediction for the state following the final transition and must have
    shape ``(B,)``.
    """

    def __init__(self, gamma: float = 0.99, lam: float = 0.95) -> None:
        self.gamma = float(gamma)
        self.lam = float(lam)

    def estimate(
        self,
        *,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        bootstrap_value: torch.Tensor,
    ) -> AdvantageEstimate:
        if values.dim() != rewards.dim():  # pragma: no cover - defensive
            raise ValueError("values and rewards must have matching dimensionality")
        bootstrap = torch.cat([values, bootstrap_value.unsqueeze(0)], dim=0)
        advantages = compute_gae(
            rewards=rewards,
            values=bootstrap,
            dones=dones,
            gamma=self.gamma,
            lam=self.lam,
        )
        returns = advantages + values
        return AdvantageEstimate(advantages=advantages, returns=returns)


def normalize_advantages(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Return z-scored advantages preserving the original tensor layout."""

    if advantages.numel() == 0:  # pragma: no cover - defensive
        return advantages
    mean = advantages.mean()
    std = advantages.std(unbiased=False)
    return (advantages - mean) / (std + eps)


__all__ = ["AdvantageEstimate", "AdvantageEstimator", "GAEAdvantageEstimator", "normalize_advantages"]

