from __future__ import annotations

import torch


def compute_discounted_returns(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute discounted returns with bootstrap values."""

    t, _ = rewards.shape
    returns = torch.zeros_like(rewards)
    next_return = values[-1]
    for step in reversed(range(t)):
        mask = 1.0 - dones[step]
        next_return = rewards[step] + gamma * next_return * mask
        returns[step] = next_return
    return returns


def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    t, b = rewards.shape
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros(b, device=rewards.device, dtype=rewards.dtype)
    for step in reversed(range(t)):
        delta = rewards[step] + gamma * values[step + 1] * (1.0 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1.0 - dones[step]) * gae
        advantages[step] = gae
    return advantages


def explained_variance(y_true: torch.Tensor, y_pred: torch.Tensor) -> float:
    var_y = torch.var(y_true)
    if var_y.item() == 0:
        return 0.0
    return float(1.0 - torch.var(y_true - y_pred) / var_y)


__all__ = ["compute_discounted_returns", "compute_gae", "explained_variance"]
