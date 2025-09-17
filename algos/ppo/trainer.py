from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from core.interfaces import Trainer
from core.types import TrajectoryBatch


class PPOTrainer(Trainer):
    """A minimal PPO implementation suitable for toy examples."""

    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        clip_range: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def step(self, batch: TrajectoryBatch) -> Mapping[str, float]:
        self.policy.train()
        flat = batch.flatten()
        observations = flat.observations
        actions = flat.actions.long()
        old_log_probs = flat.log_probs
        advantages = flat.advantages
        returns = flat.returns

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        outputs = self.policy(observations)
        logits = outputs["logits"]
        values = outputs["value"].squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(values, returns)
        loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        approx_kl = torch.mean(old_log_probs - log_probs).clamp_min(0).item()
        clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean().item()

        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": float(approx_kl),
            "clip_fraction": float(clip_fraction),
            "reward_mean": float(flat.rewards.mean().item()),
        }
        return metrics


__all__ = ["PPOTrainer"]
