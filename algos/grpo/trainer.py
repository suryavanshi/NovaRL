from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, Optional, Union

import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from core.interfaces import Trainer
from core.types import TrajectoryBatch
from core.utils.advantages import normalize_advantages


@dataclass(slots=True)
class GRPOGroupingConfig:
    """Configuration describing how completions are grouped for GRPO."""

    group_size: int = 1
    drop_incomplete_groups: bool = True


class GRPOTrainer(Trainer):
    """Implementation of Group Relative Policy Optimization."""

    def __init__(
        self,
        policy: nn.Module,
        optimizer: torch.optim.Optimizer,
        grouping: Optional[GRPOGroupingConfig] = None,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 1.0,
        kl_coef: float = 0.0,
        adaptive_kl: bool = False,
        kl_target: float = 0.01,
        kl_adaptation_speed: float = 1.5,
        reference_model: Optional[Union[nn.Module, str]] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.grouping = grouping or GRPOGroupingConfig()
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.kl_coef = kl_coef
        self.adaptive_kl = adaptive_kl
        self.kl_target = kl_target
        self.kl_adaptation_speed = kl_adaptation_speed
        self.reference_model = self._setup_reference_model(reference_model)

    def step(self, batch: TrajectoryBatch) -> Mapping[str, float]:
        self.policy.train()
        flat = batch.flatten()
        observations = flat.observations
        actions = flat.actions.long()
        returns = flat.returns

        policy_out = self.policy(observations)
        logits = policy_out["logits"]
        values = policy_out.get("value")
        value_predictions = None
        if values is not None:
            value_predictions = values.squeeze(-1)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()

        advantages = self._group_relative_advantages(returns)
        norm_advantages = normalize_advantages(advantages)

        policy_loss = -(log_probs * norm_advantages.detach()).mean()

        value_loss = torch.tensor(0.0, device=observations.device)
        if value_predictions is not None:
            value_loss = F.mse_loss(value_predictions, returns)

        kl_penalty = torch.tensor(0.0, device=observations.device)
        ref_kl = torch.tensor(0.0, device=observations.device)
        if self.reference_model is not None:
            with torch.no_grad():
                ref_outputs = self.reference_model(observations)
                ref_logits = ref_outputs["logits"]
            ref_dist = Categorical(logits=ref_logits)
            ref_kl = torch.distributions.kl_divergence(dist, ref_dist).mean()
            if self.kl_coef > 0:
                kl_penalty = self.kl_coef * ref_kl

        loss = policy_loss + self.value_coef * value_loss + kl_penalty - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.adaptive_kl and self.reference_model is not None and ref_kl.item() > 0:
            self._update_kl_coef(float(ref_kl.item()))

        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": float(ref_kl.item()),
            "kl_coef": float(self.kl_coef),
            "kl_penalty": float(kl_penalty.item()),
            "reward_mean": float(flat.rewards.mean().item()),
            "advantage_norm": float(norm_advantages.mean().abs().item()),
        }
        return metrics

    def _group_relative_advantages(self, returns: torch.Tensor) -> torch.Tensor:
        group_size = max(int(self.grouping.group_size), 1)
        if group_size == 1:
            return returns - returns.mean()

        total = returns.shape[0]
        trimmed = (total // group_size) * group_size
        grouped = returns[:trimmed].reshape(-1, group_size)
        group_mean = grouped.mean(dim=1, keepdim=True)
        adjusted = grouped - group_mean
        if trimmed == total:
            return adjusted.reshape(-1)

        remainder = returns[trimmed:]
        if self.grouping.drop_incomplete_groups:
            return torch.cat([adjusted.reshape(-1), remainder - remainder.mean()], dim=0)
        padding = group_size - remainder.shape[0]
        padded = torch.cat([remainder, remainder.new_zeros(padding)], dim=0)
        padded = padded.reshape(1, group_size)
        pad_mean = padded.mean(dim=1, keepdim=True)
        pad_adjusted = padded - pad_mean
        pad_values = pad_adjusted.reshape(-1)[: remainder.shape[0]]
        return torch.cat([adjusted.reshape(-1), pad_values], dim=0)

    def _setup_reference_model(
        self, reference_model: Optional[Union[nn.Module, str]]
    ) -> Optional[nn.Module]:
        if isinstance(reference_model, str):
            mode = reference_model.lower()
            if mode == "tie":
                return self.policy
            if mode not in {"copy", "auto"}:
                raise ValueError("reference_model string must be 'copy', 'tie', or 'auto'")
            reference_model = None
        if reference_model is None:
            if self.kl_coef == 0 and not self.adaptive_kl:
                return None
            reference_model = copy.deepcopy(self.policy)
        if reference_model is self.policy:
            return reference_model
        for param in reference_model.parameters():
            param.requires_grad_(False)
        reference_model.eval()
        params = list(self.policy.parameters())
        if params:
            reference_model.to(params[0].device)
        return reference_model

    def _update_kl_coef(self, kl_value: float) -> None:
        if self.kl_target <= 0:
            return
        if kl_value > self.kl_target * 1.5:
            self.kl_coef *= self.kl_adaptation_speed
        elif kl_value < self.kl_target / 1.5:
            self.kl_coef /= self.kl_adaptation_speed


__all__ = ["GRPOTrainer", "GRPOGroupingConfig"]

