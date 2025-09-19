from __future__ import annotations

import copy
from typing import Mapping, Optional, Union

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
        kl_coef: float = 0.0,
        adaptive_kl: bool = False,
        kl_target: float = 0.01,
        kl_adaptation_speed: float = 1.5,
        reference_model: Optional[Union[nn.Module, str]] = None,
    ) -> None:
        self.policy = policy
        self.optimizer = optimizer
        self.clip_range = clip_range
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
        if self.clip_range is not None and self.clip_range > 0:
            clipped_ratio = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range)
            surr2 = clipped_ratio * advantages
            pg_objective = torch.min(surr1, surr2)
            clip_fraction = (torch.abs(ratio - 1.0) > self.clip_range).float().mean()
        else:
            pg_objective = surr1
            clip_fraction = torch.tensor(0.0, device=pg_objective.device)
        policy_loss = -pg_objective.mean()

        value_loss = F.mse_loss(values, returns)
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

        approx_kl = torch.mean(old_log_probs - log_probs).clamp_min(0).item()
        if self.adaptive_kl and self.reference_model is not None and ref_kl.item() > 0:
            self._update_kl_coef(float(ref_kl.item()))
        clip_fraction_value = float(clip_fraction.item())

        metrics = {
            "loss": float(loss.item()),
            "policy_loss": float(policy_loss.item()),
            "value_loss": float(value_loss.item()),
            "entropy": float(entropy.item()),
            "kl": float(approx_kl),
            "policy_objective": float(pg_objective.mean().item()),
            "kl_to_ref": float(ref_kl.item()),
            "kl_coef": float(self.kl_coef),
            "kl_penalty": float(kl_penalty.item()),
            "clip_fraction": clip_fraction_value,
            "reward_mean": float(flat.rewards.mean().item()),
        }
        return metrics

    def _setup_reference_model(
        self, reference_model: Optional[Union[nn.Module, str]]
    ) -> Optional[nn.Module]:
        if isinstance(reference_model, str):
            mode = reference_model.lower()
            if mode == "tie":
                return self.policy
            if mode not in {"copy", "auto"}:
                raise ValueError(
                    "reference_model string must be 'copy', 'tie', or 'auto'"
                )
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



__all__ = ["PPOTrainer"]
