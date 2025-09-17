from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from torch.distributions import Categorical

from core.interfaces import RolloutEngine
from core.types import TrajectoryBatch
from core.utils.stats import compute_gae
from rewards.base import RewardManagerBase


class SynchronousRolloutEngine(RolloutEngine):
    """Collects rollouts synchronously from a batched environment."""

    def __init__(
        self,
        env,
        policy: nn.Module,
        reward_manager: RewardManagerBase,
        horizon: int,
        gamma: float = 0.99,
        lam: float = 0.95,
        device: Optional[torch.device] = None,
    ) -> None:
        self.env = env
        self.policy = policy
        self.reward_manager = reward_manager
        self.horizon = horizon
        self.gamma = gamma
        self.lam = lam
        self.device = device or torch.device("cpu")

    def generate(self, batch: Optional[TrajectoryBatch] = None) -> TrajectoryBatch:
        del batch  # Unused in the synchronous implementation.
        self.policy.eval()
        step = self.env.reset()
        observations = []
        actions = []
        log_probs = []
        rewards = []
        dones = []
        values = []

        with torch.no_grad():
            for _ in range(self.horizon):
                policy_out = self.policy(step.observations)
                logits = policy_out["logits"]
                value = policy_out["value"].squeeze(-1)
                dist = Categorical(logits=logits)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_step = self.env.step(action)
                reward = self.reward_manager.score(next_step)

                observations.append(step.observations)
                actions.append(action)
                log_probs.append(log_prob)
                rewards.append(reward)
                dones.append(next_step.dones)
                values.append(value)

                step = next_step

            final_value = self.policy(step.observations)["value"].squeeze(-1)

        observations_tensor = torch.stack(observations, dim=0)
        actions_tensor = torch.stack(actions, dim=0)
        log_probs_tensor = torch.stack(log_probs, dim=0)
        rewards_tensor = torch.stack(rewards, dim=0)
        dones_tensor = torch.stack(dones, dim=0)
        values_tensor = torch.stack(values, dim=0)
        bootstrap_values = torch.cat([values_tensor, final_value.unsqueeze(0)], dim=0)

        advantages = compute_gae(
            rewards=rewards_tensor,
            values=bootstrap_values,
            dones=dones_tensor,
            gamma=self.gamma,
            lam=self.lam,
        )
        returns = advantages + values_tensor

        return TrajectoryBatch(
            observations=observations_tensor,
            actions=actions_tensor,
            log_probs=log_probs_tensor,
            rewards=rewards_tensor,
            dones=dones_tensor,
            values=values_tensor,
            advantages=advantages,
            returns=returns,
        )


__all__ = ["SynchronousRolloutEngine"]
