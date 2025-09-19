from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch


@dataclass
class TrajectoryBatch:
    """Container for a batch of rollouts.

    The batch is stored with a time-major layout (T, B, ...), where ``T`` is the
    rollout horizon and ``B`` is the number of parallel environments.
    """

    observations: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def __post_init__(self) -> None:
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        if self.observations.dim() < 2:
            raise ValueError(
                "TrajectoryBatch expects time-major tensors with at least 2 dimensions"
            )
        time_steps = self.observations.shape[0]
        tensors = {
            "actions": self.actions,
            "log_probs": self.log_probs,
            "rewards": self.rewards,
            "dones": self.dones,
            "values": self.values,
            "advantages": self.advantages,
            "returns": self.returns,
        }
        for name, tensor in tensors.items():
            if tensor.dim() < 1:
                raise ValueError(f"Tensor '{name}' must have at least one dimension")
            if tensor.shape[0] != time_steps:
                raise ValueError(
                    f"Tensor '{name}' has mismatched time dimension: {tensor.shape[0]} vs {time_steps}"
                )
        if self.observations.dim() >= 3:
            batch_size = self.observations.shape[1]
            for name, tensor in tensors.items():
                if tensor.dim() < 2:
                    raise ValueError(
                        f"Tensor '{name}' must include batch dimension when observations are time-major"
                    )
                if tensor.shape[1] != batch_size:
                    raise ValueError(
                        f"Tensor '{name}' has mismatched batch dimension: {tensor.shape[1]} vs {batch_size}"
                    )

    @property
    def horizon(self) -> int:
        return int(self.observations.shape[0])

    @property
    def batch_size(self) -> int:
        return int(self.observations.shape[1])

    def to(self, device: torch.device) -> "TrajectoryBatch":
        return TrajectoryBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            log_probs=self.log_probs.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )

    def detach(self) -> "TrajectoryBatch":
        return TrajectoryBatch(
            observations=self.observations.detach(),
            actions=self.actions.detach(),
            log_probs=self.log_probs.detach(),
            rewards=self.rewards.detach(),
            dones=self.dones.detach(),
            values=self.values.detach(),
            advantages=self.advantages.detach(),
            returns=self.returns.detach(),
        )

    def flatten(self) -> "TrajectoryBatch":
        """Flatten the time and batch dimensions."""

        t, b = self.horizon, self.batch_size
        new_shape = (t * b,)
        return TrajectoryBatch(
            observations=self.observations.reshape(new_shape + self.observations.shape[2:]),
            actions=self.actions.reshape(new_shape + self.actions.shape[2:]),
            log_probs=self.log_probs.reshape(new_shape),
            rewards=self.rewards.reshape(new_shape),
            dones=self.dones.reshape(new_shape),
            values=self.values.reshape(new_shape),
            advantages=self.advantages.reshape(new_shape),
            returns=self.returns.reshape(new_shape),
        )

    def mean_reward(self) -> float:
        return float(self.rewards.mean().item())

    def completed_episodes(self) -> int:
        return int(self.dones.sum().item())

    def clone(self) -> "TrajectoryBatch":
        return TrajectoryBatch(
            observations=self.observations.clone(),
            actions=self.actions.clone(),
            log_probs=self.log_probs.clone(),
            rewards=self.rewards.clone(),
            dones=self.dones.clone(),
            values=self.values.clone(),
            advantages=self.advantages.clone(),
            returns=self.returns.clone(),
        )

    @staticmethod
    def concat(batches: Sequence["TrajectoryBatch"]) -> "TrajectoryBatch":
        if not batches:
            raise ValueError("Cannot concatenate an empty list of batches")
        observations = torch.cat([b.observations for b in batches], dim=0)
        actions = torch.cat([b.actions for b in batches], dim=0)
        log_probs = torch.cat([b.log_probs for b in batches], dim=0)
        rewards = torch.cat([b.rewards for b in batches], dim=0)
        dones = torch.cat([b.dones for b in batches], dim=0)
        values = torch.cat([b.values for b in batches], dim=0)
        advantages = torch.cat([b.advantages for b in batches], dim=0)
        returns = torch.cat([b.returns for b in batches], dim=0)
        return TrajectoryBatch(
            observations=observations,
            actions=actions,
            log_probs=log_probs,
            rewards=rewards,
            dones=dones,
            values=values,
            advantages=advantages,
            returns=returns,
        )

    @classmethod
    def stack_from(cls, items: Iterable["TrajectoryBatch"]) -> "TrajectoryBatch":
        return cls.concat(list(items))


__all__ = ["TrajectoryBatch"]
