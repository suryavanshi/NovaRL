from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from core.interfaces import EnvStep
from envs.base import BatchedEnvironment


@dataclass
class PreferenceDataset:
    """Container holding prompt features and action rewards."""

    prompt_features: torch.Tensor
    action_rewards: torch.Tensor
    prompt_texts: Sequence[str] | None = None

    def __post_init__(self) -> None:
        if self.prompt_features.dim() != 2:
            raise ValueError("prompt_features must be a 2D tensor")
        if self.action_rewards.dim() != 2:
            raise ValueError("action_rewards must be a 2D tensor")
        if self.prompt_features.shape[0] != self.action_rewards.shape[0]:
            raise ValueError("Dataset must have matching number of prompts")

    @property
    def num_prompts(self) -> int:
        return int(self.prompt_features.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.prompt_features.shape[1])

    @property
    def action_dim(self) -> int:
        return int(self.action_rewards.shape[1])


class SingleTurnPreferenceEnvironment(BatchedEnvironment):
    """Environment that samples prompts and rewards single actions.

    Each episode consists of a single action. After an action is taken, a new
    prompt is sampled for the subsequent observation so rollouts can continue
    without explicit resets.
    """

    def __init__(
        self,
        dataset: PreferenceDataset,
        batch_size: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, device=device)
        self.dataset = dataset
        self.prompt_features = dataset.prompt_features.to(self.device)
        self.action_rewards = dataset.action_rewards.to(self.device)
        self.prompt_texts = dataset.prompt_texts
        self.current_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._sample_indices()

    def reset(self, batch_size: int | None = None) -> EnvStep:
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("SingleTurnPreferenceEnvironment has fixed batch size")
        self._sample_indices()
        observations = self.prompt_features[self.current_indices]
        zeros = torch.zeros(self.batch_size, device=self.device)
        infos: Sequence[dict[str, float]] = tuple(
            {"prompt_index": int(idx)} for idx in self.current_indices
        )
        return EnvStep(observations=observations, rewards=zeros, dones=zeros, infos=infos)

    def step(self, actions: torch.Tensor) -> EnvStep:
        if actions.shape[0] != self.batch_size:
            raise ValueError("Action batch size mismatch")
        if actions.dim() != 1:
            actions = actions.squeeze(-1)
        actions = actions.long()
        if torch.any(actions < 0) or torch.any(actions >= self.dataset.action_dim):
            raise ValueError("Action index out of bounds for preference dataset")
        prompt_indices = self.current_indices
        rewards = self.action_rewards[prompt_indices, actions]
        dones = torch.ones(self.batch_size, device=self.device)
        infos: Sequence[dict[str, float]] = tuple(
            {
                "prompt_index": int(p.item()),
                "action": int(a.item()),
                "reward": float(r.item()),
            }
            for p, a, r in zip(prompt_indices, actions, rewards)
        )
        self._sample_indices()
        next_obs = self.prompt_features[self.current_indices]
        return EnvStep(observations=next_obs, rewards=rewards, dones=dones, infos=infos)

    def _sample_indices(self) -> None:
        self.current_indices = torch.randint(
            0, self.dataset.num_prompts, (self.batch_size,), device=self.device
        )


__all__ = ["PreferenceDataset", "SingleTurnPreferenceEnvironment"]
