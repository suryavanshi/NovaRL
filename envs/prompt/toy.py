from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from core.interfaces import EnvStep
from envs.base import BatchedEnvironment


@dataclass
class ToyPromptState:
    targets: torch.Tensor
    turns: torch.Tensor


class ToyPromptEnvironment(BatchedEnvironment):
    """A minimal prompt-style environment with synthetic rewards."""

    def __init__(
        self,
        batch_size: int,
        observation_dim: int,
        action_dim: int,
        max_turns: int = 4,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, device=device)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.max_turns = max_turns
        self.state = self._initial_state()

    def _initial_state(self) -> ToyPromptState:
        targets = torch.randint(0, self.action_dim, (self.batch_size,), device=self.device)
        turns = torch.zeros(self.batch_size, device=self.device)
        return ToyPromptState(targets=targets, turns=turns)

    def reset(self, batch_size: int | None = None) -> EnvStep:
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("ToyPromptEnvironment does not support changing batch size at runtime")
        self.state = self._initial_state()
        observations = torch.randn(self.batch_size, self.observation_dim, device=self.device)
        zeros = torch.zeros(self.batch_size, device=self.device)
        infos: Sequence[dict[str, float]] = tuple({} for _ in range(self.batch_size))
        return EnvStep(observations=observations, rewards=zeros, dones=zeros, infos=infos)

    def step(self, actions: torch.Tensor) -> EnvStep:
        if actions.shape[0] != self.batch_size:
            raise ValueError("Action batch size mismatch")
        rewards = (actions == self.state.targets).float()
        self.state.turns += 1
        dones = (self.state.turns >= self.max_turns).float()
        next_obs = torch.randn(self.batch_size, self.observation_dim, device=self.device)
        # When an episode ends, refresh target and reset turn counter.
        done_mask = dones.bool()
        if done_mask.any():
            self.state.targets[done_mask] = torch.randint(
                0, self.action_dim, (done_mask.sum(),), device=self.device
            )
            self.state.turns[done_mask] = 0
            next_obs[done_mask] = torch.randn(
                done_mask.sum(), self.observation_dim, device=self.device
            )
        infos: Sequence[dict[str, float]] = tuple({"reward": float(r.item())} for r in rewards)
        return EnvStep(observations=next_obs, rewards=rewards, dones=dones, infos=infos)


__all__ = ["ToyPromptEnvironment"]
