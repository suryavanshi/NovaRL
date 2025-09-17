from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

from .types import TrajectoryBatch


@dataclass
class EnvStep:
    observations: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    infos: Sequence[Mapping[str, Any]]


class Environment(abc.ABC):
    """Abstract environment supporting batched interactions."""

    @abc.abstractmethod
    def reset(self, batch_size: Optional[int] = None) -> EnvStep:
        """Reset the environment."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> EnvStep:
        """Advance the environment by one step for the provided actions."""


class RolloutEngine(abc.ABC):
    """Adapter that turns policy calls into trajectory batches."""

    @abc.abstractmethod
    def generate(self, batch: Optional[TrajectoryBatch] = None) -> TrajectoryBatch:
        """Produce a batch of trajectories."""


class Trainer(abc.ABC):
    """Base class for optimizing policies from trajectory data."""

    @abc.abstractmethod
    def step(self, batch: TrajectoryBatch) -> Mapping[str, float]:
        """Run one optimization step and return scalar metrics."""


class Buffer(abc.ABC):
    """Storage abstraction for rollouts."""

    @abc.abstractmethod
    def put(self, batch: TrajectoryBatch) -> None: ...

    @abc.abstractmethod
    def get(self) -> TrajectoryBatch: ...

    @abc.abstractmethod
    def __len__(self) -> int: ...


class RewardManager(abc.ABC):
    """Responsible for computing rewards from raw samples."""

    @abc.abstractmethod
    def score(self, samples: Any) -> torch.Tensor: ...


__all__ = [
    "EnvStep",
    "Environment",
    "RolloutEngine",
    "Trainer",
    "Buffer",
    "RewardManager",
]
