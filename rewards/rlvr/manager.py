"""Reward manager that mixes multiple verifiable reward signals."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from core.interfaces import EnvStep
from rewards.base import RewardManagerBase
from rewards.rlvr.signals import VerifiableSignal


class VerifiableRewardManager(RewardManagerBase):
    """Combines several verifiable signals into a single reward tensor."""

    def __init__(
        self,
        signals: Sequence[VerifiableSignal],
        *,
        include_environment_reward: bool = False,
        mix: str = "sum",
        normalize_weights: bool = False,
    ) -> None:
        if not signals:
            raise ValueError("VerifiableRewardManager requires at least one signal")
        self.signals = list(signals)
        self.include_environment_reward = include_environment_reward
        if mix not in {"sum", "mean"}:
            raise ValueError("mix must be either 'sum' or 'mean'")
        self.mix = mix
        self.normalize_weights = normalize_weights
        self.latest_signal_values: dict[str, torch.Tensor] = {}
        self.latest_histograms: dict[str, dict[str, list[float]] | None] = {}

    def score(self, samples: Any) -> torch.Tensor:
        if isinstance(samples, EnvStep):
            infos = samples.infos
            base_reward = samples.rewards
            device = base_reward.device
            zeros = torch.zeros_like(base_reward, dtype=torch.float32)
        elif isinstance(samples, Mapping):
            infos = samples.get("infos")
            if infos is None:
                raise KeyError("Mapping samples must contain an 'infos' key")
            zeros = torch.zeros(len(infos), dtype=torch.float32)
            device = zeros.device
            base_reward = zeros
        else:
            raise TypeError("VerifiableRewardManager expects an EnvStep or mapping")

        if not isinstance(infos, Sequence):
            raise TypeError("infos must be a sequence of rollout metadata")
        total = torch.zeros(len(infos), dtype=torch.float32, device=device)
        weight_sum = 0.0
        self.latest_signal_values = {}
        self.latest_histograms = {}

        for signal in self.signals:
            result = signal(infos)
            weighted = result.weighted.to(device)
            total = total + weighted
            weight_sum += abs(signal.weight)
            self.latest_signal_values[signal.name] = result.raw.detach().cpu()
            self.latest_histograms[signal.name] = signal.latest_histogram

        if self.normalize_weights and weight_sum > 0:
            total = total / weight_sum

        if self.mix == "mean" and self.signals:
            total = total / float(len(self.signals))

        if self.include_environment_reward:
            total = total + base_reward.to(device)

        return total


__all__ = ["VerifiableRewardManager"]

