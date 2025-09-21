"""Prompt environment that surfaces verifiable reward metadata."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Mapping, MutableMapping, Sequence

import torch

from core.interfaces import EnvStep
from envs.base import BatchedEnvironment


@dataclass
class VerifiablePromptDataset:
    """Container describing prompts with candidate completions and metadata."""

    prompt_features: torch.Tensor
    completions: Sequence[Sequence[str]]
    prompt_texts: Sequence[str]
    metadata: Sequence[Mapping[str, object]]

    def __post_init__(self) -> None:
        if self.prompt_features.dim() != 2:
            raise ValueError("prompt_features must be a 2D tensor")
        num_prompts = self.prompt_features.shape[0]
        if len(self.completions) != num_prompts:
            raise ValueError("completions must match number of prompts")
        if len(self.prompt_texts) != num_prompts:
            raise ValueError("prompt_texts must match number of prompts")
        if len(self.metadata) != num_prompts:
            raise ValueError("metadata must match number of prompts")
        lengths = {len(options) for options in self.completions}
        if not lengths:
            raise ValueError("Dataset must contain at least one prompt")
        if len(lengths) != 1:
            raise ValueError("All prompts must have the same number of completions")

    @property
    def num_prompts(self) -> int:
        return int(self.prompt_features.shape[0])

    @property
    def observation_dim(self) -> int:
        return int(self.prompt_features.shape[1])

    @property
    def action_dim(self) -> int:
        return len(self.completions[0])


class VerifiablePromptEnvironment(BatchedEnvironment):
    """Batched environment that defers reward computation to verifiers."""

    def __init__(
        self,
        dataset: VerifiablePromptDataset,
        batch_size: int,
        *,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(batch_size=batch_size, device=device)
        self.dataset = dataset
        self.prompt_features = dataset.prompt_features.to(self.device)
        self.current_indices = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self._sample_indices()

    def reset(self, batch_size: int | None = None) -> EnvStep:
        if batch_size is not None and batch_size != self.batch_size:
            raise ValueError("VerifiablePromptEnvironment has fixed batch size")
        self._sample_indices()
        observations = self.prompt_features[self.current_indices]
        zeros = torch.zeros(self.batch_size, device=self.device)
        infos: Sequence[MutableMapping[str, object]] = tuple(
            self._make_info(idx, None) for idx in self.current_indices
        )
        return EnvStep(observations=observations, rewards=zeros, dones=zeros, infos=infos)

    def step(self, actions: torch.Tensor) -> EnvStep:
        if actions.shape[0] != self.batch_size:
            raise ValueError("Action batch size mismatch")
        actions = actions.long().view(-1)
        if torch.any(actions < 0) or torch.any(actions >= self.dataset.action_dim):
            raise ValueError("Action index out of bounds for verifiable dataset")
        prompt_indices = self.current_indices
        infos: Sequence[MutableMapping[str, object]] = tuple(
            self._make_info(int(p.item()), int(a.item()))
            for p, a in zip(prompt_indices, actions)
        )
        rewards = torch.zeros(self.batch_size, device=self.device)
        dones = torch.ones(self.batch_size, device=self.device)
        self._sample_indices()
        next_obs = self.prompt_features[self.current_indices]
        return EnvStep(observations=next_obs, rewards=rewards, dones=dones, infos=infos)

    def _make_info(
        self, prompt_index: int, action: int | None
    ) -> MutableMapping[str, object]:
        base: MutableMapping[str, object] = {
            "prompt_index": prompt_index,
            "prompt": self.dataset.prompt_texts[prompt_index],
        }
        meta = copy.deepcopy(self.dataset.metadata[prompt_index])
        if action is not None:
            completion = self.dataset.completions[prompt_index][action]
            base["completion_index"] = action
            base["completion"] = completion
        base.update(meta)
        return base

    def _sample_indices(self) -> None:
        self.current_indices = torch.randint(
            0, self.dataset.num_prompts, (self.batch_size,), device=self.device
        )


__all__ = ["VerifiablePromptDataset", "VerifiablePromptEnvironment"]

