"""Hugging Face wrappers for NovaRL policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

import torch
from torch import nn


try:  # pragma: no cover - optional heavy dependency
    from transformers import PreTrainedModel, PretrainedConfig
    from transformers.modeling_outputs import ModelOutput
except Exception as exc:  # pragma: no cover - transformers is optional
    raise ImportError(
        "The Hugging Face policy helpers require the `transformers` package to be installed."
    ) from exc


@dataclass
class TinyPreferencePolicyOutput(ModelOutput):
    """Minimal output container matching Hugging Face conventions."""

    logits: torch.FloatTensor
    values: Optional[torch.FloatTensor] = None


class TinyPreferencePolicyConfig(PretrainedConfig):
    """Configuration for the tiny preference policy used in tutorials."""

    model_type = "novarl_tiny_policy"

    def __init__(
        self,
        *,
        observation_dim: int,
        action_dim: int,
        hidden_size: int = 64,
        activation: str = "tanh",
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.hidden_size = int(hidden_size)
        self.activation = activation


class TinyPreferencePolicyModel(PreTrainedModel):
    """Hugging Face compatible module wrapping :class:`TinyPreferencePolicy`."""

    config_class = TinyPreferencePolicyConfig

    def __init__(self, config: TinyPreferencePolicyConfig) -> None:
        super().__init__(config)
        activation = config.activation.lower()
        if activation == "tanh":
            act_cls = nn.Tanh
        elif activation == "relu":
            act_cls = nn.ReLU
        else:
            raise ValueError(f"Unsupported activation {activation!r} for TinyPreferencePolicyModel")

        self.encoder = nn.Sequential(
            nn.Linear(config.observation_dim, config.hidden_size),
            act_cls(),
            nn.Linear(config.hidden_size, config.hidden_size),
            act_cls(),
        )
        self.policy_head = nn.Linear(config.hidden_size, config.action_dim)
        self.value_head = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_features: torch.Tensor,
        *,
        return_dict: bool = True,
        **kwargs: Any,
    ) -> TinyPreferencePolicyOutput | tuple[torch.FloatTensor, torch.FloatTensor]:
        """Compute action logits and value estimates."""

        del kwargs  # Unused but retained for HF API compatibility.
        if input_features.dim() == 1:
            input_features = input_features.unsqueeze(0)
        hidden = self.encoder(input_features)
        logits = self.policy_head(hidden)
        values = self.value_head(hidden)

        if not return_dict:
            return logits, values
        return TinyPreferencePolicyOutput(logits=logits, values=values)

    def get_input_embeddings(self) -> nn.Module:
        # The tiny policy operates directly on dense features.
        raise NotImplementedError("TinyPreferencePolicyModel does not use token embeddings")

    @classmethod
    def from_state_dict(
        cls,
        state_dict: Mapping[str, torch.Tensor],
        *,
        observation_dim: int,
        action_dim: int,
        hidden_size: int,
        activation: str = "tanh",
    ) -> "TinyPreferencePolicyModel":
        """Instantiate the model from a raw ``state_dict`` and shape metadata."""

        config = TinyPreferencePolicyConfig(
            observation_dim=observation_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            activation=activation,
        )
        model = cls(config)
        model.load_state_dict(state_dict)
        return model

    def export_metadata(self) -> Dict[str, Any]:
        """Return lightweight metadata bundled alongside checkpoints."""

        return {
            "model_type": self.config.model_type,
            "observation_dim": self.config.observation_dim,
            "action_dim": self.config.action_dim,
            "hidden_size": self.config.hidden_size,
            "activation": self.config.activation,
        }

