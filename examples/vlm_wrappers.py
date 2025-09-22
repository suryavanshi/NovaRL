"""Utility vision-language policy wrappers used in the VQA PPO example."""

from __future__ import annotations

import torch
from torch import nn

from envs.vision_prompt_env import (
    VisionPromptDataset,
    VisionPromptEmbeddingCollator,
    VisionPromptTokenCollator,
)


class EmbeddingFusionPolicy(nn.Module):
    """Tiny policy that consumes pre-computed text and image embeddings."""

    def __init__(
        self,
        dataset: VisionPromptDataset,
        collator: VisionPromptEmbeddingCollator,
        *,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.collator = collator

        sample = collator([0])
        text_dim = sample["text_embeddings"].shape[-1]
        image_dim = sample["image_embeddings"].shape[-1]

        self.text_proj = nn.Linear(text_dim, hidden_size)
        self.image_proj = nn.Linear(image_dim, hidden_size)
        self.activation = nn.Tanh()
        self.policy_head = nn.Linear(hidden_size, dataset.action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        device = observations.device
        indices = observations.view(-1).tolist()
        batch = self.collator(indices)
        text_embeddings = batch["text_embeddings"].to(device)
        image_embeddings = batch["image_embeddings"].to(device)

        hidden = self.activation(
            self.text_proj(text_embeddings) + self.image_proj(image_embeddings)
        )
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return {"logits": logits, "value": value}


class TokenFusionPolicy(nn.Module):
    """Policy wrapper for token-based prompts with an injected image token."""

    def __init__(
        self,
        dataset: VisionPromptDataset,
        collator: VisionPromptTokenCollator,
        *,
        embed_dim: int = 128,
        hidden_size: int = 128,
    ) -> None:
        super().__init__()
        self.dataset = dataset
        self.collator = collator
        self.image_token_id = collator.image_token_id

        vocab_size = collator.tokenizer.vocab_size
        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.image_projector = nn.Linear(32, embed_dim)
        self.fusion_proj = nn.Linear(embed_dim, hidden_size)
        self.activation = nn.Tanh()
        self.policy_head = nn.Linear(hidden_size, dataset.action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def _inject_image_embeddings(
        self, input_ids: torch.Tensor, token_embeddings: torch.Tensor, image_embeds: torch.Tensor
    ) -> torch.Tensor:
        if token_embeddings.dim() != 3:
            raise ValueError("token_embeddings must be [batch, seq, dim]")
        mask = (input_ids == self.image_token_id).unsqueeze(-1)
        if mask.any():
            expanded = image_embeds.unsqueeze(1).expand_as(token_embeddings)
            token_embeddings = torch.where(mask, expanded, token_embeddings)
        return token_embeddings

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        device = observations.device
        indices = observations.view(-1).tolist()
        batch = self.collator(indices)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device).float()
        pixel_values = batch["pixel_values"].to(device)

        token_embeds = self.token_embeddings(input_ids)
        image_feats = self.image_encoder(pixel_values).flatten(start_dim=1)
        image_embeds = self.image_projector(image_feats)
        fused_tokens = self._inject_image_embeddings(input_ids, token_embeds, image_embeds)

        masked = fused_tokens * attention_mask.unsqueeze(-1)
        pooled = masked.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        hidden = self.activation(self.fusion_proj(pooled))
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return {"logits": logits, "value": value}


__all__ = ["EmbeddingFusionPolicy", "TokenFusionPolicy"]

