"""Train a tiny vision-language agent on a toy multiple-choice VQA task."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch

from algos.ppo.trainer import PPOTrainer
from core.buffers.memory import TrajectoryBuffer
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.vision_prompt_env import (
    VisionPromptDataset,
    VisionPromptEnvironment,
    VisionPromptSample,
    VisionPromptTokenCollator,
)
from examples.vlm_wrappers import TokenFusionPolicy
from rewards.fake.basic import IdentityRewardManager

try:  # pragma: no cover - optional dependency for lightweight testing
    from transformers import AutoTokenizer
except Exception:  # pragma: no cover - transformers is optional
    AutoTokenizer = None


class SimpleTokenizer:
    """Fallback whitespace tokenizer mirroring a tiny subset of HF tokenisers."""

    def __init__(self) -> None:
        self.token_to_id: dict[str, int] = {"<pad>": 0, "<unk>": 1}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.token_to_id)

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    def get_vocab(self) -> dict[str, int]:  # pragma: no cover - trivial
        return dict(self.token_to_id)

    def add_special_tokens(self, mapping: dict[str, Sequence[str]]) -> int:
        tokens = mapping.get("additional_special_tokens", [])
        added = 0
        for token in tokens:
            added += self._add_token(token)
        return added

    def add_tokens(self, tokens: Sequence[str]) -> int:  # pragma: no cover - helper
        added = 0
        for token in tokens:
            added += self._add_token(token)
        return added

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.token_to_id.get(token, self.token_to_id[self.unk_token])

    def encode(self, text: str) -> list[int]:
        tokens = text.strip().split()
        ids: list[int] = []
        for token in tokens:
            ids.append(self.convert_tokens_to_ids(token))
        return ids if ids else [self.token_to_id[self.unk_token]]

    def encode_with_new_tokens(self, text: str) -> list[int]:
        tokens = text.strip().split()
        ids: list[int] = []
        for token in tokens:
            self._add_token(token)
            ids.append(self.token_to_id[token])
        return ids if ids else [self.token_to_id[self.unk_token]]

    def __call__(self, texts, padding=True, truncation=True, return_tensors="pt"):
        del truncation  # Unused in the simple fallback implementation.
        sequences = [self.encode(text) for text in texts]
        max_len = max(len(seq) for seq in sequences) if sequences else 0
        pad_id = self.convert_tokens_to_ids(self.pad_token)
        input_ids = torch.full((len(sequences), max_len), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(sequences), max_len), dtype=torch.long)
        for row, seq in enumerate(sequences):
            input_ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            attention_mask[row, : len(seq)] = 1
        if return_tensors != "pt":  # pragma: no cover - defensive
            raise ValueError("SimpleTokenizer only supports return_tensors='pt'")
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def _add_token(self, token: str) -> int:
        if token not in self.token_to_id:
            self.token_to_id[token] = len(self.token_to_id)
            return 1
        return 0


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ExperimentConfig:
    batch_size: int = 4
    horizon: int = 1
    total_iterations: int = 50
    learning_rate: float = 3e-4
    embed_dim: int = 96
    hidden_size: int = 128
    image_size: int = 48


def _make_coloured_square(colour: tuple[int, int, int], size: int) -> np.ndarray:
    array = np.zeros((size, size, 3), dtype=np.float32)
    array[...] = np.array(colour, dtype=np.float32)
    return array


def build_dataset(image_size: int) -> VisionPromptDataset:
    colours: Sequence[tuple[str, tuple[int, int, int]]] = (
        ("red", (220, 20, 60)),
        ("green", (50, 205, 50)),
        ("blue", (65, 105, 225)),
        ("yellow", (255, 215, 0)),
    )
    choices: Sequence[str] = tuple(name for name, _ in colours)
    samples: list[VisionPromptSample] = []
    for answer_index, (name, rgb) in enumerate(colours):
        image = _make_coloured_square(rgb, image_size)
        sample = VisionPromptSample(
            image=image,
            question="What colour is the square?",
            choices=choices,
            answer_index=answer_index,
            metadata={"colour": name},
        )
        samples.append(sample)
    return VisionPromptDataset(samples)


def main(cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)

    dataset = build_dataset(cfg.image_size)
    if AutoTokenizer is not None:
        tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2")
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    else:
        tokenizer = SimpleTokenizer()
        for sample in dataset:
            tokenizer.encode_with_new_tokens(sample.question)
            for choice in sample.choices:
                tokenizer.encode_with_new_tokens(choice)
        option_tokens = [f"({chr(ord('A') + idx)})" for idx in range(dataset.action_dim)]
        tokenizer.add_tokens(["Question:", "Choices:", "Answer:", *option_tokens])
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})
    collator = VisionPromptTokenCollator(dataset, tokenizer, image_token="<image>")

    env = VisionPromptEnvironment(dataset=dataset, batch_size=cfg.batch_size, device=device)
    policy = TokenFusionPolicy(
        dataset=dataset,
        collator=collator,
        embed_dim=cfg.embed_dim,
        hidden_size=cfg.hidden_size,
    ).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    reward_manager = IdentityRewardManager()
    rollout_engine = SynchronousRolloutEngine(
        env=env,
        policy=policy,
        reward_manager=reward_manager,
        horizon=cfg.horizon,
    )
    buffer = TrajectoryBuffer(capacity=4)
    trainer = PPOTrainer(policy=policy, optimizer=optimizer)
    rate_tracker = RateTracker(window_seconds=30.0)

    total_episodes = 0
    for iteration in range(cfg.total_iterations):
        trajectories = rollout_engine.generate()
        buffer.put(trajectories)
        batch = buffer.get()
        metrics = trainer.step(batch)
        completed = trajectories.completed_episodes()
        total_episodes += completed
        rate_tracker.update(completed)
        eps_per_sec = rate_tracker.rate()
        logging.info(
            "iter=%d reward=%.3f kl=%.4f entropy=%.3f eps/s=%.2f",
            iteration,
            metrics["reward_mean"],
            metrics["kl"],
            metrics["entropy"],
            eps_per_sec,
        )

    policy.eval()
    with torch.no_grad():
        indices = torch.arange(len(dataset), device=device)
        logits = policy(indices)["logits"].softmax(dim=-1).cpu()
    for sample, probs in zip(dataset, logits):
        logging.info(
            "question=%s answer=%s probs=%s",
            sample.question,
            sample.choices[sample.answer_index],
            [round(p.item(), 3) for p in probs],
        )

    logging.info(
        "Finished training %d iterations over %d episodes", cfg.total_iterations, total_episodes
    )


if __name__ == "__main__":  # pragma: no cover - script entry point
    main()

