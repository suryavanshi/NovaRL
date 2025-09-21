"""Tiny RLVR example using synthetic GSM8K-style math prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers.memory import TrajectoryBuffer
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.verifiable import VerifiablePromptDataset, VerifiablePromptEnvironment
from rewards.rlvr.manager import VerifiableRewardManager
from rewards.rlvr.signals import MathAnswerSignal, RegexMatchSignal

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ExperimentConfig:
    batch_size: int = 6
    horizon: int = 1
    total_iterations: int = 60
    learning_rate: float = 5e-4
    clip_range: float | None = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    kl_coef: float = 0.05
    kl_target: float = 0.02
    adaptive_kl: bool = True
    kl_adaptation_speed: float = 1.5


class TinyPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 128) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.encoder(observations)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return {"logits": logits, "value": value}


def build_dataset() -> VerifiablePromptDataset:
    prompts: Sequence[str] = (
        "Mia buys 3 apples and then 4 more. How many apples does she have in total?",
        "A class of 5 students each solves 3 problems. How many problems were solved?",
        "Sam had $12 and spent $7 on a book. How much money does he have now?",
        "A jar contains 8 red marbles and 5 blue marbles. How many marbles are there?",
        "Liam runs 400 meters every day for 6 days. What distance does he cover?",
        "Two numbers add up to 15. If the first is 9, what is the second?",
    )
    num_prompts = len(prompts)
    action_dim = 3
    prompt_features = torch.eye(num_prompts, dtype=torch.float32)
    completions: list[list[str]] = []
    metadata: list[dict[str, object]] = []
    answers = [7, 15, 5, 13, 2400, 6]
    incorrect_templates = [
        "We start by adding and end with the wrong value. Answer: {}",
        "The student multiplies incorrectly. Answer: {}",
        "Subtracting gives us a mistaken answer. Answer: {}",
    ]
    for idx, (prompt, answer) in enumerate(zip(prompts, answers)):
        wrong = [answer + 2, max(answer - 3, 0), answer + 10]
        texts = [
            f"We add carefully. #### {answer}\nAnswer: {answer}",
            incorrect_templates[idx % len(incorrect_templates)].format(wrong[0]),
            f"Another try concludes with Answer: {wrong[1]}",
        ]
        completions.append(texts)
        metadata.append(
            {
                "answer": float(answer),
                "format_pattern": r"Answer:\s*-?\d+",
            }
        )
    return VerifiablePromptDataset(
        prompt_features=prompt_features,
        completions=completions,
        prompt_texts=prompts,
        metadata=metadata,
    )


def build_reward_manager() -> VerifiableRewardManager:
    signals = [
        MathAnswerSignal(
            name="math_exact",
            answer_key="answer",
            completion_key="completion",
            correct_reward=1.0,
            incorrect_reward=0.0,
            weight=1.0,
        ),
        RegexMatchSignal(
            name="format_bonus",
            pattern_key="format_pattern",
            completion_key="completion",
            positive_reward=0.2,
            negative_reward=0.0,
            weight=0.5,
        ),
    ]
    return VerifiableRewardManager(signals=signals, normalize_weights=True)


def main(cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(2024)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2024)

    dataset = build_dataset()
    env = VerifiablePromptEnvironment(dataset=dataset, batch_size=cfg.batch_size, device=device)
    policy = TinyPolicy(dataset.observation_dim, dataset.action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    reward_manager = build_reward_manager()
    rollout_engine = SynchronousRolloutEngine(
        env=env,
        policy=policy,
        reward_manager=reward_manager,
        horizon=cfg.horizon,
    )
    buffer = TrajectoryBuffer(capacity=4)
    trainer = PPOTrainer(
        policy=policy,
        optimizer=optimizer,
        clip_range=cfg.clip_range,
        value_coef=cfg.value_coef,
        entropy_coef=cfg.entropy_coef,
        kl_coef=cfg.kl_coef,
        adaptive_kl=cfg.adaptive_kl,
        kl_target=cfg.kl_target,
        kl_adaptation_speed=cfg.kl_adaptation_speed,
        reference_model="copy",
    )
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

        signal_means = {
            name: float(values.mean())
            for name, values in reward_manager.latest_signal_values.items()
        }
        logging.info(
            (
                "iter=%d reward=%.3f obj=%.4f kl=%.4f kl_ref=%.4f entropy=%.3f eps/s=%.2f "
                "signals=%s"
            ),
            iteration,
            metrics["reward_mean"],
            metrics["policy_objective"],
            metrics["kl"],
            metrics["kl_to_ref"],
            metrics["entropy"],
            eps_per_sec,
            {k: round(v, 3) for k, v in signal_means.items()},
        )
        if iteration % 10 == 0 and reward_manager.latest_histograms:
            for name, hist in reward_manager.latest_histograms.items():
                if hist is None:
                    continue
                counts = [round(c, 2) for c in hist["counts"]]
                logging.info("hist_%s=%s", name, counts)

    logging.info("Finished RLVR math run over %d episodes", total_episodes)


if __name__ == "__main__":  # pragma: no cover - example script
    main()

