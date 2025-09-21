"""RLVR example that rewards code solutions via unit tests and learned signals."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers.memory import TrajectoryBuffer
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.verifiable import VerifiablePromptDataset, VerifiablePromptEnvironment
from rewards.rlvr.manager import VerifiableRewardManager
from rewards.rlvr.signals import HTTPRewardModelSignal, RegexMatchSignal, UnitTestSignal

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ExperimentConfig:
    batch_size: int = 4
    horizon: int = 1
    total_iterations: int = 80
    learning_rate: float = 3e-4
    clip_range: float | None = 0.2
    entropy_coef: float = 0.02
    value_coef: float = 0.6
    kl_coef: float = 0.05
    kl_target: float = 0.02
    adaptive_kl: bool = True
    kl_adaptation_speed: float = 1.5


class TinyPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 96) -> None:
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


def _length_prior_client(scale: float = 1.0) -> Callable[[dict[str, object]], Sequence[float]]:
    def scorer(payload: dict[str, object]) -> list[float]:
        completions = payload.get("completions", [])
        scores: list[float] = []
        for text in completions:
            length = len(str(text))
            score = max(0.0, 1.0 - (length / (200.0 * scale)))
            scores.append(score)
        return scores

    return scorer


def build_dataset() -> VerifiablePromptDataset:
    prompts: Sequence[str] = (
        "Implement solve(x: int) -> int that returns x squared.",
        "Write solve(nums: list[int]) that returns the maximum element.",
        "Create solve(n: int) that returns the sum of integers from 1 to n.",
        "Write solve(words: list[str]) returning the concatenation separated by spaces.",
    )
    prompt_features = torch.eye(len(prompts), dtype=torch.float32)
    completions: list[list[str]] = []
    metadata: list[dict[str, object]] = []

    # Candidate completions for each prompt (APPS-style multiple proposals).
    completions.append(
        [
            "def solve(x):\n    return x * x\n",
            "def solve(x):\n    return x + x\n",
            "def solve(x):\n    return x ** 3\n",
        ]
    )
    metadata.append(
        {
            "entry_point": "solve",
            "tests": (
                {"args": [2], "expected": 4},
                {"args": [5], "expected": 25},
            ),
            "format_pattern": r"def\s+solve",
        }
    )

    completions.append(
        [
            "def solve(nums):\n    return max(nums) if nums else None\n",
            "def solve(nums):\n    return sum(nums)\n",
            "def solve(nums):\n    return min(nums)\n",
        ]
    )
    metadata.append(
        {
            "entry_point": "solve",
            "tests": (
                {"args": [[1, 4, 2]], "expected": 4},
                {"args": [[-5, -2, -9]], "expected": -2},
            ),
            "format_pattern": r"def\s+solve",
        }
    )

    completions.append(
        [
            "def solve(n):\n    return n * (n + 1) // 2\n",
            "def solve(n):\n    total = 0\n    for i in range(n):\n        total += i\n    return total\n",
            "def solve(n):\n    return n\n",
        ]
    )
    metadata.append(
        {
            "entry_point": "solve",
            "tests": (
                {"args": [3], "expected": 6},
                {"args": [1], "expected": 1},
                {"args": [10], "expected": 55},
            ),
            "format_pattern": r"def\s+solve",
        }
    )

    completions.append(
        [
            "def solve(words):\n    return ' '.join(words)\n",
            "def solve(words):\n    return ''.join(words)\n",
            "def solve(words):\n    return words[0] if words else ''\n",
        ]
    )
    metadata.append(
        {
            "entry_point": "solve",
            "tests": (
                {"args": [["hello", "world"]], "expected": "hello world"},
                {"args": [["Nova", "RL"]], "expected": "Nova RL"},
            ),
            "format_pattern": r"def\s+solve",
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
        UnitTestSignal(
            name="unit",
            completion_key="completion",
            tests_key="tests",
            entry_point_key="entry_point",
            success_reward=1.0,
            failure_reward=0.0,
            weight=1.0,
        ),
        RegexMatchSignal(
            name="signature",
            pattern_key="format_pattern",
            completion_key="completion",
            positive_reward=0.1,
            negative_reward=0.0,
            weight=0.3,
        ),
        HTTPRewardModelSignal(
            name="length_prior",
            endpoint="http://localhost:9999/reward",
            completion_key="completion",
            prompt_key="prompt",
            client=_length_prior_client(scale=1.5),
            weight=0.2,
        ),
    ]
    return VerifiableRewardManager(signals=signals, normalize_weights=True)


def main(cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

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

    logging.info("Finished RLVR code run over %d episodes", total_episodes)


if __name__ == "__main__":  # pragma: no cover - example script
    main()

