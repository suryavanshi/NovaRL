from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers.memory import TrajectoryBuffer
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.toy import ToyPromptEnvironment
from rewards.fake.basic import IdentityRewardManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ExperimentConfig:
    batch_size: int = 4
    horizon: int = 4
    observation_dim: int = 8
    action_dim: int = 4
    total_iterations: int = 10
    learning_rate: float = 3e-3


class SimplePolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 32) -> None:
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


def main(cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    device = torch.device("cpu")

    env = ToyPromptEnvironment(
        batch_size=cfg.batch_size,
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        max_turns=cfg.horizon,
        device=device,
    )
    policy = SimplePolicy(cfg.observation_dim, cfg.action_dim).to(device)
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
            "iteration=%d reward=%.3f kl=%.4f entropy=%.3f eps/s=%.2f",
            iteration,
            metrics["reward_mean"],
            metrics["kl"],
            metrics["entropy"],
            eps_per_sec,
        )

    logging.info(
        "Finished training %d iterations over %d episodes",
        cfg.total_iterations,
        total_episodes,
    )


if __name__ == "__main__":
    main()
