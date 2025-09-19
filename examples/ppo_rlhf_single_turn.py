from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers.memory import TrajectoryBuffer
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.single_turn import PreferenceDataset, SingleTurnPreferenceEnvironment
from rewards.fake.basic import IdentityRewardManager

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


@dataclass
class ExperimentConfig:
    batch_size: int = 8
    horizon: int = 1
    total_iterations: int = 80
    learning_rate: float = 3e-4
    clip_range: float | None = 0.2
    kl_coef: float = 0.05
    adaptive_kl: bool = True
    kl_target: float = 0.02
    kl_adaptation_speed: float = 1.5
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    checkpoint_dir: str = "checkpoints/ppo_rlhf_single_turn"


class TinyPreferencePolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_size: int = 64) -> None:
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


def build_tiny_dataset() -> PreferenceDataset:
    prompt_features = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.2, -0.3, 0.1],
            [0.0, 1.0, 0.0, -0.4, 0.3, 0.2],
            [0.0, 0.0, 1.0, 0.3, 0.1, -0.2],
            [0.5, 0.5, 0.0, -0.1, 0.2, 0.4],
        ],
        dtype=torch.float32,
    )
    action_rewards = torch.tensor(
        [
            [1.2, 0.0, -0.5],
            [0.0, 1.1, -0.6],
            [0.6, -0.2, 1.0],
            [-0.1, 0.4, 0.9],
        ],
        dtype=torch.float32,
    )
    prompt_texts: Sequence[str] = (
        "Prefer action 0",
        "Prefer action 1",
        "Action 2 wins",
        "Action 2 slightly better",
    )
    return PreferenceDataset(prompt_features=prompt_features, action_rewards=action_rewards, prompt_texts=prompt_texts)


def main(cfg: ExperimentConfig | None = None) -> None:
    cfg = cfg or ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(1234)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(1234)
    dataset = build_tiny_dataset()

    env = SingleTurnPreferenceEnvironment(dataset=dataset, batch_size=cfg.batch_size, device=device)
    policy = TinyPreferencePolicy(dataset.observation_dim, dataset.action_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)

    reward_manager = IdentityRewardManager()
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
        logging.info(
            (
                "iter=%d reward=%.3f obj=%.4f kl_old=%.4f kl_ref=%.4f kl_coef=%.4f "
                "entropy=%.3f eps/s=%.2f"
            ),
            iteration,
            metrics["reward_mean"],
            metrics["policy_objective"],
            metrics["kl"],
            metrics["kl_to_ref"],
            metrics["kl_coef"],
            metrics["entropy"],
            eps_per_sec,
        )

    checkpoint_dir = Path(cfg.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy_path = checkpoint_dir / "policy.pt"
    optimizer_path = checkpoint_dir / "optimizer.pt"
    torch.save({"model_state_dict": policy.state_dict()}, policy_path)
    torch.save({"optimizer_state_dict": optimizer.state_dict()}, optimizer_path)
    logging.info("Saved policy checkpoint to %s", policy_path)
    logging.info("Saved optimizer checkpoint to %s", optimizer_path)

    policy.eval()
    with torch.no_grad():
        logits = policy(dataset.prompt_features.to(device))["logits"]
        probs = logits.softmax(dim=-1).cpu()
    labels: Sequence[str]
    if dataset.prompt_texts is not None:
        labels = dataset.prompt_texts
    else:
        labels = [f"prompt_{i}" for i in range(len(probs))]
    for text, prob in zip(labels, probs):
        logging.info("prompt=%s probs=%s", text, [round(p.item(), 3) for p in prob])

    logging.info(
        "Finished training %d iterations over %d episodes", cfg.total_iterations, total_episodes
    )


if __name__ == "__main__":
    main()
