from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn

from algos.grpo import GRPOTrainer, GRPOGroupingConfig
from algos.ppo import PPOTrainer
from core.distributed import MoERouterConfig, ParallelLayout
from core.utils.advantages import GAEAdvantageEstimator
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.toy import ToyPromptEnvironment
from rewards.fake.basic import IdentityRewardManager

logger = logging.getLogger(__name__)


@dataclass
class MoEDemoConfig:
    batch_size: int = 4
    horizon: int = 4
    observation_dim: int = 16
    action_dim: int = 6
    hidden_size: int = 64
    total_iterations: int = 6
    learning_rate: float = 3e-3
    algorithm: str = "ppo"
    grpo_group_size: int = 2
    num_experts: int = 4
    top_k: int = 2
    capacity_factor: float = 1.25
    world_size: int = 8
    tensor_parallel_size: int = 2
    pipeline_parallel_size: int = 2
    expert_parallel_size: int = 2
    gamma: float = 0.99
    lam: float = 0.95


class TinyMoELayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.GELU(),
                    nn.Linear(hidden_size, hidden_size),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        logits = self.router(inputs)
        weights = torch.softmax(logits, dim=-1)
        top_values, top_indices = torch.topk(weights, k=self.top_k, dim=-1)
        mask = torch.zeros_like(weights)
        mask.scatter_(-1, top_indices, 1.0)
        masked_weights = weights * mask
        masked_weights = masked_weights / (masked_weights.sum(dim=-1, keepdim=True) + 1e-9)
        expert_outputs = torch.stack([expert(inputs) for expert in self.experts], dim=1)
        combined = torch.einsum("b e, b e h -> b h", masked_weights, expert_outputs)
        return combined, weights


class TinyMoEPolicy(nn.Module):
    def __init__(
        self,
        observation_dim: int,
        action_dim: int,
        hidden_size: int,
        num_experts: int,
        top_k: int,
    ) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, hidden_size),
            nn.GELU(),
        )
        self.moe = TinyMoELayer(hidden_size, num_experts, top_k)
        self.norm = nn.LayerNorm(hidden_size)
        self.policy_head = nn.Linear(hidden_size, action_dim)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        encoded = self.encoder(observations)
        moe_out, router = self.moe(encoded)
        hidden = self.norm(moe_out + encoded)
        logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        return {"logits": logits, "value": value, "router_probs": router}


def _build_trainer(cfg: MoEDemoConfig, policy: nn.Module, optimizer: torch.optim.Optimizer):
    algorithm = cfg.algorithm.lower()
    if algorithm == "ppo":
        return PPOTrainer(policy=policy, optimizer=optimizer)
    if algorithm == "grpo":
        grouping = GRPOGroupingConfig(group_size=cfg.grpo_group_size)
        return GRPOTrainer(policy=policy, optimizer=optimizer, grouping=grouping)
    raise ValueError(f"Unsupported algorithm '{cfg.algorithm}'")


def _describe_parallel_layout(cfg: MoEDemoConfig) -> tuple[ParallelLayout, MoERouterConfig]:
    layout = ParallelLayout.from_world_size(
        total_world_size=cfg.world_size,
        tensor_parallel_size=cfg.tensor_parallel_size,
        pipeline_parallel_size=cfg.pipeline_parallel_size,
        expert_parallel_size=cfg.expert_parallel_size,
    )
    router = MoERouterConfig(
        num_experts=cfg.num_experts,
        top_k=cfg.top_k,
        capacity_factor=cfg.capacity_factor,
    )
    return layout, router


def main(cfg: MoEDemoConfig | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    cfg = cfg or MoEDemoConfig()
    layout, router = _describe_parallel_layout(cfg)
    layout.validate(total_world_size=cfg.world_size)
    logger.info("MoE parallel layout: %s", layout.describe())
    logger.info("Router config: num_experts=%d top_k=%d capacity=%.2f", router.num_experts, router.top_k, router.capacity_factor)

    device = torch.device("cpu")
    env = ToyPromptEnvironment(
        batch_size=cfg.batch_size,
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        max_turns=cfg.horizon,
        device=device,
    )
    policy = TinyMoEPolicy(
        cfg.observation_dim,
        cfg.action_dim,
        cfg.hidden_size,
        cfg.num_experts,
        cfg.top_k,
    ).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.learning_rate)
    estimator = GAEAdvantageEstimator(gamma=cfg.gamma, lam=cfg.lam)

    rollout_engine = SynchronousRolloutEngine(
        env=env,
        policy=policy,
        reward_manager=IdentityRewardManager(),
        horizon=cfg.horizon,
        advantage_estimator=estimator,
    )
    trainer = _build_trainer(cfg, policy, optimizer)
    rate_tracker = RateTracker(window_seconds=30.0)

    for iteration in range(cfg.total_iterations):
        batch = rollout_engine.generate()
        metrics = trainer.step(batch)
        completed = batch.completed_episodes()
        rate_tracker.update(completed)
        logger.info(
            "iter=%d reward=%.3f kl=%.4f entropy=%.3f eps/s=%.2f",
            iteration,
            metrics.get("reward_mean", 0.0),
            metrics.get("kl", 0.0),
            metrics.get("entropy", 0.0),
            rate_tracker.rate(),
        )


if __name__ == "__main__":
    main()


__all__ = ["MoEDemoConfig", "TinyMoEPolicy", "main"]

