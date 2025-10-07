import torch
from torch import nn

from algos.grpo import GRPOGroupingConfig, GRPOTrainer
from core.types import TrajectoryBatch


class _TinyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.linear(observations)
        logits = hidden
        value = hidden.sum(dim=-1, keepdim=True)
        return {"logits": logits, "value": value}


def _make_batch(returns: torch.Tensor) -> TrajectoryBatch:
    t, b = returns.shape
    observations = torch.randn(t, b, 3)
    actions = torch.zeros(t, b, dtype=torch.long)
    log_probs = torch.zeros(t, b)
    rewards = torch.zeros(t, b)
    dones = torch.zeros(t, b)
    values = torch.zeros(t, b)
    advantages = torch.zeros(t, b)
    return TrajectoryBatch(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        dones=dones,
        values=values,
        advantages=advantages,
        returns=returns,
    )


def test_group_relative_advantages_zero_mean():
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, grouping=GRPOGroupingConfig(group_size=2))
    returns = torch.tensor([[1.0, 4.0, 5.0, 9.0]])
    adv = trainer._group_relative_advantages(returns.flatten())
    assert torch.isclose(adv[:2].sum(), torch.tensor(0.0), atol=1e-6)
    assert torch.isclose(adv[2:].sum(), torch.tensor(0.0), atol=1e-6)


def test_grpo_step_runs():
    policy = _TinyPolicy()
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    trainer = GRPOTrainer(policy=policy, optimizer=optimizer, grouping=GRPOGroupingConfig(group_size=2))
    returns = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    batch = _make_batch(returns)
    metrics = trainer.step(batch)
    assert "loss" in metrics
    assert metrics["advantage_norm"] >= 0.0
