from __future__ import annotations

import torch
from torch import nn

from core.buffers.memory import TrajectoryBuffer
from core.types import TrajectoryBatch
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.toy import ToyPromptEnvironment
from rewards.fake.basic import IdentityRewardManager


class DummyPolicy(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(observation_dim, action_dim)
        self.value = nn.Linear(observation_dim, 1)

    def forward(self, observations: torch.Tensor) -> dict[str, torch.Tensor]:
        logits = self.linear(observations)
        value = self.value(observations)
        return {"logits": logits, "value": value}


def make_batch(
    time_steps: int = 2,
    batch_size: int = 2,
    obs_dim: int = 3,
    action_dim: int = 2,
) -> TrajectoryBatch:
    observations = torch.randn(time_steps, batch_size, obs_dim)
    actions = torch.randint(0, action_dim, (time_steps, batch_size))
    log_probs = torch.randn(time_steps, batch_size)
    rewards = torch.randn(time_steps, batch_size)
    dones = torch.zeros(time_steps, batch_size)
    values = torch.randn(time_steps, batch_size)
    advantages = torch.randn(time_steps, batch_size)
    returns = torch.randn(time_steps, batch_size)
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


def test_trajectory_batch_flatten() -> None:
    batch = make_batch()
    flat = batch.flatten()
    assert flat.observations.shape[0] == batch.horizon * batch.batch_size
    assert flat.actions.shape == flat.log_probs.shape


def test_buffer_concat() -> None:
    buffer = TrajectoryBuffer(capacity=2)
    batch1 = make_batch()
    batch2 = make_batch()
    buffer.put(batch1)
    buffer.put(batch2)
    combined = buffer.get()
    assert combined.horizon == batch1.horizon + batch2.horizon


def test_rollout_engine_generates_batch() -> None:
    env = ToyPromptEnvironment(batch_size=2, observation_dim=4, action_dim=3, max_turns=2)
    policy = DummyPolicy(4, 3)
    reward_manager = IdentityRewardManager()
    engine = SynchronousRolloutEngine(
        env=env,
        policy=policy,
        reward_manager=reward_manager,
        horizon=2,
    )
    batch = engine.generate()
    assert batch.observations.shape[0] == 2
    assert batch.actions.shape == batch.rewards.shape
    assert batch.completed_episodes() >= 0
