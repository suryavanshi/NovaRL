from __future__ import annotations

import json
import queue
import time
from pathlib import Path

import pytest

from core.buffers import DataBuffer
from core.buffers.memory import TrajectoryBuffer
from core.types import TrajectoryBatch
from core.utils.logging import JsonlMetricsSink, MetricsAggregator, ProcessMetricsLogger
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.toy import ToyPromptEnvironment
from rewards.fake.basic import IdentityRewardManager

torch = pytest.importorskip("torch")

class DummyPolicy(torch.nn.Module):
    def __init__(self, observation_dim: int, action_dim: int) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(observation_dim, action_dim)
        self.value = torch.nn.Linear(observation_dim, 1)

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


def test_data_buffer_get_many() -> None:
    buffer = DataBuffer(capacity=4)
    batch1 = make_batch()
    batch2 = make_batch()
    buffer.put(batch1)
    buffer.put(batch2)
    items = buffer.get_many(min_items=2, max_items=2)
    assert len(items) == 2
    assert items[0] is batch1
    assert items[1] is batch2


def test_data_buffer_close_rejects_put() -> None:
    buffer = DataBuffer(capacity=1)
    buffer.close()
    with pytest.raises(RuntimeError):
        buffer.put(make_batch())


def test_data_buffer_snapshot_roundtrip() -> None:
    buffer: DataBuffer[int] = DataBuffer(capacity=4)
    buffer.put(1)
    buffer.put(2)
    snapshot = buffer.snapshot()
    assert snapshot == [1, 2]
    assert buffer.get() == 1
    assert buffer.get() == 2


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


def test_metrics_aggregator_records_events(tmp_path: Path) -> None:
    metrics_queue: queue.Queue = queue.Queue()
    sink_path = tmp_path / "metrics.jsonl"
    sink = JsonlMetricsSink(sink_path)
    history = []
    aggregator = MetricsAggregator(metrics_queue, sinks=[sink], history=history, poll_interval_s=0.1)
    aggregator.start()
    logger = ProcessMetricsLogger(metrics_queue, role="trainer")
    logger.log({"loss": 1.0}, step=1)
    time.sleep(0.2)
    aggregator.stop()
    assert history and history[0].metrics["loss"] == 1.0
    content = sink_path.read_text().strip().splitlines()
    assert content and json.loads(content[0])["metrics"]["loss"] == 1.0
