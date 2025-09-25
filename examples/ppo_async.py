from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers import DataBuffer
from core.types import TrajectoryBatch
from core.utils.timing import RateTracker
from engines.sync.sync_engine import SynchronousRolloutEngine
from envs.prompt.toy import ToyPromptEnvironment
from rewards.fake.basic import IdentityRewardManager

logger = logging.getLogger(__name__)


@dataclass
class AsyncExperimentConfig:
    """Configuration used by the asynchronous PPO demo."""

    batch_size: int = 8
    horizon: int = 4
    observation_dim: int = 8
    action_dim: int = 4
    total_updates: int = 40
    learning_rate: float = 3e-3
    num_workers: int = 2
    buffer_capacity: int = 8
    min_batches_per_update: int = 2
    weight_sync_interval: int = 4
    staleness_window: int = 50
    queue_timeout_s: float = 30.0
    colocate_rollouts: bool = True
    rollout_device: str | None = None
    trainer_device: str | None = None
    seed: int = 1234


class SimplePolicy(nn.Module):
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


@dataclass
class StampedBatch:
    batch: TrajectoryBatch
    policy_version: int
    created_at: float


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _policy_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in policy.state_dict().items()}


def _load_policy_state(policy: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    policy.load_state_dict(state_dict)


def _make_rollout_components(
    cfg: AsyncExperimentConfig, device: torch.device
) -> tuple[nn.Module, SynchronousRolloutEngine]:
    env = ToyPromptEnvironment(
        batch_size=cfg.batch_size,
        observation_dim=cfg.observation_dim,
        action_dim=cfg.action_dim,
        max_turns=cfg.horizon,
        device=device,
    )
    policy = SimplePolicy(cfg.observation_dim, cfg.action_dim).to(device)
    reward_manager = IdentityRewardManager()
    engine = SynchronousRolloutEngine(
        env=env,
        policy=policy,
        reward_manager=reward_manager,
        horizon=cfg.horizon,
    )
    return policy, engine


def _rollout_worker(
    worker_id: int,
    cfg: AsyncExperimentConfig,
    data_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
    initial_state: dict[str, torch.Tensor],
    device: str,
) -> None:
    _seed_all(cfg.seed + worker_id)
    torch.set_num_threads(1)

    rollout_device = torch.device(device)
    policy, engine = _make_rollout_components(cfg, rollout_device)
    _load_policy_state(policy, initial_state)

    rate_tracker = RateTracker(window_seconds=60.0)
    policy_version = 0

    while not stop_event.is_set():
        try:
            while True:
                update = weight_queue.get_nowait()
                policy_version = update["version"]
                _load_policy_state(policy, update["weights"])
        except queue.Empty:
            pass

        trajectories = engine.generate()
        cpu_batch = trajectories.detach().to(torch.device("cpu"))
        stamped = StampedBatch(
            batch=cpu_batch,
            policy_version=policy_version,
            created_at=time.time(),
        )
        try:
            data_queue.put(stamped, timeout=cfg.queue_timeout_s)
            rate_tracker.update(cpu_batch.completed_episodes())
        except queue.Full:
            logger.warning("worker %d data queue full; dropping rollout", worker_id)
            continue

    logger.info("worker %d stopping; final eps/s=%.2f", worker_id, rate_tracker.rate())


def _query_gpu_util(device_index: int = 0) -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        import pynvml  # type: ignore

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        return float(util.gpu)
    except Exception:  # pragma: no cover - best effort for optional dependency
        return None


def _concatenate_batches(items: Iterable[StampedBatch]) -> TrajectoryBatch:
    batches = [item.batch for item in items]
    if len(batches) == 1:
        return batches[0]
    return TrajectoryBatch.concat(batches)


def _maybe_broadcast_weights(
    policy: nn.Module,
    weight_queues: Sequence[mp.Queue],
    version: int,
) -> None:
    payload = {"version": version, "weights": _policy_state_dict(policy)}
    for q in weight_queues:
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        q.put(payload)


def run_async_training(cfg: AsyncExperimentConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting asynchronous PPO example with config: %s", cfg)
    _seed_all(cfg.seed)

    trainer_device = torch.device(
        cfg.trainer_device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    rollout_device = (
        trainer_device
        if cfg.colocate_rollouts
        else torch.device(cfg.rollout_device or "cpu")
    )

    policy = SimplePolicy(cfg.observation_dim, cfg.action_dim).to(trainer_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.learning_rate)
    trainer = PPOTrainer(policy=policy, optimizer=optimizer)

    ctx = mp.get_context("spawn")
    data_queue = ctx.Queue(cfg.buffer_capacity)
    data_buffer = DataBuffer(capacity=cfg.buffer_capacity, queue_obj=data_queue)
    stop_event = ctx.Event()
    weight_queues: List[mp.Queue] = [ctx.Queue(maxsize=1) for _ in range(cfg.num_workers)]

    workers: List[mp.Process] = []
    initial_state = _policy_state_dict(policy)
    for worker_id in range(cfg.num_workers):
        proc = ctx.Process(
            target=_rollout_worker,
            args=(
                worker_id,
                cfg,
                data_queue,
                weight_queues[worker_id],
                stop_event,
                initial_state,
                str(rollout_device),
            ),
            name=f"rollout-worker-{worker_id}",
        )
        proc.daemon = True
        proc.start()
        workers.append(proc)

    _maybe_broadcast_weights(policy, weight_queues, version=0)

    def _shutdown(*_: object) -> None:
        logger.info("Received shutdown signal; stopping workers")
        stop_event.set()
        data_buffer.close()
        for proc in workers:
            proc.join(timeout=5)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    rate_tracker = RateTracker(window_seconds=120.0)
    staleness_samples: deque[float] = deque(maxlen=cfg.staleness_window)

    updates = 0
    try:
        while updates < cfg.total_updates:
            items = data_buffer.get_many(
                min_items=cfg.min_batches_per_update,
                timeout=cfg.queue_timeout_s,
            )
            staleness = [max(0, updates - item.policy_version) for item in items]
            staleness_samples.extend(staleness)
            batch = _concatenate_batches(items)
            batch = batch.to(trainer_device)
            metrics = trainer.step(batch)
            updates += 1

            completed = sum(item.batch.completed_episodes() for item in items)
            rate_tracker.update(completed)
            avg_staleness = sum(staleness_samples) / max(len(staleness_samples), 1)
            gpu_util = _query_gpu_util()
            logger.info(
                (
                    "update=%d reward=%.3f kl=%.4f entropy=%.3f eps/s=%.2f lag=%.2f "
                    "buffer=%d gpu_util=%s"
                ),
                updates,
                metrics["reward_mean"],
                metrics["kl"],
                metrics["entropy"],
                rate_tracker.rate(),
                avg_staleness,
                len(data_buffer),
                f"{gpu_util:.1f}%" if gpu_util is not None else "n/a",
            )

            if updates % cfg.weight_sync_interval == 0:
                _maybe_broadcast_weights(policy, weight_queues, version=updates)

    finally:
        _shutdown()

    logger.info("Async PPO training complete: updates=%d", updates)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Asynchronous PPO training example")
    parser.add_argument("--mode", choices=["colocate", "disaggregate"], default="colocate")
    parser.add_argument("--updates", type=int, default=40)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--buffer-capacity", type=int, default=8)
    parser.add_argument("--min-batches", type=int, default=2)
    parser.add_argument("--weight-sync", type=int, default=4)
    parser.add_argument("--staleness-window", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=3e-3)
    return parser.parse_args()


def main(cfg: AsyncExperimentConfig | None = None) -> None:
    if cfg is None:
        args = _parse_args()
        colocate = args.mode == "colocate"
        cfg = AsyncExperimentConfig(
            total_updates=args.updates,
            num_workers=args.workers,
            buffer_capacity=args.buffer_capacity,
            min_batches_per_update=args.min_batches,
            weight_sync_interval=args.weight_sync,
            staleness_window=args.staleness_window,
            colocate_rollouts=colocate,
            learning_rate=args.learning_rate,
        )
    run_async_training(cfg)


if __name__ == "__main__":
    main()


__all__ = ["AsyncExperimentConfig", "SimplePolicy", "run_async_training", "main"]

