from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import queue
import signal
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
from torch import nn

from algos.ppo.trainer import PPOTrainer
from core.buffers import DataBuffer
from core.types import TrajectoryBatch
from core.utils.checkpoint import CheckpointManager, CheckpointState
from core.utils.logging import (
    JsonlMetricsSink,
    MetricsAggregator,
    MetricsEvent,
    ProcessMetricsLogger,
    TensorBoardMetricsSink,
    WandBMetricsSink,
)
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
    log_dir: str = "runs/async_ppo"
    metrics_jsonl: str = "metrics.jsonl"
    enable_tensorboard: bool = True
    enable_wandb: bool = False
    wandb_project: str = "NovaRL"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    checkpoint_dir: str = "checkpoints/async_ppo"
    checkpoint_interval: int = 5
    resume_from: str | None = None
    debug_rollout_only: bool = False
    debug_train_only: bool = False
    deterministic: bool = False
    replay_path: str | None = None
    dump_replay: str | None = None
    metrics_queue_capacity: int = 1024
    auto_retry_rollouts: bool = True
    pause_timeout_s: float = 0.5
    debug_iterations: int = 3


@dataclass
class StampedBatch:
    batch: TrajectoryBatch
    policy_version: int
    created_at: float


def _seed_all(seed: int, *, deterministic: bool = False) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
        torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]


def _policy_state_dict(policy: nn.Module) -> dict[str, torch.Tensor]:
    return {name: param.detach().cpu() for name, param in policy.state_dict().items()}


def _load_policy_state(policy: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    policy.load_state_dict(state_dict)


def _stamped_to_cpu(item: StampedBatch) -> StampedBatch:
    cpu_batch = item.batch.detach().to(torch.device("cpu"))
    return StampedBatch(batch=cpu_batch, policy_version=item.policy_version, created_at=item.created_at)


def _save_replay(path: Path, items: Sequence[StampedBatch]) -> None:
    payload = [_stamped_to_cpu(item) for item in items]
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _load_replay(path: Path) -> List[StampedBatch]:
    payload: List[StampedBatch] = torch.load(path, map_location="cpu")
    return payload


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


def _rollout_worker(
    worker_id: int,
    cfg: AsyncExperimentConfig,
    data_queue: mp.Queue,
    weight_queue: mp.Queue,
    stop_event: mp.Event,
    resume_event: mp.Event,
    metrics_queue: mp.Queue,
    initial_state: dict[str, torch.Tensor],
    device: str,
) -> None:
    _seed_all(cfg.seed + worker_id, deterministic=cfg.deterministic)
    torch.set_num_threads(1)

    rollout_device = torch.device(device)
    policy, engine = _make_rollout_components(cfg, rollout_device)
    _load_policy_state(policy, initial_state)

    rate_tracker = RateTracker(window_seconds=60.0)
    policy_version = 0
    metrics_logger = ProcessMetricsLogger(metrics_queue, role="rollout", worker_id=str(worker_id))

    while not stop_event.is_set():
        resume_event.wait()
        if stop_event.is_set():
            break
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
            episodes = cpu_batch.completed_episodes()
            rate_tracker.update(episodes)
            metrics_logger.log(
                {"episodes": float(episodes), "eps_per_sec": float(rate_tracker.rate())},
                extra={"policy_version": float(policy_version)},
            )
        except queue.Full:
            logger.warning("worker %d data queue full; dropping rollout", worker_id)
            continue

    metrics_logger.log({"eps_per_sec": float(rate_tracker.rate())}, extra={"final": 1.0})


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


def _setup_metrics(
    cfg: AsyncExperimentConfig,
    ctx: mp.context.BaseContext,
    experiment_config: dict[str, float | int | str | bool],
) -> tuple[mp.Queue, MetricsAggregator, dict[str, ProcessMetricsLogger]]:
    metrics_queue: mp.Queue = ctx.Queue(cfg.metrics_queue_capacity)
    log_dir = Path(cfg.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    sinks = []
    history: list[MetricsEvent] = []
    metrics_path = log_dir / cfg.metrics_jsonl
    sinks.append(JsonlMetricsSink(metrics_path))
    if cfg.enable_tensorboard:
        try:
            sinks.append(TensorBoardMetricsSink(log_dir / "tensorboard"))
        except RuntimeError as exc:
            logger.warning("TensorBoard sink unavailable: %s", exc)
    if cfg.enable_wandb:
        try:
            sinks.append(
                WandBMetricsSink(
                    project=cfg.wandb_project,
                    run_name=cfg.wandb_run_name,
                    entity=cfg.wandb_entity,
                    config=experiment_config,
                )
            )
        except RuntimeError as exc:
            logger.warning("Weights & Biases sink unavailable: %s", exc)
    aggregator = MetricsAggregator(metrics_queue, sinks=sinks, history=history)
    aggregator.start()
    process_loggers = {
        "trainer": ProcessMetricsLogger(metrics_queue, role="trainer"),
        "buffer": ProcessMetricsLogger(metrics_queue, role="buffer"),
        "reward": ProcessMetricsLogger(metrics_queue, role="reward"),
        "system": ProcessMetricsLogger(metrics_queue, role="system"),
    }
    return metrics_queue, aggregator, process_loggers


def run_async_training(cfg: AsyncExperimentConfig) -> None:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logger.info("Starting asynchronous PPO example with config: %s", cfg)
    _seed_all(cfg.seed, deterministic=cfg.deterministic)

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
    resume_event = ctx.Event()
    resume_event.set()
    weight_queues: List[mp.Queue] = [ctx.Queue(maxsize=1) for _ in range(cfg.num_workers)]

    metrics_queue, aggregator, process_loggers = _setup_metrics(
        cfg,
        ctx,
        experiment_config={"num_workers": cfg.num_workers, "batch_size": cfg.batch_size},
    )

    checkpoint_manager = CheckpointManager(cfg.checkpoint_dir)
    resume_state: CheckpointState | None = None
    if cfg.resume_from:
        path = Path(cfg.resume_from)
        resume_manager = CheckpointManager(path.parent, filename=path.name)
        resume_state = resume_manager.load()

    trainer_logger = process_loggers["trainer"]
    buffer_logger = process_loggers["buffer"]
    reward_logger = process_loggers["reward"]
    system_logger = process_loggers["system"]

    def _spawn_worker(worker_id: int, initial_state: dict[str, torch.Tensor]) -> mp.Process:
        proc = ctx.Process(
            target=_rollout_worker,
            args=(
                worker_id,
                cfg,
                data_queue,
                weight_queues[worker_id],
                stop_event,
                resume_event,
                metrics_queue,
                initial_state,
                str(rollout_device),
            ),
            name=f"rollout-worker-{worker_id}",
        )
        proc.daemon = True
        proc.start()
        return proc

    workers: List[mp.Process] = []
    if not cfg.debug_train_only:
        initial_state = _policy_state_dict(policy)
        for worker_id in range(cfg.num_workers):
            workers.append(_spawn_worker(worker_id, initial_state))

    resume_buffer: list[StampedBatch] = []
    updates = 0
    policy_version = 0
    if resume_state is not None:
        payload = resume_state.payload
        policy.load_state_dict(payload["policy_state"])
        optimizer.load_state_dict(payload["optimizer_state"])
        updates = int(payload.get("updates", resume_state.step))
        policy_version = int(payload.get("policy_version", updates))
        resume_buffer = list(payload.get("buffer_items", []))
        torch_state = payload.get("torch_rng_state")
        if torch_state is not None:
            torch.random.set_rng_state(torch_state)
        cuda_state = payload.get("cuda_rng_state")
        if torch.cuda.is_available() and cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)  # type: ignore[arg-type]
        system_logger.log({"resume_step": float(updates)})

    if workers:
        _maybe_broadcast_weights(policy, weight_queues, version=policy_version)

    for item in resume_buffer:
        data_queue.put(item)

    paused_for_checkpoint = False
    last_checkpoint_step: int | None = None

    def _shutdown(*_: object) -> None:
        nonlocal paused_for_checkpoint
        logger.info("Received shutdown signal; stopping workers")
        stop_event.set()
        data_buffer.close()
        resume_event.set()
        for proc in workers:
            proc.join(timeout=5)
        if not paused_for_checkpoint and cfg.checkpoint_interval > 0:
            try:
                if last_checkpoint_step != updates:
                    _save_checkpoint(updates, policy_version)
            except Exception as exc:  # pragma: no cover - best effort during shutdown
                logger.error("Failed to save checkpoint during shutdown: %s", exc)
        aggregator.stop()

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    staleness_samples: deque[float] = deque(maxlen=cfg.staleness_window)
    rate_tracker = RateTracker(window_seconds=120.0)

    dump_written = False

    def _save_checkpoint(current_update: int, version: int) -> None:
        nonlocal last_checkpoint_step
        resume_event.clear()
        try:
            time.sleep(cfg.pause_timeout_s)
            snapshot_items = [_stamped_to_cpu(item) for item in data_buffer.snapshot()]
        finally:
            resume_event.set()
        payload = {
            "policy_state": _policy_state_dict(policy),
            "optimizer_state": optimizer.state_dict(),
            "buffer_items": snapshot_items,
            "policy_version": version,
            "updates": current_update,
            "torch_rng_state": torch.random.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        checkpoint_manager.save(CheckpointState(step=current_update, payload=payload))
        system_logger.log({"checkpoint_step": float(current_update)})
        last_checkpoint_step = current_update

    try:
        if cfg.debug_train_only:
            if cfg.replay_path is None:
                raise ValueError("debug_train_only requires replay_path")
            replay_batches = _load_replay(Path(cfg.replay_path))
            for _ in range(cfg.total_updates):
                batch = _concatenate_batches(replay_batches).to(trainer_device)
                metrics = trainer.step(batch)
                updates += 1
                trainer_logger.log(metrics, step=updates)
                reward_logger.log({"reward_mean": metrics["reward_mean"]}, step=updates)
            return

        if cfg.debug_rollout_only:
            iterations = max(cfg.debug_iterations, 1)
            processed = 0
            while processed < iterations:
                items = data_buffer.get_many(min_items=1, timeout=cfg.queue_timeout_s)
                processed += 1
                completed = sum(item.batch.completed_episodes() for item in items)
                buffer_logger.log({"size": float(len(data_buffer))}, step=processed)
                reward_logger.log({"reward_mean": float(items[0].batch.mean_reward())}, step=processed)
            return

        while updates < cfg.total_updates:
            if cfg.auto_retry_rollouts:
                for idx, proc in enumerate(workers):
                    if not proc.is_alive() and not stop_event.is_set():
                        logger.warning("worker %d died; restarting", idx)
                        system_logger.log({"worker_restart": float(idx)})
                        workers[idx] = _spawn_worker(idx, _policy_state_dict(policy))
                        _maybe_broadcast_weights(policy, [weight_queues[idx]], version=policy_version)

            if resume_buffer:
                items = resume_buffer
                resume_buffer = []
            else:
                items = data_buffer.get_many(
                    min_items=cfg.min_batches_per_update,
                    timeout=cfg.queue_timeout_s,
                )

            staleness = [max(0, policy_version - item.policy_version) for item in items]
            staleness_samples.extend(staleness)
            batch = _concatenate_batches(items).to(trainer_device)
            metrics = trainer.step(batch)
            updates += 1
            policy_version = max(policy_version, updates)

            completed = sum(item.batch.completed_episodes() for item in items)
            rate_tracker.update(completed)
            avg_staleness = sum(staleness_samples) / max(len(staleness_samples), 1)
            gpu_util = _query_gpu_util()
            trainer_logger.log(metrics, step=updates)
            reward_logger.log({"reward_mean": metrics["reward_mean"]}, step=updates)
            buffer_logger.log(
                {"size": float(len(data_buffer)), "lag": float(avg_staleness)},
                step=updates,
                extra={"eps_per_sec": rate_tracker.rate(), "gpu_util": gpu_util or 0.0},
            )

            if not dump_written and cfg.dump_replay is not None:
                _save_replay(Path(cfg.dump_replay), items)
                dump_written = True

            if updates % cfg.weight_sync_interval == 0:
                _maybe_broadcast_weights(policy, weight_queues, version=updates)
                policy_version = updates

            if cfg.checkpoint_interval > 0 and updates % cfg.checkpoint_interval == 0:
                paused_for_checkpoint = True
                _save_checkpoint(updates, policy_version)
                paused_for_checkpoint = False

    finally:
        _shutdown()

    logger.info("Async PPO training complete: updates=%d", updates)


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
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--debug-rollout-only", action="store_true")
    parser.add_argument("--debug-train-only", action="store_true")
    parser.add_argument("--replay-path", type=str, default=None)
    parser.add_argument("--dump-replay", type=str, default=None)
    parser.add_argument("--deterministic", action="store_true")
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
            resume_from=args.resume_from,
            debug_rollout_only=args.debug_rollout_only,
            debug_train_only=args.debug_train_only,
            replay_path=args.replay_path,
            dump_replay=args.dump_replay,
            deterministic=args.deterministic,
        )
    run_async_training(cfg)


if __name__ == "__main__":
    main()


__all__ = ["AsyncExperimentConfig", "SimplePolicy", "run_async_training", "main"]

