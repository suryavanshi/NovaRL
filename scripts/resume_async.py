from __future__ import annotations

import argparse
from pathlib import Path

from examples.ppo_async import AsyncExperimentConfig, run_async_training


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Resume asynchronous PPO training from a checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint file to load")
    parser.add_argument("--updates", type=int, default=None, help="Override the remaining updates to run")
    parser.add_argument("--workers", type=int, default=None, help="Override the number of rollout workers")
    parser.add_argument("--buffer-capacity", type=int, default=None, help="Override rollout buffer capacity")
    parser.add_argument("--deterministic", action="store_true", help="Enable deterministic kernels where possible")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    cfg = AsyncExperimentConfig()
    cfg.resume_from = str(Path(args.checkpoint).expanduser())
    if args.updates is not None:
        cfg.total_updates = args.updates
    if args.workers is not None:
        cfg.num_workers = args.workers
    if args.buffer_capacity is not None:
        cfg.buffer_capacity = args.buffer_capacity
    if args.deterministic:
        cfg.deterministic = True
    run_async_training(cfg)


if __name__ == "__main__":
    main()

