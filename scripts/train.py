from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from examples.minimal_ppo_sync import ExperimentConfig
from examples.minimal_ppo_sync import main as run_sync_example
from examples.moe_demo import MoEDemoConfig
from examples.moe_demo import main as run_moe_demo
from examples.ppo_async import AsyncExperimentConfig, run_async_training

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "sync")
    if mode == "sync":
        exp_cfg = ExperimentConfig(**cfg.get("experiment", {}))
        run_sync_example(exp_cfg)
    elif mode == "async":
        exp_cfg = AsyncExperimentConfig(**cfg.get("experiment", {}))
        run_async_training(exp_cfg)
    elif mode == "moe_demo":
        exp_cfg = MoEDemoConfig(**cfg.get("experiment", {}))
        run_moe_demo(exp_cfg)
    else:
        raise ValueError(f"Unsupported trainer mode: {mode}")


if __name__ == "__main__":
    main()
