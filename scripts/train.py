from __future__ import annotations

import logging

import hydra
from omegaconf import DictConfig

from examples.minimal_ppo_sync import ExperimentConfig
from examples.minimal_ppo_sync import main as run_sync_example

logger = logging.getLogger(__name__)


@hydra.main(config_path="../configs", config_name="train", version_base=None)
def main(cfg: DictConfig) -> None:
    mode = cfg.get("mode", "sync")
    if mode == "sync":
        exp_cfg = ExperimentConfig(**cfg.get("experiment", {}))
        run_sync_example(exp_cfg)
    elif mode == "async":
        logger.warning("Async mode is not yet implemented; falling back to sync example.")
        exp_cfg = ExperimentConfig(**cfg.get("experiment", {}))
        run_sync_example(exp_cfg)
    else:
        raise ValueError(f"Unsupported trainer mode: {mode}")


if __name__ == "__main__":
    main()
