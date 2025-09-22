"""DeepSpeed ZeRO presets."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Literal, Mapping


@dataclass(frozen=True, slots=True)
class DeepSpeedZeROConfig:
    """Configuration helper for DeepSpeed ZeRO strategies."""

    stage: Literal[2, 3]
    bf16: bool = True
    fp16: bool = False
    offload_optimizer: bool = False
    offload_parameters: bool = False
    optimizer_sharding: bool = True
    gradient_accumulation_steps: int | str = "auto"
    train_micro_batch_size_per_gpu: int | str = "auto"
    gradient_clipping: float = 1.0
    steps_per_print: int = 2000
    optimizer: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "AdamW",
            "params": {
                "lr": "auto",
                "betas": [0.9, 0.95],
                "eps": 1e-8,
                "weight_decay": 0.0,
            },
        }
    )
    scheduler: Mapping[str, Any] = field(
        default_factory=lambda: {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": "auto",
                "warmup_num_steps": "auto",
            },
        }
    )

    def to_dict(self) -> Dict[str, Any]:
        zero_optimization: Dict[str, Any] = {
            "stage": self.stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_scatter": self.optimizer_sharding,
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_gather_16bit_weights_on_model_save": True,
        }
        if self.stage == 2:
            zero_optimization.pop("stage3_max_live_parameters")
            zero_optimization.pop("stage3_max_reuse_distance")
            zero_optimization.pop("stage3_prefetch_bucket_size")
            zero_optimization.pop("stage3_param_persistence_threshold")
            zero_optimization.pop("stage3_gather_16bit_weights_on_model_save")
        if self.offload_optimizer:
            zero_optimization["offload_optimizer"] = {"device": "cpu", "pin_memory": True}
        else:
            zero_optimization["offload_optimizer"] = {"device": "none"}
        if self.stage == 3:
            if self.offload_parameters:
                zero_optimization["offload_param"] = {"device": "cpu", "pin_memory": True}
            else:
                zero_optimization["offload_param"] = {"device": "none"}
        config: Dict[str, Any] = {
            "bf16": {"enabled": self.bf16},
            "fp16": {"enabled": self.fp16},
            "train_batch_size": "auto",
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "train_micro_batch_size_per_gpu": self.train_micro_batch_size_per_gpu,
            "gradient_clipping": self.gradient_clipping,
            "wall_clock_breakdown": False,
            "steps_per_print": self.steps_per_print,
            "zero_optimization": zero_optimization,
            "optimizer": dict(self.optimizer),
            "scheduler": dict(self.scheduler),
        }
        return config

    def dump_json(self, path: str | Path) -> Path:
        """Serialize the config to ``path``."""

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True)
        return target

    @classmethod
    def stage2_default(cls, **overrides: Any) -> "DeepSpeedZeROConfig":
        return cls(stage=2, **overrides)

    @classmethod
    def stage3_default(cls, **overrides: Any) -> "DeepSpeedZeROConfig":
        return cls(stage=3, **overrides)


__all__ = ["DeepSpeedZeROConfig"]
