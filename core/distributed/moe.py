"""Utilities describing MoE parallel layouts for Megatron/DeepSpeed."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional


@dataclass(slots=True)
class ParallelLayout:
    """Represents the combination of data, tensor, pipeline and expert parallelism."""

    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1

    def world_size(self) -> int:
        return (
            self.data_parallel_size
            * self.tensor_parallel_size
            * self.pipeline_parallel_size
            * self.expert_parallel_size
        )

    def validate(self, *, total_world_size: Optional[int] = None) -> None:
        if self.data_parallel_size <= 0:
            raise ValueError("data_parallel_size must be positive")
        if self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be positive")
        if self.pipeline_parallel_size <= 0:
            raise ValueError("pipeline_parallel_size must be positive")
        if self.expert_parallel_size <= 0:
            raise ValueError("expert_parallel_size must be positive")
        if total_world_size is not None and self.world_size() != total_world_size:
            raise ValueError(
                "Parallel layout world size mismatch: "
                f"{self.world_size()} vs expected {total_world_size}"
            )

    @classmethod
    def from_world_size(
        cls,
        total_world_size: int,
        *,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
        expert_parallel_size: int = 1,
    ) -> "ParallelLayout":
        if total_world_size <= 0:
            raise ValueError("total_world_size must be positive")
        denom = tensor_parallel_size * pipeline_parallel_size * expert_parallel_size
        if denom <= 0 or total_world_size % denom != 0:
            raise ValueError(
                "World size must be divisible by tp*pp*ep: "
                f"world={total_world_size} tp={tensor_parallel_size} pp={pipeline_parallel_size} ep={expert_parallel_size}"
            )
        data_parallel_size = total_world_size // denom
        layout = cls(
            data_parallel_size=data_parallel_size,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            expert_parallel_size=expert_parallel_size,
        )
        layout.validate()
        return layout

    def describe(self) -> Dict[str, int]:
        return {
            "dp": self.data_parallel_size,
            "tp": self.tensor_parallel_size,
            "pp": self.pipeline_parallel_size,
            "ep": self.expert_parallel_size,
            "world": self.world_size(),
        }

    def megatron_kwargs(self) -> Dict[str, int]:
        return {
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "expert_parallel_size": self.expert_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }

    def deepspeed_moe_dict(
        self,
        *,
        num_experts: int,
        top_k: int = 1,
        capacity_factor: float = 1.25,
        min_capacity: int = 4,
        router_aux_loss_coef: float = 0.01,
        router_jitter_noise: float = 0.0,
        type_: str = "dropless",
    ) -> Dict[str, object]:
        return {
            "enabled": True,
            "type": type_,
            "num_experts": num_experts,
            "top_k": top_k,
            "capacity_factor": capacity_factor,
            "min_capacity": min_capacity,
            "router_aux_loss_coef": router_aux_loss_coef,
            "router_jitter_noise": router_jitter_noise,
            "expert_parallel_size": self.expert_parallel_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "data_parallel_size": self.data_parallel_size,
        }


@dataclass(slots=True)
class MoERouterConfig:
    """Simple container for router hyper-parameters shared across backends."""

    num_experts: int
    top_k: int = 2
    capacity_factor: float = 1.2
    min_capacity: int = 4
    router_aux_loss_coef: float = 0.01
    router_jitter_noise: float = 0.0

    def to_deepspeed(self, layout: ParallelLayout) -> Dict[str, object]:
        return layout.deepspeed_moe_dict(
            num_experts=self.num_experts,
            top_k=self.top_k,
            capacity_factor=self.capacity_factor,
            min_capacity=self.min_capacity,
            router_aux_loss_coef=self.router_aux_loss_coef,
            router_jitter_noise=self.router_jitter_noise,
        )

    def to_megatron(self) -> Mapping[str, object]:
        return {
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "capacity_factor": self.capacity_factor,
            "min_capacity": self.min_capacity,
            "router_aux_loss_coef": self.router_aux_loss_coef,
            "router_jitter_noise": self.router_jitter_noise,
        }


__all__ = ["MoERouterConfig", "ParallelLayout"]

