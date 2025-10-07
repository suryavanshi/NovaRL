"""Interfaces for integrating Megatron-LM style parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Protocol, Sequence, Tuple

from .moe import ParallelLayout


@dataclass(slots=True)
class MegatronAdapterConfig:
    """Common configuration values for Megatron-aware adapters."""

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    expert_parallel_size: int = 1
    sequence_parallel: bool = True
    micro_batch_size: int = 1
    global_batch_size: int = 1
    gradient_checkpointing: bool = True
    moe_aux_loss_coeff: float = 0.01
    use_flash_attention: bool = True

    def parallel_layout(self, world_size: Optional[int] = None) -> ParallelLayout:
        if world_size is None:
            layout = ParallelLayout(
                data_parallel_size=1,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                expert_parallel_size=self.expert_parallel_size,
            )
            layout.validate()
            return layout
        return ParallelLayout.from_world_size(
            total_world_size=world_size,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            expert_parallel_size=self.expert_parallel_size,
        )


class MegatronAdapter(Protocol):
    """Protocol describing the hooks required to plug Megatron into NovaRL."""

    config: MegatronAdapterConfig

    def build_model(self) -> Any:
        """Return the Megatron-compatible model instance."""

    def configure_distributed_strategy(self) -> Mapping[str, Any]:
        """Return keyword arguments used when launching Megatron processes."""

    def build_train_dataloader(self) -> Iterable[Any]:
        """Return the iterable used for training."""

    def build_eval_dataloader(self) -> Optional[Iterable[Any]]:
        """Return the iterable used for evaluation, if any."""

    def optimizer_and_scheduler(self, model: Any) -> Tuple[Any, Any]:
        """Return ``(optimizer, scheduler)`` configured for ``model``."""

    def save_extra_state(self) -> MutableMapping[str, Any]:
        """Return auxiliary state such as MoE router statistics."""

    def load_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore adapter specific state."""

    def modules_to_migrate(self) -> Sequence[str]:
        """Return dotted paths of modules that require Megatron overrides."""


class NullMegatronAdapter:
    """No-op adapter useful for unit tests or single GPU debugging."""

    def __init__(self, config: Optional[MegatronAdapterConfig] = None) -> None:
        self.config = config or MegatronAdapterConfig()

    def build_model(self) -> Any:  # pragma: no cover - trivial container
        return None

    def configure_distributed_strategy(self) -> Mapping[str, Any]:
        layout = self.config.parallel_layout()
        return layout.megatron_kwargs()

    def build_train_dataloader(self) -> Iterable[Any]:  # pragma: no cover - trivial container
        return []

    def build_eval_dataloader(
        self,
    ) -> Optional[Iterable[Any]]:  # pragma: no cover - trivial container
        return None

    def optimizer_and_scheduler(
        self, model: Any
    ) -> Tuple[Any, Any]:  # pragma: no cover - trivial container
        return (None, None)

    def save_extra_state(self) -> MutableMapping[str, Any]:
        return {}

    def load_extra_state(self, state: Mapping[str, Any]) -> None:  # pragma: no cover - no-op
        return None

    def modules_to_migrate(self) -> Sequence[str]:
        return ()


class MegatronAdapterRegistry:
    """Registry mapping adapter names to factories."""

    def __init__(self) -> None:
        self._factories: MutableMapping[str, type[MegatronAdapter]] = {}

    def register(self, name: str, factory: type[MegatronAdapter]) -> None:
        if name in self._factories:
            raise ValueError(f"Megatron adapter {name!r} already registered")
        self._factories[name] = factory

    def create(self, name: str, **kwargs: Any) -> MegatronAdapter:
        try:
            factory = self._factories[name]
        except KeyError as exc:  # pragma: no cover - defensive branch
            raise KeyError(f"Megatron adapter {name!r} is not registered") from exc
        return factory(**kwargs)

    def available(self) -> Sequence[str]:
        return tuple(sorted(self._factories))


__all__ = [
    "MegatronAdapter",
    "MegatronAdapterConfig",
    "MegatronAdapterRegistry",
    "NullMegatronAdapter",
]
