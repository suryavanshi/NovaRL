"""High-level FSDP presets for Hugging Face Trainer integration."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, Optional, Sequence


_FSDP_AUTO_WRAP_POLICY = "TRANSFORMER_BASED_WRAP"


@dataclass(frozen=True, slots=True)
class FSDPStrategyPreset:
    """Container describing a reusable FSDP strategy.

    The preset mirrors the knobs consumed by :class:`transformers.TrainingArguments` via
    ``fsdp`` / ``fsdp_config``.  It exposes toggles for gradient checkpointing, mixed
    precision, and CPU offload while leaving plenty of room for future extensions such as
    mixed expert/pipeline/tensor parallelism.
    """

    name: str = "fsdp_default"
    transformer_layer_cls_to_wrap: Sequence[str] = ("LlamaDecoderLayer", "DecoderLayer")
    backward_prefetch: str = "BACKWARD_POST"
    state_dict_type: str = "FULL_STATE_DICT"
    limit_all_gathers: bool = True
    sync_module_states: bool = True
    use_orig_params: bool = True
    gradient_checkpointing: bool = True
    activation_checkpointing_reentrant: bool = True
    mixed_precision: str = "bf16"
    cpu_offload: bool = False
    forward_prefetch: bool = False

    def with_overrides(
        self,
        *,
        name: Optional[str] = None,
        transformer_layer_cls_to_wrap: Optional[Sequence[str]] = None,
        gradient_checkpointing: Optional[bool] = None,
        mixed_precision: Optional[str] = None,
        cpu_offload: Optional[bool] = None,
    ) -> "FSDPStrategyPreset":
        """Return a copy of the preset with user supplied overrides."""

        return replace(
            self,
            name=self.name if name is None else name,
            transformer_layer_cls_to_wrap=(
                self.transformer_layer_cls_to_wrap
                if transformer_layer_cls_to_wrap is None
                else tuple(transformer_layer_cls_to_wrap)
            ),
            gradient_checkpointing=(
                self.gradient_checkpointing
                if gradient_checkpointing is None
                else gradient_checkpointing
            ),
            mixed_precision=self.mixed_precision if mixed_precision is None else mixed_precision,
            cpu_offload=self.cpu_offload if cpu_offload is None else cpu_offload,
        )

    # ------------------------------------------------------------------
    # Hugging Face Trainer helpers.
    # ------------------------------------------------------------------
    @property
    def fsdp_argument(self) -> str:
        """String consumed by ``TrainingArguments.fsdp``."""

        return "full_shard auto_wrap"

    def to_hf_dict(
        self,
        *,
        gradient_checkpointing: Optional[bool] = None,
        mixed_precision: Optional[str] = None,
        cpu_offload: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Translate the preset into a Hugging Face compatible dictionary."""

        grad_ckpt = (
            self.gradient_checkpointing
            if gradient_checkpointing is None
            else gradient_checkpointing
        )
        precision = self.mixed_precision if mixed_precision is None else mixed_precision
        offload = self.cpu_offload if cpu_offload is None else cpu_offload

        config: Dict[str, Any] = {
            "fsdp_auto_wrap_policy": _FSDP_AUTO_WRAP_POLICY,
            "fsdp_backward_prefetch": self.backward_prefetch,
            "fsdp_state_dict_type": self.state_dict_type,
            "fsdp_transformer_layer_cls_to_wrap": list(self.transformer_layer_cls_to_wrap),
            "limit_all_gathers": self.limit_all_gathers,
            "sync_module_states": self.sync_module_states,
            "use_orig_params": self.use_orig_params,
            "forward_prefetch": self.forward_prefetch,
            "activation_checkpointing": bool(grad_ckpt),
            "activation_checkpointing_reentrant": self.activation_checkpointing_reentrant,
            "mixed_precision": precision,
            "cpu_offload": bool(offload),
            "fsdp_cpu_offload": bool(offload),
        }
        if precision not in {"bf16", "fp16", "fp32"}:
            raise ValueError(
                "mixed_precision must be one of {'bf16', 'fp16', 'fp32'}; received"
                f" {precision!r}"
            )
        if precision == "fp32":
            # The Trainer expects ``None`` when full precision is requested.
            config["mixed_precision"] = None
        return config


def get_default_fsdp_preset() -> FSDPStrategyPreset:
    """Return the canonical FSDP preset used across launch scripts."""

    return FSDPStrategyPreset()


__all__ = ["FSDPStrategyPreset", "get_default_fsdp_preset"]
