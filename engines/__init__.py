"""Rollout engine adapters."""

from .vllm_engine import (
    VLLMGeneration,
    VLLMGenerationEngine,
    VLLMStreamResponse,
    meets_perf_target,
)

__all__ = [
    "VLLMGeneration",
    "VLLMGenerationEngine",
    "VLLMStreamResponse",
    "meets_perf_target",
]

try:  # pragma: no cover - optional dependency guard
    from .sync.sync_engine import SynchronousRolloutEngine
except Exception:  # pragma: no cover - torch may be unavailable in lightweight envs
    SynchronousRolloutEngine = None  # type: ignore[assignment]
else:  # pragma: no branch
    __all__.append("SynchronousRolloutEngine")
