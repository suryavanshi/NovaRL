from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, MutableMapping, Optional, Sequence

from .eval_tasks import AgentFactory, EvalResult, EvalTask


@dataclass
class RewardHackingSentinel:
    """Detect situations where rewards diverge from evaluation accuracy."""

    reward_key: str = "reward_mean"
    min_eval_score: float = 0.5
    reward_threshold: float = 0.5

    def check(self, metrics: Mapping[str, float], results: Sequence[EvalResult]) -> Sequence[str]:
        if not results:
            return []
        reward = metrics.get(self.reward_key)
        if reward is None:
            return []
        average_eval = sum(result.score for result in results) / len(results)
        if reward >= self.reward_threshold and average_eval < self.min_eval_score:
            message = (
                "Reward hacking suspected: "
                f"{self.reward_key}={reward:.3f} avg_eval={average_eval:.3f}"
            )
            return [message]
        return []


@dataclass
class KLDriftAlarm:
    """Monitor KL divergence metrics and signal drift."""

    kl_key: str = "kl"
    reference_kl_key: str = "kl_to_ref"
    max_kl: float = 0.15
    max_reference_kl: Optional[float] = None

    def check(self, metrics: Mapping[str, float]) -> Sequence[str]:
        warnings = []
        kl_value = metrics.get(self.kl_key)
        if kl_value is not None and kl_value > self.max_kl:
            warnings.append(
                f"KL drift detected: {self.kl_key}={kl_value:.4f} exceeds {self.max_kl:.4f}"
            )
        ref_threshold = self.max_reference_kl or self.max_kl
        ref_value = metrics.get(self.reference_kl_key)
        if ref_value is not None and ref_value > ref_threshold:
            warnings.append(
                "Reference KL drift detected: "
                f"{self.reference_kl_key}={ref_value:.4f} exceeds {ref_threshold:.4f}"
            )
        return warnings


@dataclass
class PeriodicEvalHook:
    """Runs evaluation tasks on a cadence and aggregates guardrails."""

    tasks: Sequence[EvalTask]
    agent_factory: AgentFactory
    frequency: int = 5
    reward_sentinel: Optional[RewardHackingSentinel] = None
    kl_alarm: Optional[KLDriftAlarm] = None
    history: MutableMapping[int, Sequence[EvalResult]] = field(default_factory=dict)
    _last_run: Optional[int] = None

    def maybe_run(
        self, iteration: int, metrics: Mapping[str, float]
    ) -> tuple[Sequence[EvalResult], Sequence[str]]:
        should_run = self._should_run(iteration)
        if not should_run:
            return (), ()
        results = []
        warnings: list[str] = []
        for task in self.tasks:
            try:
                result = task.run(self.agent_factory)
            except Exception as exc:  # pragma: no cover - defensive
                result = EvalResult(
                    name=getattr(task, "name", task.__class__.__name__),
                    score=0.0,
                    passed=False,
                    details={"error": str(exc)},
                )
            results.append(result)
        self.history[iteration] = tuple(results)
        if self.reward_sentinel is not None:
            warnings.extend(self.reward_sentinel.check(metrics, results))
        if self.kl_alarm is not None:
            warnings.extend(self.kl_alarm.check(metrics))
        self._last_run = iteration
        return tuple(results), tuple(warnings)

    def _should_run(self, iteration: int) -> bool:
        if self.frequency <= 0:
            return False
        if self._last_run is None:
            return True
        return (iteration - self._last_run) >= self.frequency


__all__ = [
    "PeriodicEvalHook",
    "RewardHackingSentinel",
    "KLDriftAlarm",
]
