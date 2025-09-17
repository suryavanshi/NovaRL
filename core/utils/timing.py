from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class RateTracker:
    """Utility for tracking throughput metrics."""

    window_seconds: float = 10.0

    def __post_init__(self) -> None:
        self._timestamps: list[float] = []
        self._counts: list[int] = []

    def update(self, count: int) -> None:
        now = time.time()
        self._timestamps.append(now)
        self._counts.append(count)
        self._trim(now)

    def rate(self) -> float:
        if not self._timestamps:
            return 0.0
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return 0.0
        total = sum(self._counts)
        return total / elapsed

    def _trim(self, now: float) -> None:
        while self._timestamps and now - self._timestamps[0] > self.window_seconds:
            self._timestamps.pop(0)
            self._counts.pop(0)


__all__ = ["RateTracker"]
