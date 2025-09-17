from __future__ import annotations

from collections import deque
from typing import Deque

from core.interfaces import Buffer
from core.types import TrajectoryBatch


class TrajectoryBuffer(Buffer):
    """Simple FIFO buffer with bounded capacity."""

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("Buffer capacity must be positive")
        self.capacity = capacity
        self._queue: Deque[TrajectoryBatch] = deque()

    def put(self, batch: TrajectoryBatch) -> None:
        if len(self._queue) >= self.capacity:
            self._queue.popleft()
        self._queue.append(batch)

    def get(self) -> TrajectoryBatch:
        if not self._queue:
            raise RuntimeError("No batches available in buffer")
        batches = list(self._queue)
        self._queue.clear()
        return TrajectoryBatch.concat(batches)

    def __len__(self) -> int:
        return len(self._queue)


__all__ = ["TrajectoryBuffer"]
