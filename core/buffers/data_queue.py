"""Concurrent data buffer utilities used by asynchronous trainers."""

from __future__ import annotations

import queue
import threading
from typing import Any, Generic, List, Optional, TypeVar

T = TypeVar("T")


class DataBuffer(Generic[T]):
    """A bounded FIFO queue with optional batch retrieval helpers.

    The buffer is safe to use from multiple producer and consumer threads or
    processes as long as the underlying queue object obeys the standard Python
    ``Queue`` interface.  By default the buffer creates a ``queue.Queue`` which
    works well for intra-process coordination.  Callers that need to share data
    across processes can construct a ``multiprocessing.Queue`` externally and
    pass it to the constructor via the ``queue`` argument.
    """

    _SENTINEL: object = object()

    def __init__(
        self,
        capacity: int,
        *,
        queue_obj: Optional[queue.Queue] = None,
        drop_oldest: bool = False,
    ) -> None:
        if capacity <= 0:
            raise ValueError("DataBuffer capacity must be positive")
        if queue_obj is None:
            queue_obj = queue.Queue(maxsize=capacity)
        self._queue: Any = queue_obj
        self._capacity = capacity
        self._drop_oldest = drop_oldest
        self._closed = False
        self._lock = threading.Lock()

    @property
    def queue(self) -> Any:
        """Expose the underlying queue for compatibility with multiprocessing."""

        return self._queue

    def put(self, item: T, block: bool = True, timeout: Optional[float] = None) -> None:
        """Insert an item into the buffer.

        When ``drop_oldest`` is ``True`` the call never blocks: instead the
        oldest item is removed to make room for the new element.  This is
        convenient for telemetry queues where the most recent samples are more
        valuable than stale ones.  For the default behaviour (``drop_oldest`` is
        ``False``) the underlying queue's blocking semantics are respected and
        callers benefit from natural back-pressure.
        """

        if self._closed:
            raise RuntimeError("Cannot put items into a closed DataBuffer")
        if not self._drop_oldest:
            self._queue.put(item, block=block, timeout=timeout)
            return

        while True:
            try:
                self._queue.put(item, block=block, timeout=timeout)
                break
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    continue

    def get(self, block: bool = True, timeout: Optional[float] = None) -> T:
        """Remove and return a single item from the buffer."""

        item = self._queue.get(block=block, timeout=timeout)
        if item is self._SENTINEL:
            self._queue.put(item)
            raise RuntimeError("DataBuffer has been closed")
        return item

    def get_many(
        self,
        *,
        min_items: int = 1,
        max_items: Optional[int] = None,
        timeout: Optional[float] = None,
    ) -> List[T]:
        """Fetch a batch of items with optional bounds on the batch size.

        Args:
            min_items: Minimum number of items to return.  The method blocks
                until at least this many items are available.
            max_items: Upper bound on the number of items to return.  When
                ``None`` the method drains the entire queue after satisfying the
                ``min_items`` requirement.
            timeout: Maximum number of seconds to wait for the initial item.  A
                value of ``None`` blocks indefinitely.
        """

        if min_items <= 0:
            raise ValueError("min_items must be positive")
        if max_items is not None and max_items < min_items:
            raise ValueError("max_items must be >= min_items")

        items: List[T] = []
        first = self.get(block=True, timeout=timeout)
        items.append(first)
        target = max_items or float("inf")
        while len(items) < target:
            try:
                item = self._queue.get_nowait()
            except queue.Empty:
                break
            if item is self._SENTINEL:
                self._queue.put(item)
                break
            items.append(item)
        while len(items) < min_items:
            item = self.get(block=True)
            items.append(item)
        return items

    def close(self) -> None:
        """Prevent further ``put`` operations and unblock waiting consumers."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                self._queue.put_nowait(self._SENTINEL)
            except queue.Full:
                # Best-effort: remove one item to make space for the sentinel.
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(self._SENTINEL)
                except queue.Full:
                    pass

    def snapshot(self) -> list[T]:
        """Return a shallow copy of the buffer contents without consuming them."""

        with self._lock:
            drained: list[T] = []
            sentinel_seen = False
            while True:
                try:
                    item = self._queue.get_nowait()
                except queue.Empty:
                    break
                if item is self._SENTINEL:
                    sentinel_seen = True
                    continue
                drained.append(item)
            for item in drained:
                self._queue.put_nowait(item)
            if sentinel_seen:
                self._queue.put_nowait(self._SENTINEL)
            return list(drained)

    def __len__(self) -> int:
        try:
            return int(self._queue.qsize())
        except (NotImplementedError, AttributeError):
            return 0


__all__ = ["DataBuffer"]

