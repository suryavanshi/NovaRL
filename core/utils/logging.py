"""Structured metrics logging utilities for NovaRL experiments."""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, MutableSequence, Optional, Sequence

try:  # pragma: no cover - optional dependency
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - tensorboard is optional
    SummaryWriter = None  # type: ignore[assignment]


@dataclass(slots=True)
class MetricsEvent:
    """Container describing a single metrics emission.

    Attributes:
        timestamp: Seconds since the UNIX epoch.
        role: Identifier describing which system component produced the event
            (for example ``"trainer"`` or ``"rollout"``).
        metrics: Mapping from metric names to numeric values.
        step: Optional global step index associated with the metrics.
        process_id: Operating system process identifier, useful when multiple
            worker processes share the same ``role``.
        worker_id: Optional logical worker identifier.
        extra: Arbitrary metadata that should be preserved in structured logs.
    """

    timestamp: float
    role: str
    metrics: Mapping[str, float]
    step: Optional[int] = None
    process_id: Optional[int] = None
    worker_id: Optional[str] = None
    extra: Mapping[str, Any] | None = None


class MetricsSink:
    """Abstract interface implemented by metrics consumers."""

    def write(self, event: MetricsEvent) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class JsonlMetricsSink(MetricsSink):
    """Persists metrics as structured JSON lines."""

    def __init__(self, path: os.PathLike[str] | str) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8")
        self._lock = threading.Lock()

    def write(self, event: MetricsEvent) -> None:
        payload = {
            "timestamp": event.timestamp,
            "role": event.role,
            "metrics": dict(event.metrics),
            "step": event.step,
            "process_id": event.process_id,
            "worker_id": event.worker_id,
            "extra": dict(event.extra) if event.extra else None,
        }
        with self._lock:
            self._file.write(json.dumps(payload, sort_keys=True) + "\n")
            self._file.flush()

    def close(self) -> None:
        with self._lock:
            self._file.close()


class TensorBoardMetricsSink(MetricsSink):
    """Emits metrics to TensorBoard via :class:`SummaryWriter`."""

    def __init__(self, log_dir: os.PathLike[str] | str) -> None:
        if SummaryWriter is None:  # pragma: no cover - optional dependency
            raise RuntimeError("TensorBoard is not available in this environment")
        self._writer = SummaryWriter(log_dir=str(log_dir))
        self._event_index = 0

    def write(self, event: MetricsEvent) -> None:
        step = event.step if event.step is not None else self._event_index
        tag_prefix = event.role
        for key, value in event.metrics.items():
            self._writer.add_scalar(f"{tag_prefix}/{key}", value, step)
        if event.extra:
            for key, value in event.extra.items():
                if isinstance(value, (int, float)):
                    self._writer.add_scalar(f"{tag_prefix}/extra/{key}", value, step)
        self._event_index += 1

    def close(self) -> None:
        self._writer.flush()
        self._writer.close()


class WandBMetricsSink(MetricsSink):
    """Streams metrics to Weights & Biases if it is available."""

    def __init__(
        self,
        *,
        project: str,
        run_name: str | None = None,
        entity: str | None = None,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        try:  # pragma: no cover - optional dependency
            import wandb
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Weights & Biases is not available") from exc

        self._wandb = wandb
        self._wandb.init(project=project, name=run_name, entity=entity, config=config)

    def write(self, event: MetricsEvent) -> None:
        payload: dict[str, Any] = {}
        prefix = event.role
        for key, value in event.metrics.items():
            payload[f"{prefix}/{key}"] = value
        if event.extra:
            for key, value in event.extra.items():
                if isinstance(value, (int, float)):
                    payload[f"{prefix}/extra/{key}"] = value
        self._wandb.log(payload, step=event.step)

    def close(self) -> None:
        self._wandb.finish()


class MetricsAggregator:
    """Background consumer that fans metrics out to configured sinks."""

    _SENTINEL = object()

    def __init__(
        self,
        metrics_queue: "queue.Queue[MetricsEvent | object]",
        *,
        sinks: Sequence[MetricsSink] | None = None,
        history: MutableSequence[MetricsEvent] | None = None,
        poll_interval_s: float = 0.5,
    ) -> None:
        self._queue = metrics_queue
        self._sinks = list(sinks or [])
        self._history = history
        self._poll_interval_s = poll_interval_s
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("MetricsAggregator has already been started")
        self._thread = threading.Thread(target=self._run, name="metrics-aggregator", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                event = self._queue.get(timeout=self._poll_interval_s)
            except queue.Empty:
                continue
            if event is self._SENTINEL:
                break
            assert isinstance(event, MetricsEvent)
            if self._history is not None:
                self._history.append(event)
            for sink in self._sinks:
                try:
                    sink.write(event)
                except Exception:  # pragma: no cover - logging should be best effort
                    continue
        for sink in self._sinks:
            try:
                sink.close()
            except Exception:  # pragma: no cover - best effort
                continue

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        try:
            self._queue.put_nowait(self._SENTINEL)
        except queue.Full:  # pragma: no cover - unlikely for large queues
            pass
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None


class ProcessMetricsLogger:
    """Helper used by individual processes to submit metrics events."""

    def __init__(
        self,
        metrics_queue: "queue.Queue[MetricsEvent | object]",
        *,
        role: str,
        worker_id: str | None = None,
    ) -> None:
        self._queue = metrics_queue
        self._role = role
        self._worker_id = worker_id

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        step: Optional[int] = None,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        event = MetricsEvent(
            timestamp=time.time(),
            role=self._role,
            metrics=dict(metrics),
            step=step,
            process_id=os.getpid(),
            worker_id=self._worker_id,
            extra=dict(extra) if extra else None,
        )
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            # Drop metrics on the floor when overwhelmed; metrics are best-effort.
            return


__all__ = [
    "JsonlMetricsSink",
    "MetricsAggregator",
    "MetricsEvent",
    "MetricsSink",
    "ProcessMetricsLogger",
    "TensorBoardMetricsSink",
    "WandBMetricsSink",
]

