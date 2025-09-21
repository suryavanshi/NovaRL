"""Signal definitions for reinforcement learning from verifiable rewards."""

from __future__ import annotations

import abc
import json
import math
import re
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

import torch


@dataclass(slots=True)
class SignalResult:
    """Container holding both raw and weighted scores for a signal."""

    raw: torch.Tensor
    weighted: torch.Tensor


class VerifiableSignal(abc.ABC):
    """Base class for verifiable reward signals.

    Subclasses implement :meth:`_batch_score` or :meth:`_score_single` to
    transform rollout metadata (typically stored in ``EnvStep.infos``) into a
    tensor of scalar scores. Each signal can be assigned an arbitrary weight and
    records a histogram of the most recent batch of raw scores to make reward
    debugging easier.
    """

    def __init__(
        self,
        name: str,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        if not name:
            raise ValueError("Signal name must be a non-empty string")
        self.name = name
        self.weight = float(weight)
        self.histogram_bins = int(histogram_bins)
        if self.histogram_bins <= 0:
            raise ValueError("histogram_bins must be positive")
        self.latest_histogram: dict[str, list[float]] | None = None
        self.latest_raw: torch.Tensor | None = None

    def __call__(self, infos: Sequence[Mapping[str, Any]]) -> SignalResult:
        if not isinstance(infos, Sequence):
            raise TypeError("infos must be a sequence of mappings")
        raw = self._batch_score(infos)
        if not isinstance(raw, torch.Tensor):
            raw = torch.as_tensor(raw, dtype=torch.float32)
        raw = raw.to(dtype=torch.float32)
        if raw.dim() != 1:
            raise ValueError(
                f"Signal '{self.name}' expected a 1D tensor of scores, got {raw.shape}"
            )
        self.latest_raw = raw.detach().cpu()
        self.latest_histogram = self._compute_histogram(raw)
        weighted = raw * self.weight
        return SignalResult(raw=raw, weighted=weighted)

    def _compute_histogram(self, values: torch.Tensor) -> dict[str, list[float]] | None:
        if values.numel() == 0:
            return None
        hist = torch.histogram(values, bins=self.histogram_bins)
        edges = hist.bin_edges.detach().cpu().tolist()
        counts = hist.hist.detach().cpu().tolist()
        return {"edges": edges, "counts": counts}

    def _batch_score(self, infos: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        scores = [self._score_single(info) for info in infos]
        if scores and isinstance(scores[0], torch.Tensor):
            return torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in scores])
        return torch.as_tensor(scores, dtype=torch.float32)

    def _score_single(self, info: Mapping[str, Any]) -> float | torch.Tensor:
        """Score a single rollout info mapping.

        Subclasses overriding :meth:`_batch_score` do not need to implement this
        method. The base implementation raises ``NotImplementedError`` to catch
        accidental fall-through during development.
        """

        raise NotImplementedError(
            f"Signal '{self.name}' does not implement '_score_single'"
        )


class RegexMatchSignal(VerifiableSignal):
    """Scores outputs that satisfy a regular expression constraint."""

    def __init__(
        self,
        name: str,
        pattern: str | None = None,
        *,
        completion_key: str = "completion",
        pattern_key: str | None = None,
        positive_reward: float = 1.0,
        negative_reward: float = 0.0,
        flags: int = 0,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        super().__init__(name=name, weight=weight, histogram_bins=histogram_bins)
        if pattern is None and pattern_key is None:
            raise ValueError("RegexMatchSignal requires either pattern or pattern_key")
        self._compiled = re.compile(pattern, flags) if pattern is not None else None
        self.pattern_key = pattern_key
        self.completion_key = completion_key
        self.positive_reward = float(positive_reward)
        self.negative_reward = float(negative_reward)

    def _resolve_pattern(self, info: Mapping[str, Any]) -> re.Pattern[str]:
        if self._compiled is not None:
            return self._compiled
        if self.pattern_key is None:
            raise RuntimeError("No pattern configured for RegexMatchSignal")
        raw = info.get(self.pattern_key)
        if raw is None:
            raise KeyError(f"Info dictionary missing pattern key '{self.pattern_key}'")
        if not isinstance(raw, str):
            raise TypeError(
                f"Pattern extracted from key '{self.pattern_key}' must be a string, got {type(raw)!r}"
            )
        return re.compile(raw)

    def _score_single(self, info: Mapping[str, Any]) -> float:
        text = str(info.get(self.completion_key, ""))
        pattern = self._resolve_pattern(info)
        return self.positive_reward if pattern.search(text) else self.negative_reward


class MathAnswerSignal(VerifiableSignal):
    """Checks whether the extracted final answer matches the reference solution."""

    def __init__(
        self,
        name: str,
        *,
        answer_key: str = "answer",
        completion_key: str = "completion",
        extractor: Callable[[str], float | None] | None = None,
        correct_reward: float = 1.0,
        incorrect_reward: float = 0.0,
        tolerance: float = 1e-3,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        super().__init__(name=name, weight=weight, histogram_bins=histogram_bins)
        self.answer_key = answer_key
        self.completion_key = completion_key
        self.extractor = extractor or self._default_extractor
        self.correct_reward = float(correct_reward)
        self.incorrect_reward = float(incorrect_reward)
        self.tolerance = float(tolerance)

    @staticmethod
    def _default_extractor(text: str) -> float | None:
        """Heuristic extractor that returns the final numeric answer from text."""

        if not text:
            return None
        # Common GSM8K/MATH formats include "#### 42" or "\boxed{42}".
        candidates = re.findall(r"-?\d+(?:\.\d+)?", text)
        if not candidates:
            return None
        try:
            return float(candidates[-1])
        except ValueError:
            return None

    def _score_single(self, info: Mapping[str, Any]) -> float:
        completion = str(info.get(self.completion_key, ""))
        expected = info.get(self.answer_key)
        if expected is None:
            raise KeyError(f"Info dictionary missing answer key '{self.answer_key}'")
        predicted = self.extractor(completion)
        if predicted is None:
            return self.incorrect_reward
        try:
            expected_value = float(expected)
        except (TypeError, ValueError):
            expected_value = None
        if expected_value is None:
            return self.correct_reward if str(expected).strip() == str(predicted).strip() else self.incorrect_reward
        return (
            self.correct_reward
            if math.isclose(predicted, expected_value, rel_tol=0.0, abs_tol=self.tolerance)
            else self.incorrect_reward
        )


class UnitTestSignal(VerifiableSignal):
    """Executes unit tests defined in rollout metadata."""

    def __init__(
        self,
        name: str,
        *,
        completion_key: str = "completion",
        tests_key: str = "tests",
        entry_point_key: str = "entry_point",
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        super().__init__(name=name, weight=weight, histogram_bins=histogram_bins)
        self.completion_key = completion_key
        self.tests_key = tests_key
        self.entry_point_key = entry_point_key
        self.success_reward = float(success_reward)
        self.failure_reward = float(failure_reward)

    def _score_single(self, info: Mapping[str, Any]) -> float:
        source = info.get(self.completion_key, "")
        tests = info.get(self.tests_key)
        entry_point = info.get(self.entry_point_key, "solution")
        if not isinstance(source, str):
            raise TypeError(
                f"UnitTestSignal expects completion text to be a string, got {type(source)!r}"
            )
        if not isinstance(entry_point, str):
            raise TypeError(
                f"Entry point must be a string, got {type(entry_point)!r}"
            )
        if not tests:
            return self.failure_reward
        if not isinstance(tests, Sequence):
            raise TypeError("tests metadata must be a sequence of dictionaries")
        namespace: MutableMapping[str, Any] = {}
        try:
            exec(source, namespace)
        except Exception:
            return self.failure_reward
        fn = namespace.get(entry_point)
        if not callable(fn):
            return self.failure_reward
        total = 0
        passed = 0
        for case in tests:
            if not isinstance(case, Mapping):
                raise TypeError("Test cases must be mappings")
            args = list(case.get("args", ()))
            kwargs = dict(case.get("kwargs", {}))
            expected = case.get("expected")
            total += 1
            try:
                result = fn(*args, **kwargs)
            except Exception:
                continue
            if result == expected:
                passed += 1
        if total == 0:
            return self.failure_reward
        ratio = passed / total
        return self.failure_reward + ratio * (self.success_reward - self.failure_reward)


class HFRewardModelSignal(VerifiableSignal):
    """Scores outputs using a Hugging Face reward model."""

    def __init__(
        self,
        name: str,
        *,
        model_name: str,
        completion_key: str = "completion",
        prompt_key: str | None = "prompt",
        device: torch.device | None = None,
        apply_sigmoid: bool = True,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        super().__init__(name=name, weight=weight, histogram_bins=histogram_bins)
        self.model_name = model_name
        self.completion_key = completion_key
        self.prompt_key = prompt_key
        self.device = device or torch.device("cpu")
        self.apply_sigmoid = apply_sigmoid
        self._model = None
        self._tokenizer = None

    def _ensure_model(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "transformers is required for HFRewardModelSignal"
            ) from exc
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.to(self.device)
        self._model.eval()

    def _batch_score(self, infos: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        self._ensure_model()
        assert self._tokenizer is not None  # for type checker
        assert self._model is not None
        texts = [str(info.get(self.completion_key, "")) for info in infos]
        if self.prompt_key is not None:
            prompts = [str(info.get(self.prompt_key, "")) for info in infos]
            encoded = self._tokenizer(
                prompts,
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        else:
            encoded = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():  # pragma: no cover - depends on external model
            outputs = self._model(**encoded)
        logits = outputs.logits
        if logits.dim() == 1:
            scores = logits
        else:
            # Use the first column when multiple logits are provided.
            scores = logits[:, 0]
        if self.apply_sigmoid:
            scores = torch.sigmoid(scores)
        return scores.detach().cpu()


class HTTPRewardModelSignal(VerifiableSignal):
    """Scores outputs by querying an HTTP microservice."""

    def __init__(
        self,
        name: str,
        *,
        endpoint: str,
        completion_key: str = "completion",
        prompt_key: str | None = "prompt",
        timeout: float = 5.0,
        client: Callable[[Mapping[str, Any]], Sequence[float]] | None = None,
        weight: float = 1.0,
        histogram_bins: int = 20,
    ) -> None:
        super().__init__(name=name, weight=weight, histogram_bins=histogram_bins)
        self.endpoint = endpoint
        self.completion_key = completion_key
        self.prompt_key = prompt_key
        self.timeout = timeout
        self.client = client

    def _call_remote(self, payload: Mapping[str, Any]) -> Sequence[float]:
        if self.client is not None:
            return self.client(payload)
        import urllib.request  # pragma: no cover - network path

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            raw = response.read().decode("utf-8")
        parsed = json.loads(raw)
        scores = parsed.get("scores")
        if not isinstance(scores, Sequence):
            raise ValueError("HTTP reward model response missing 'scores' list")
        return [float(x) for x in scores]

    def _batch_score(self, infos: Sequence[Mapping[str, Any]]) -> torch.Tensor:
        payload: dict[str, Any] = {
            "completions": [str(info.get(self.completion_key, "")) for info in infos]
        }
        if self.prompt_key is not None:
            payload["prompts"] = [str(info.get(self.prompt_key, "")) for info in infos]
        scores = self._call_remote(payload)
        if len(scores) != len(infos):
            raise ValueError(
                f"HTTP reward model returned {len(scores)} scores for {len(infos)} requests"
            )
        return torch.as_tensor(scores, dtype=torch.float32)


__all__ = [
    "SignalResult",
    "VerifiableSignal",
    "RegexMatchSignal",
    "MathAnswerSignal",
    "UnitTestSignal",
    "HFRewardModelSignal",
    "HTTPRewardModelSignal",
]

