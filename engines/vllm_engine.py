"""High-throughput rollout engine backed by vLLM."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union


try:  # pragma: no cover - optional dependency
    from vllm import LLM as _VLLMLLM  # type: ignore
    from vllm import SamplingParams  # type: ignore
except Exception:  # pragma: no cover - import guard for environments without vLLM
    _VLLMLLM = None

    class SamplingParams:  # type: ignore[override]
        """Lightweight stand-in used for unit tests when vLLM is unavailable."""

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)


@dataclass(frozen=True)
class VLLMGeneration:
    """Represents the final text completion for a single request."""

    prompt: str
    request_id: str
    text: str
    token_ids: Sequence[int]
    logprobs: Optional[Sequence[float]]
    finish_reason: Optional[str]
    metrics: Mapping[str, Any]
    index: int = 0


@dataclass(frozen=True)
class VLLMStreamResponse:
    """Incremental token update emitted during streaming generation."""

    prompt: str
    request_id: str
    text: str
    delta: str
    token_ids: Sequence[int]
    logprobs: Optional[Sequence[float]]
    finished: bool
    finish_reason: Optional[str]
    metrics: Mapping[str, Any]
    index: int = 0


def meets_perf_target(vllm_qps: float, baseline_qps: float, target_ratio: float = 10.0) -> bool:
    """Return ``True`` when the vLLM throughput meets the desired speedup.

    Args:
        vllm_qps: Achieved queries-per-second when using vLLM.
        baseline_qps: Reference throughput from a baseline implementation.
        target_ratio: Desired multiplicative speedup over the baseline.

    Raises:
        ValueError: If ``baseline_qps`` is not strictly positive.
    """

    if baseline_qps <= 0:
        raise ValueError("baseline_qps must be positive")
    return (vllm_qps / baseline_qps) >= target_ratio


class VLLMGenerationEngine:
    """Wrapper around ``vllm.LLM`` with streaming-friendly helpers."""

    def __init__(
        self,
        model: Optional[str] = None,
        *,
        llm: Any | None = None,
        tokenizer: Optional[str] = None,
        tensor_parallel_size: int | None = 1,
        dtype: Optional[Union[str, Any]] = None,
        trust_remote_code: bool = False,
        enforce_eager: Optional[bool] = None,
        max_model_len: Optional[int] = None,
        llm_kwargs: Optional[Mapping[str, Any]] = None,
        max_tokens: int = 512,
        top_p: float = 1.0,
        temperature: float = 0.0,
        seed: Optional[int] = None,
    ) -> None:
        if llm is None:
            if _VLLMLLM is None:
                raise ImportError(
                    "vllm is not installed. Either install vllm or provide an ``llm`` instance."
                )
            if model is None:
                raise ValueError("model must be provided when constructing an in-proc vLLM engine")
            init_kwargs: Dict[str, Any] = {"model": model}
            if tokenizer is not None:
                init_kwargs["tokenizer"] = tokenizer
            if tensor_parallel_size is not None:
                init_kwargs["tensor_parallel_size"] = tensor_parallel_size
            if dtype is not None:
                init_kwargs["dtype"] = dtype
            if trust_remote_code:
                init_kwargs["trust_remote_code"] = trust_remote_code
            if enforce_eager is not None:
                init_kwargs["enforce_eager"] = enforce_eager
            if max_model_len is not None:
                init_kwargs["max_model_len"] = max_model_len
            if llm_kwargs:
                init_kwargs.update(dict(llm_kwargs))
            llm = _VLLMLLM(**init_kwargs)  # type: ignore[arg-type]

        self._llm = llm
        self._default_sampling: Dict[str, Any] = {
            "max_tokens": max_tokens,
            "top_p": top_p,
            "temperature": temperature,
            "seed": seed,
        }

    # ------------------------------------------------------------------
    # Default sampling parameter accessors.
    # ------------------------------------------------------------------
    @property
    def llm(self) -> Any:
        """Expose the underlying ``vllm.LLM`` instance for advanced users."""

        return self._llm

    @property
    def max_tokens(self) -> int:
        return int(self._default_sampling["max_tokens"])

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._default_sampling["max_tokens"] = int(value)

    @property
    def top_p(self) -> float:
        return float(self._default_sampling["top_p"])

    @top_p.setter
    def top_p(self, value: float) -> None:
        self._default_sampling["top_p"] = float(value)

    @property
    def temperature(self) -> float:
        return float(self._default_sampling["temperature"])

    @temperature.setter
    def temperature(self, value: float) -> None:
        self._default_sampling["temperature"] = float(value)

    @property
    def seed(self) -> Optional[int]:
        seed = self._default_sampling.get("seed")
        return int(seed) if seed is not None else None

    @seed.setter
    def seed(self, value: Optional[int]) -> None:
        self._default_sampling["seed"] = int(value) if value is not None else None

    # ------------------------------------------------------------------
    # Public generation APIs.
    # ------------------------------------------------------------------
    def generate(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        **sampling_overrides: Any,
    ) -> List[VLLMGeneration]:
        """Blocking generation for a batch of prompts."""

        prompt_list = self._normalize_prompts(prompts)
        sampling_params = self._build_sampling_params(
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **sampling_overrides,
        )
        validated_prompt_ids = self._validate_prompt_token_ids(prompt_list, prompt_token_ids)
        raw_outputs = self._llm.generate(  # type: ignore[call-arg]
            prompt_list,
            sampling_params,
            use_tqdm=False,
            stream=False,
            prompt_token_ids=validated_prompt_ids,
        )
        return list(self._finalize_outputs(raw_outputs))

    def stream_generate(
        self,
        prompts: Union[str, Sequence[str]],
        *,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        seed: Optional[int] = None,
        prompt_token_ids: Optional[Sequence[Sequence[int]]] = None,
        **sampling_overrides: Any,
    ) -> Iterator[VLLMStreamResponse]:
        """Stream tokens for a batch of prompts."""

        prompt_list = self._normalize_prompts(prompts)
        sampling_params = self._build_sampling_params(
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature,
            seed=seed,
            **sampling_overrides,
        )
        validated_prompt_ids = self._validate_prompt_token_ids(prompt_list, prompt_token_ids)
        iterator = self._llm.generate(  # type: ignore[call-arg]
            prompt_list,
            sampling_params,
            use_tqdm=False,
            stream=True,
            prompt_token_ids=validated_prompt_ids,
        )
        return self._stream_outputs(iterator)

    # ------------------------------------------------------------------
    # Internal helpers.
    # ------------------------------------------------------------------
    def _build_sampling_params(
        self,
        *,
        max_tokens: Optional[int],
        top_p: Optional[float],
        temperature: Optional[float],
        seed: Optional[int],
        **overrides: Any,
    ) -> SamplingParams:
        params = dict(self._default_sampling)
        if max_tokens is not None:
            params["max_tokens"] = int(max_tokens)
        if top_p is not None:
            params["top_p"] = float(top_p)
        if temperature is not None:
            params["temperature"] = float(temperature)
        if seed is not None:
            params["seed"] = int(seed)
        params.update(overrides)
        cleaned = {key: value for key, value in params.items() if value is not None}
        return SamplingParams(**cleaned)

    @staticmethod
    def _normalize_prompts(prompts: Union[str, Sequence[str]]) -> List[str]:
        if isinstance(prompts, str):
            return [prompts]
        if not isinstance(prompts, Sequence):  # pragma: no cover - defensive
            raise TypeError("prompts must be a string or a sequence of strings")
        return list(prompts)

    @staticmethod
    def _validate_prompt_token_ids(
        prompts: Sequence[str], prompt_token_ids: Optional[Sequence[Sequence[int]]]
    ) -> Optional[List[List[int]]]:
        if prompt_token_ids is None:
            return None
        prompt_ids_list = [list(item) for item in prompt_token_ids]
        if len(prompt_ids_list) != len(prompts):
            raise ValueError("prompt_token_ids must match the number of prompts")
        return prompt_ids_list

    @staticmethod
    def _safe_getattr(obj: Any, name: str, default: Any = None) -> Any:
        return getattr(obj, name, default)

    def _finalize_outputs(self, outputs: Iterable[Any]) -> Iterator[VLLMGeneration]:
        for request_output in outputs:
            prompt = self._safe_getattr(request_output, "prompt", "")
            request_id = self._safe_getattr(request_output, "request_id", prompt)
            metrics = self._safe_getattr(request_output, "metrics", {}) or {}
            for index, candidate in enumerate(self._safe_getattr(request_output, "outputs", []) or []):
                text = self._safe_getattr(candidate, "text", "")
                token_ids = list(self._safe_getattr(candidate, "token_ids", []) or [])
                logprobs = self._safe_getattr(candidate, "logprobs", None)
                if logprobs is not None:
                    logprobs = list(logprobs)
                finish_reason = self._safe_getattr(candidate, "finish_reason", None)
                yield VLLMGeneration(
                    prompt=prompt,
                    request_id=request_id,
                    text=text,
                    token_ids=token_ids,
                    logprobs=logprobs,
                    finish_reason=finish_reason,
                    metrics=metrics,
                    index=index,
                )

    def _stream_outputs(self, iterator: Iterable[Any]) -> Iterator[VLLMStreamResponse]:
        def generator() -> Iterator[VLLMStreamResponse]:
            seen_text: MutableMapping[Tuple[str, int], str] = {}
            for request_output in iterator:
                prompt = self._safe_getattr(request_output, "prompt", "")
                request_id = self._safe_getattr(request_output, "request_id", prompt)
                metrics = self._safe_getattr(request_output, "metrics", {}) or {}
                finished_request = bool(self._safe_getattr(request_output, "finished", False))
                candidates = self._safe_getattr(request_output, "outputs", []) or []
                if not candidates:
                    yield VLLMStreamResponse(
                        prompt=prompt,
                        request_id=request_id,
                        text="",
                        delta="",
                        token_ids=(),
                        logprobs=None,
                        finished=finished_request,
                        finish_reason=None,
                        metrics=metrics,
                        index=0,
                    )
                    continue
                for index, candidate in enumerate(candidates):
                    full_text = self._safe_getattr(candidate, "text", "") or ""
                    key = (request_id, index)
                    previous_text = seen_text.get(key, "")
                    delta = full_text[len(previous_text) :] if full_text.startswith(previous_text) else full_text
                    seen_text[key] = full_text
                    token_ids = list(self._safe_getattr(candidate, "token_ids", []) or [])
                    logprobs = self._safe_getattr(candidate, "logprobs", None)
                    if logprobs is not None:
                        logprobs = list(logprobs)
                    finish_reason = self._safe_getattr(candidate, "finish_reason", None)
                    finished = finished_request and finish_reason is not None
                    yield VLLMStreamResponse(
                        prompt=prompt,
                        request_id=request_id,
                        text=full_text,
                        delta=delta,
                        token_ids=token_ids,
                        logprobs=logprobs,
                        finished=finished,
                        finish_reason=finish_reason,
                        metrics=metrics,
                        index=index,
                    )

        return generator()


__all__ = [
    "VLLMGeneration",
    "VLLMGenerationEngine",
    "VLLMStreamResponse",
    "meets_perf_target",
]

