from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator, List, Sequence

import pytest

from engines.vllm_engine import (
    VLLMGenerationEngine,
    VLLMStreamResponse,
    meets_perf_target,
)


@dataclass
class _FakeCandidate:
    text: str
    token_ids: Sequence[int]
    logprobs: Sequence[float]
    finish_reason: str | None


@dataclass
class _FakeRequestOutput:
    prompt: str
    request_id: str
    outputs: List[_FakeCandidate]
    finished: bool
    metrics: dict[str, Any]


class _FakeLLM:
    def __init__(self) -> None:
        self.last_sampling_params: Any | None = None
        self.last_prompt_token_ids: Any | None = None

    def generate(
        self,
        prompts: Sequence[str],
        sampling_params: Any,
        *,
        use_tqdm: bool,
        stream: bool,
        prompt_token_ids: Sequence[Sequence[int]] | None = None,
    ) -> Iterable[_FakeRequestOutput] | Iterator[_FakeRequestOutput]:
        self.last_sampling_params = sampling_params
        self.last_prompt_token_ids = prompt_token_ids

        def make_output(idx: int, prompt: str, step: int, total_steps: int) -> _FakeRequestOutput:
            text = "|".join(f"{prompt}-{i}" for i in range(step + 1))
            candidate = _FakeCandidate(
                text=text,
                token_ids=list(range(step + 1)),
                logprobs=[-0.1] * (step + 1),
                finish_reason="length" if step + 1 == total_steps else None,
            )
            return _FakeRequestOutput(
                prompt=prompt,
                request_id=f"req-{idx}",
                outputs=[candidate],
                finished=step + 1 == total_steps,
                metrics={"batch_index": idx},
            )

        steps = 2
        if stream:
            def iterator() -> Iterator[_FakeRequestOutput]:
                for idx, prompt in enumerate(prompts):
                    for step in range(steps):
                        yield make_output(idx, prompt, step, steps)

            return iterator()

        outputs: List[_FakeRequestOutput] = []
        for idx, prompt in enumerate(prompts):
            outputs.append(make_output(idx, prompt, steps - 1, steps))
        return outputs


def test_generate_batches_requests_with_sampling_overrides() -> None:
    engine = VLLMGenerationEngine(llm=_FakeLLM(), max_tokens=64, top_p=0.95, temperature=0.2, seed=7)
    responses = engine.generate(["alpha", "beta"], max_tokens=8, temperature=0.5, top_p=0.9, seed=11)
    assert len(responses) == 2
    assert [item.prompt for item in responses] == ["alpha", "beta"]
    assert responses[0].text.endswith("alpha-1")
    fake = engine.llm
    assert fake.last_sampling_params.max_tokens == 8
    assert fake.last_sampling_params.temperature == 0.5
    assert fake.last_sampling_params.top_p == 0.9
    assert fake.last_sampling_params.seed == 11


def test_stream_generate_yields_incremental_updates() -> None:
    engine = VLLMGenerationEngine(llm=_FakeLLM())
    chunks = list(engine.stream_generate(["prompt"]))
    assert len(chunks) == 2
    first, second = chunks
    assert isinstance(first, VLLMStreamResponse)
    assert first.delta == "prompt-0"
    assert not first.finished
    assert second.delta.endswith("prompt-1")
    assert second.finished


def test_prompt_token_ids_forwarded_for_pinned_kv_cache() -> None:
    fake_llm = _FakeLLM()
    engine = VLLMGenerationEngine(llm=fake_llm)
    prompts = ["a", "b"]
    prompt_ids = [[1, 2], [3, 4]]
    engine.generate(prompts, prompt_token_ids=prompt_ids)
    assert fake_llm.last_prompt_token_ids == prompt_ids


@pytest.mark.parametrize("vllm_qps, baseline_qps, expected", [(100.0, 5.0, True), (20.0, 5.0, False)])
def test_meets_perf_target(vllm_qps: float, baseline_qps: float, expected: bool) -> None:
    assert meets_perf_target(vllm_qps, baseline_qps, target_ratio=10.0) is expected


def test_meets_perf_target_requires_positive_baseline() -> None:
    with pytest.raises(ValueError):
        meets_perf_target(10.0, 0.0)

