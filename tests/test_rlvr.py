from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from core.interfaces import EnvStep
from rewards.rlvr.manager import VerifiableRewardManager
from rewards.rlvr.signals import (
    HTTPRewardModelSignal,
    MathAnswerSignal,
    RegexMatchSignal,
    UnitTestSignal,
)


def make_env_step(infos: list[dict[str, object]]) -> EnvStep:
    batch = len(infos)
    observations = torch.zeros(batch, 3)
    rewards = torch.zeros(batch)
    dones = torch.zeros(batch)
    return EnvStep(observations=observations, rewards=rewards, dones=dones, infos=infos)


def test_math_answer_signal() -> None:
    signal = MathAnswerSignal(name="math", answer_key="answer", completion_key="completion")
    infos = [
        {"completion": "We compute carefully. Answer: 42", "answer": 42},
        {"completion": "Final response is 13", "answer": 10},
    ]
    result = signal(infos)
    assert pytest.approx(result.raw.tolist()) == [1.0, 0.0]


def test_regex_signal_pattern_key() -> None:
    signal = RegexMatchSignal(
        name="format",
        pattern_key="pattern",
        completion_key="completion",
        positive_reward=0.2,
        negative_reward=-0.1,
    )
    infos = [
        {"completion": "Answer: 7", "pattern": r"Answer:\s*\d+"},
        {"completion": "No prefix", "pattern": r"Answer:\s*\d+"},
    ]
    result = signal(infos)
    assert pytest.approx(result.raw.tolist()) == [0.2, -0.1]


def test_unit_test_signal_partial_pass() -> None:
    signal = UnitTestSignal(
        name="unit",
        completion_key="completion",
        tests_key="tests",
        entry_point_key="entry",
        success_reward=1.0,
        failure_reward=-1.0,
    )
    good_impl = "def solve(x):\n    return x * x\n"
    bad_impl = "def solve(x):\n    return x + 1\n"
    infos = [
        {
            "completion": good_impl,
            "tests": [{"args": [2], "expected": 4}, {"args": [3], "expected": 9}],
            "entry": "solve",
        },
        {
            "completion": bad_impl,
            "tests": [{"args": [2], "expected": 4}, {"args": [3], "expected": 9}],
            "entry": "solve",
        },
    ]
    result = signal(infos)
    # Good implementation gets the full success reward, the bad one fails both tests.
    assert pytest.approx(result.raw.tolist()) == [1.0, -1.0]


def test_http_reward_signal_stub() -> None:
    def client(payload: dict[str, object]) -> list[float]:
        return [float(len(text)) for text in payload["completions"]]

    signal = HTTPRewardModelSignal(
        name="http",
        endpoint="http://unused",
        completion_key="completion",
        prompt_key="prompt",
        client=client,
    )
    infos = [
        {"completion": "abc", "prompt": "p"},
        {"completion": "abcd", "prompt": "q"},
    ]
    result = signal(infos)
    assert pytest.approx(result.raw.tolist()) == [3.0, 4.0]


def test_reward_manager_mixes_signals() -> None:
    math_signal = MathAnswerSignal(name="math", answer_key="answer", completion_key="completion")
    regex_signal = RegexMatchSignal(
        name="regex",
        pattern_key="pattern",
        completion_key="completion",
        positive_reward=0.5,
        negative_reward=0.0,
        weight=0.2,
    )
    manager = VerifiableRewardManager(
        signals=[math_signal, regex_signal],
        normalize_weights=True,
    )
    infos = [
        {"completion": "Answer: 10", "answer": 10, "pattern": r"Answer:\s*\d+"},
        {"completion": "Final=5", "answer": 7, "pattern": r"Answer:\s*\d+"},
    ]
    step = make_env_step(infos)
    rewards = manager.score(step)
    assert rewards.shape == (2,)
    assert rewards[0] > rewards[1]
    assert "math" in manager.latest_histograms

