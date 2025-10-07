from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Mapping, MutableSequence, Optional, Protocol, Sequence, Tuple


class ConversationSession(Protocol):
    """Simple multi-turn conversation interface."""

    def send(self, message: str) -> str:
        """Send a message to the agent and receive a response."""


AgentFactory = Callable[[], ConversationSession]


@dataclass
class EvalResult:
    """Structured output from running an evaluation task."""

    name: str
    score: float
    passed: bool
    details: Mapping[str, Any] = field(default_factory=dict)


class EvalTask(Protocol):
    """Protocol for evaluation tasks that operate on an agent factory."""

    name: str

    def run(self, agent_factory: AgentFactory) -> EvalResult:
        ...


@dataclass
class ScriptedAgentSession:
    """Conversation session that replays scripted responses.

    The session validates that prompts are seen in the expected order and can
    optionally enforce that the agent receives prompts containing particular
    substrings. This is primarily useful for tests and for wiring up stubbed
    agents inside examples.
    """

    script: Sequence[Tuple[Optional[str], str]]
    transcript: MutableSequence[Tuple[str, str]] = field(default_factory=list)
    _position: int = 0

    def send(self, message: str) -> str:  # pragma: no cover - thin wrapper
        if self._position >= len(self.script):
            raise RuntimeError("Scripted session exhausted")
        expected, response = self.script[self._position]
        if expected is not None and expected not in message:
            raise AssertionError(
                f"Prompt mismatch at turn {self._position}: expected substring '{expected}' in '{message}'"
            )
        self._position += 1
        self.transcript.append((message, response))
        return response


@dataclass
class ScriptedAgentFactory:
    """Factory that dispenses :class:`ScriptedAgentSession` instances."""

    scripts: Sequence[Sequence[Tuple[Optional[str], str]]]
    loop: bool = False
    _cursor: int = 0

    def __call__(self) -> ScriptedAgentSession:
        if not self.scripts:
            raise RuntimeError("No scripts configured for ScriptedAgentFactory")
        if self._cursor >= len(self.scripts):
            if not self.loop:
                raise RuntimeError("No scripted conversations remaining")
            self._cursor = 0
        script = self.scripts[self._cursor]
        self._cursor += 1
        return ScriptedAgentSession(script)

    def reset(self) -> None:
        """Reset iteration over the scripted conversations."""

        self._cursor = 0


@dataclass
class MathEvalTask:
    """Evaluate elementary math reasoning over a few multi-turn prompts."""

    problems: Sequence[Tuple[str, float]] = (
        ("2 + 3", 5.0),
        ("10 / 2", 5.0),
        ("7 * 4", 28.0),
    )
    acknowledgement_prompt: str = (
        "You are participating in a math evaluation. Confirm when you are ready."
    )
    tolerance: float = 1e-3
    pass_threshold: float = 0.67

    name: str = "math_reasoning"

    def run(self, agent_factory: AgentFactory) -> EvalResult:
        transcripts: List[Mapping[str, Any]] = []
        correct = 0
        for problem, expected in self.problems:
            session = agent_factory()
            acknowledgement = session.send(self.acknowledgement_prompt)
            prompt = f"Solve the problem: {problem}"
            response = session.send(prompt)
            prediction = self._extract_number(response)
            success = prediction is not None and abs(prediction - expected) <= self.tolerance
            if success:
                correct += 1
            transcripts.append(
                {
                    "problem": problem,
                    "expected": expected,
                    "ack": acknowledgement,
                    "response": response,
                    "parsed": prediction,
                    "success": success,
                }
            )
        score = correct / len(self.problems) if self.problems else 0.0
        return EvalResult(
            name=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            details={"transcripts": transcripts},
        )

    @staticmethod
    def _extract_number(text: str) -> Optional[float]:
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None
        try:
            return float(match.group(0))
        except ValueError:
            return None


@dataclass
class CodeEvalTask:
    """Evaluate short-form code generation capabilities."""

    prompts: Sequence[Mapping[str, Any]] = (
        {
            "instruction": "Write a Python function named square that returns x * x.",
            "validators": (
                lambda resp: "def square" in resp,
                lambda resp: "return" in resp,
                lambda resp: "x * x" in resp or "x*x" in resp,
            ),
        },
    )
    acknowledgement_prompt: str = (
        "You will receive a small coding assignment. Confirm readiness."
    )
    follow_up_prompt: str = "How would you test this function?"
    pass_threshold: float = 0.75

    name: str = "code_generation"

    def run(self, agent_factory: AgentFactory) -> EvalResult:
        transcripts: List[Mapping[str, Any]] = []
        successes = 0
        total_checks = 0
        for prompt_spec in self.prompts:
            session = agent_factory()
            acknowledgement = session.send(self.acknowledgement_prompt)
            instruction = prompt_spec["instruction"]
            response = session.send(instruction)
            follow_up = session.send(self.follow_up_prompt)
            validators: Iterable[Callable[[str], bool]] = prompt_spec.get("validators", [])
            checks = [validator(response) for validator in validators]
            total_checks += len(checks)
            successes += sum(1 for passed in checks if passed)
            transcripts.append(
                {
                    "instruction": instruction,
                    "ack": acknowledgement,
                    "response": response,
                    "follow_up": follow_up,
                    "checks": checks,
                }
            )
        score = successes / total_checks if total_checks else 0.0
        return EvalResult(
            name=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            details={"transcripts": transcripts},
        )


@dataclass
class VerbalEvalTask:
    """Evaluate verbal reasoning and summarization skills."""

    scenarios: Sequence[Mapping[str, Any]] = (
        {
            "context": "A spacecraft completed a three-year mission exploring Mars and is returning home with new discoveries.",
            "question": "Summarize the news in two sentences, highlighting optimism.",
            "keywords": ("mission", "returning", "discoveries"),
        },
    )
    acknowledgement_prompt: str = "You will be asked to summarize a short scenario. Confirm readiness."
    pass_threshold: float = 0.8

    name: str = "verbal_reasoning"

    def run(self, agent_factory: AgentFactory) -> EvalResult:
        transcripts: List[Mapping[str, Any]] = []
        keyword_hits = 0
        keyword_total = 0
        for scenario in self.scenarios:
            session = agent_factory()
            acknowledgement = session.send(self.acknowledgement_prompt)
            prompt = f"Context: {scenario['context']}\nQuestion: {scenario['question']}"
            response = session.send(prompt)
            keywords: Sequence[str] = scenario.get("keywords", [])
            hits = [keyword for keyword in keywords if keyword.lower() in response.lower()]
            keyword_hits += len(hits)
            keyword_total += len(keywords)
            transcripts.append(
                {
                    "context": scenario["context"],
                    "response": response,
                    "ack": acknowledgement,
                    "keywords_hit": hits,
                }
            )
        score = keyword_hits / keyword_total if keyword_total else 0.0
        return EvalResult(
            name=self.name,
            score=score,
            passed=score >= self.pass_threshold,
            details={"transcripts": transcripts},
        )


__all__ = [
    "EvalResult",
    "EvalTask",
    "MathEvalTask",
    "CodeEvalTask",
    "VerbalEvalTask",
    "ScriptedAgentFactory",
]
