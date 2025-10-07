from __future__ import annotations

import torch

from core.monitoring import (
    CodeEvalTask,
    EvalResult,
    KLDriftAlarm,
    MathEvalTask,
    PeriodicEvalHook,
    RewardHackingSentinel,
    ScriptedAgentFactory,
    VerbalEvalTask,
    render_html_trace,
    render_text_trace,
)
from core.types import TrajectoryBatch


def _default_scripts():
    return [
        [("math evaluation", "Ready for math."), ("2 + 3", "The answer is 5")],
        [("math evaluation", "Ready again."), ("10 / 2", "5")],
        [("math evaluation", "Still ready."), ("7 * 4", "28")],
        [
            ("coding assignment", "Ready to code."),
            ("square", "def square(x):\n    return x * x"),
            ("test this function", "Use asserts with square(2) == 4"),
        ],
        [
            ("summarize", "Ready to summarize."),
            (
                "Context:",
                "The mission is returning with discoveries and inspires optimism.",
            ),
        ],
    ]


def test_eval_tasks_with_scripted_agent():
    factory = ScriptedAgentFactory(_default_scripts(), loop=True)
    math_result = MathEvalTask().run(factory)
    code_result = CodeEvalTask().run(factory)
    verbal_result = VerbalEvalTask().run(factory)
    assert math_result.passed and math_result.score == 1.0
    assert code_result.passed and code_result.score == 1.0
    assert verbal_result.passed and verbal_result.score == 1.0


def test_periodic_eval_hook_runs_on_frequency():
    factory = ScriptedAgentFactory(_default_scripts(), loop=True)
    hook = PeriodicEvalHook(
        tasks=[MathEvalTask(), CodeEvalTask(), VerbalEvalTask()],
        agent_factory=factory,
        frequency=2,
        reward_sentinel=RewardHackingSentinel(reward_threshold=0.3, min_eval_score=0.6),
        kl_alarm=KLDriftAlarm(max_kl=0.5),
    )
    metrics = {"reward_mean": 0.1, "kl": 0.05, "kl_to_ref": 0.04}
    results, warnings = hook.maybe_run(0, metrics)
    assert len(results) == 3
    assert warnings == ()
    results, warnings = hook.maybe_run(1, metrics)
    assert results == ()
    assert warnings == ()


def test_reward_hacking_sentinel_flags_when_eval_low():
    sentinel = RewardHackingSentinel(reward_threshold=0.8, min_eval_score=0.6)
    metrics = {"reward_mean": 0.9}
    results = [
        EvalResult(name="math", score=0.2, passed=False, details={}),
        EvalResult(name="code", score=0.1, passed=False, details={}),
    ]
    warnings = sentinel.check(metrics, results)
    assert warnings and "Reward hacking" in warnings[0]


def test_kl_drift_alarm_triggers_when_threshold_crossed():
    alarm = KLDriftAlarm(max_kl=0.1, max_reference_kl=0.15)
    metrics = {"kl": 0.2, "kl_to_ref": 0.3}
    warnings = alarm.check(metrics)
    assert len(warnings) == 2


def test_render_traces_formats_output():
    observations = torch.arange(6, dtype=torch.float32).reshape(2, 1, 3)
    actions = torch.arange(2, dtype=torch.float32).reshape(2, 1, 1)
    log_probs = torch.zeros(2, 1)
    rewards = torch.tensor([[0.5], [1.0]], dtype=torch.float32)
    dones = torch.tensor([[0.0], [1.0]], dtype=torch.float32)
    values = torch.zeros(2, 1)
    advantages = torch.zeros(2, 1)
    returns = torch.zeros(2, 1)
    batch = TrajectoryBatch(
        observations=observations,
        actions=actions,
        log_probs=log_probs,
        rewards=rewards,
        dones=dones,
        values=values,
        advantages=advantages,
        returns=returns,
    )
    text_trace = render_text_trace(batch)
    html_trace = render_html_trace(batch)
    assert "step=0 env=0" in text_trace
    assert "observation:" in text_trace
    assert html_trace.startswith("<table>")
    assert "<td>0</td>" in html_trace
