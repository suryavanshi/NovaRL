from __future__ import annotations

import html
from typing import Iterable

import torch

from core.types import TrajectoryBatch


def _format_tensor(tensor: torch.Tensor) -> str:
    flat = tensor.detach().cpu().reshape(-1).tolist()
    return ", ".join(f"{value:.3f}" for value in flat)


def render_text_trace(batch: TrajectoryBatch) -> str:
    """Render a trajectory batch to a plain-text transcript."""

    lines = []
    for t in range(batch.horizon):
        for b in range(batch.batch_size):
            lines.append(f"step={t} env={b}")
            obs = batch.observations[t, b]
            action = batch.actions[t, b]
            reward = batch.rewards[t, b]
            done = batch.dones[t, b]
            lines.append(f"  observation: {_format_tensor(obs)}")
            lines.append(f"  action: {_format_tensor(action)}")
            lines.append(f"  reward: {float(reward):.3f} done={bool(done.item())}")
    return "\n".join(lines)


def render_html_trace(batch: TrajectoryBatch) -> str:
    """Render a trajectory batch into a lightweight HTML table."""

    rows: Iterable[str] = []
    rows = []
    for t in range(batch.horizon):
        for b in range(batch.batch_size):
            obs = html.escape(_format_tensor(batch.observations[t, b]))
            action = html.escape(_format_tensor(batch.actions[t, b]))
            reward = float(batch.rewards[t, b])
            done = bool(batch.dones[t, b].item())
            rows.append(
                "<tr>"
                f"<td>{t}</td>"
                f"<td>{b}</td>"
                f"<td>{obs}</td>"
                f"<td>{action}</td>"
                f"<td>{reward:.3f}</td>"
                f"<td>{str(done).lower()}</td>"
                "</tr>"
            )
    table_rows = "\n".join(rows)
    return (
        "<table>"
        "<thead><tr><th>step</th><th>env</th><th>observation</th><th>action</th><th>reward</th><th>done</th></tr></thead>"
        f"<tbody>{table_rows}</tbody>"
        "</table>"
    )


__all__ = ["render_text_trace", "render_html_trace"]
