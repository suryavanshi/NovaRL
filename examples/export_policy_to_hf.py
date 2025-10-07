"""Convert NovaRL policy checkpoints into Hugging Face format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import torch

try:  # pragma: no cover - optional dependency guard
    from core.models.hf_policy import TinyPreferencePolicyModel
except ImportError as exc:  # pragma: no cover - lazy error surfacing
    _HF_IMPORT_ERROR: Exception | None = exc
else:  # pragma: no cover - clean path when deps are present
    _HF_IMPORT_ERROR = None


def _load_state_dict(path: Path) -> Mapping[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, Mapping) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, Mapping):
        state = checkpoint
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected a dict containing 'model_state_dict'."
        )
    return state


def _infer_shapes(state: Mapping[str, torch.Tensor]) -> tuple[int, int, int]:
    try:
        encoder_weight = state["encoder.0.weight"]
        policy_weight = state["policy_head.weight"]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise KeyError(
            "Checkpoint does not match TinyPreferencePolicy layout."
        ) from exc

    observation_dim = encoder_weight.shape[1]
    hidden_size = encoder_weight.shape[0]
    action_dim = policy_weight.shape[0]
    return observation_dim, action_dim, hidden_size


def export_policy(checkpoint: Path, output_dir: Path, *, activation: str = "tanh") -> None:
    if _HF_IMPORT_ERROR is not None:  # pragma: no cover - dependency guard
        raise _HF_IMPORT_ERROR
    state = _load_state_dict(checkpoint)
    observation_dim, action_dim, hidden_size = _infer_shapes(state)
    model = TinyPreferencePolicyModel.from_state_dict(
        state,
        observation_dim=observation_dim,
        action_dim=action_dim,
        hidden_size=hidden_size,
        activation=activation,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    metadata_path = output_dir / "novarl_policy_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(model.export_metadata(), handle, indent=2)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI plumbing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the torch policy checkpoint (policy.pt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the Hugging Face artefacts will be written.",
    )
    parser.add_argument(
        "--activation",
        default="tanh",
        choices=["tanh", "relu"],
        help="Activation function used by the policy network.",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    export_policy(args.checkpoint, args.output_dir, activation=args.activation)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

