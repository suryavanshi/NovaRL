# NovaRL

NovaRL is a lightweight reinforcement learning sandbox focused on reinforcement learning from human feedback (RLHF) experiments. The codebase favors clarity over completeness so that individual components such as rollout engines, buffers, and trainers can be understood and customized quickly.

## Key features

- **Composable interfaces** for environments, rollout engines, trainers, buffers, and reward managers.
- **Synchronous PPO implementation** supporting both clipping and KL-penalty modes.
- **Reference model utilities** for RLHF stabilization, including automatic frozen copies and optional weight-tied references.
- **Simple timing, logging, and checkpoint helpers** to make toy experiments easy to reproduce.
- **Examples that run in minutes**, including a tiny single-turn RLHF set-up.

## Getting started

NovaRL targets Python 3.10+ and PyTorch 2.x. Clone the repository and install it in editable mode:

```bash
pip install -e .
```

### Running the toy PPO example

The classic control-style PPO loop used for sanity checks lives in [`examples/minimal_ppo_sync.py`](examples/minimal_ppo_sync.py). Launch it with:

```bash
PYTHONPATH=. python examples/minimal_ppo_sync.py
```

The script trains a tiny feed-forward policy on a synthetic prompt environment and logs reward, KL divergence, entropy, and throughput.

### RLHF single-turn PPO example

The RLHF-focused demo in [`examples/ppo_rlhf_single_turn.py`](examples/ppo_rlhf_single_turn.py) shows how to stabilize PPO with a KL penalty against a frozen reference model:

```bash
PYTHONPATH=. python examples/ppo_rlhf_single_turn.py
```

The example:

1. Builds a four-prompt preference dataset with deterministic per-action rewards.
2. Creates a `SingleTurnPreferenceEnvironment` that serves prompts in batches.
3. Runs PPO with clipping, entropy regularization, and a KL penalty against a frozen reference model.
4. Adapts the KL coefficient online to keep divergence within a configurable target window.
5. Logs training metrics including objective value, KL divergences, entropy, and episodes/sec.
6. Saves model and optimizer checkpoints under `checkpoints/ppo_rlhf_single_turn/` and prints the learned action probabilities for each prompt.

Because the environment’s episodes last a single step, the script demonstrates how KL regularization can improve rewards without unstable spikes.

## PPO trainer overview

`algos/ppo/trainer.py` contains the core PPO logic. The trainer accepts optional knobs that are common in RLHF research:

- `clip_range`: enables the standard PPO clipping objective when set to a positive value.
- `kl_coef`: scales a KL penalty computed against a reference policy. Set to zero to disable.
- `adaptive_kl`: toggles automatic adjustment of the KL coefficient. The coefficient increases when the KL exceeds `kl_target * 1.5` and decreases when it drops below `kl_target / 1.5`.
- `reference_model`: pass `"copy"` (default) to create a frozen snapshot, `"tie"` to reuse the policy weights lazily, or provide a custom `nn.Module`.

The trainer logs additional diagnostics such as the unclipped policy objective, KL penalty magnitude, and the KL divergence to both the previous policy (`kl`) and the reference policy (`kl_to_ref`).

## Project structure

```
algos/          # Training algorithms (PPO)
core/           # Shared abstractions, data structures, and utilities
engines/        # Rollout engines (synchronous loops and vLLM adapters)
envs/           # Toy prompt environments and preference datasets
examples/       # Executable training scripts
rewards/        # Reward managers for transforming environment signals
```

## Testing

NovaRL ships with unit tests under `tests/`. Run them with:

```bash
pytest
```

## Contributing

Contributions are welcome! Feel free to open issues or pull requests that improve documentation, add environments, or extend the PPO trainer.
