# NovaRL

NovaRL is a lightweight reinforcement learning sandbox focused on reinforcement learning from human feedback (RLHF) experiments. The codebase favors clarity over completeness so that individual components such as rollout engines, buffers, and trainers can be understood and customized quickly.

## Key features

- **Composable interfaces** for environments, rollout engines, trainers, buffers, and reward managers.
- **Synchronous PPO implementation** supporting both clipping and KL-penalty modes.
- **Reference model utilities** for RLHF stabilization, including automatic frozen copies and optional weight-tied references.
- **Structured logging with optional JSON, TensorBoard, and W&B sinks** plus resilient checkpoints and resume scripts.
- **Examples that run in minutes**, including a tiny single-turn RLHF set-up.

## Getting started

NovaRL targets Python 3.10+ and PyTorch 2.x. Clone the repository and install it in editable mode:

```bash
pip install -e .

# Large-scale language model extras (FSDP/DeepSpeed)
pip install -e .[llm]
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

## Large language model fine-tuning presets

NovaRL ships with launcher scripts that wrap Hugging Face's `Trainer` to fine-tune 7B
class language models on 4–8 GPUs using either native PyTorch FSDP or DeepSpeed
ZeRO. The presets enable gradient checkpointing, bf16 training, and optimizer state
sharding out of the box while exposing toggles for CPU offload and precision control.

- `scripts/launch_fsdp.sh` – launches an FSDP-backed run. Gradient checkpointing and
  bf16 are enabled by default with optional CPU offload via `CPU_OFFLOAD=1`.
- `scripts/launch_zero3.sh` – launches a DeepSpeed ZeRO stage 2 or 3 run. Enable
  optimizer or parameter offloading through `OPTIMIZER_OFFLOAD=1` and
  `PARAM_OFFLOAD=1`. Set `ZERO_STAGE=2` to switch to a ZeRO-2 preset.

Both scripts accept the standard Hugging Face dataset arguments (`DATASET_NAME`,
`DATASET_CONFIG`, `TEXT_FIELD`, etc.) and call into
[`scripts/finetune_hf.py`](scripts/finetune_hf.py), which exposes the shared
fine-tuning workflow. The DeepSpeed launch script relies on JSON presets stored
under [`configs/deepspeed/`](configs/deepspeed), regenerating a custom
configuration automatically when offload or sharding settings are overridden.

## Contributing

Contributions are welcome! Feel free to open issues or pull requests that improve documentation, add environments, or extend the PPO trainer.
