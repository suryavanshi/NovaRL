# NovaRL Launch Recipes

The scripts in this folder provide ready-to-run launch commands for four common
reinforcement learning workflows. Each recipe comes in a single-node and
multi-node flavour. The single-node variants are designed for quick iteration on
a single workstation or DGX, while the multi-node scripts demonstrate how to use
`torchrun` to scale out.

The scripts assume the repository root as the working directory. Set the
`CUDA_VISIBLE_DEVICES` environment variable before invoking the single-node
variants to choose which GPUs participate in training.

| Workflow     | Single Node Script                   | Multi-Node Script                    |
|--------------|--------------------------------------|--------------------------------------|
| RLHF (PPO)   | `rlhf_ppo_single_node.sh`            | `rlhf_ppo_multi_node.sh`             |
| RLVR (math)  | `rlvr_math_single_node.sh`           | `rlvr_math_multi_node.sh`            |
| Code RL      | `code_rl_single_node.sh`             | `code_rl_multi_node.sh`              |
| VLM (VQA)    | `vlm_vqa_single_node.sh`             | `vlm_vqa_multi_node.sh`              |

To launch, grant execution permissions (once) and then run the desired script:

```bash
chmod +x examples/recipes/*.sh
examples/recipes/rlhf_ppo_single_node.sh
```

Feel free to treat these recipes as templates—swap in your own datasets,
hyper-parameters or parallelism knobs to match production requirements.

