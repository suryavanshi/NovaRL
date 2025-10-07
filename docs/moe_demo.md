# Mixture-of-Experts quickstart

This guide demonstrates how to exercise the lightweight MoE helpers that ship
with NovaRL.  The demo runs on the same Hydra controller that drives the other
examples and exposes flags for Megatron/DeepSpeed layout selection, algorithm
choice (PPO or GRPO), and asynchronous weight synchronisation cadence.

## Parallel layout helpers

The new `ParallelLayout` and `MoERouterConfig` helpers live under
`core.distributed`.  They encode the relationship between data-parallel (DP),
tensor-parallel (TP), pipeline-parallel (PP), and expert-parallel (EP) sizes
and can emit dictionaries suitable for Megatron-MoE launch arguments or the
DeepSpeed-MoE JSON stanza.

```python
from core.distributed import ParallelLayout, MoERouterConfig

layout = ParallelLayout.from_world_size(
    total_world_size=8, tensor_parallel_size=2, pipeline_parallel_size=2, expert_parallel_size=2
)
router = MoERouterConfig(num_experts=8, top_k=2, capacity_factor=1.2)

print(layout.describe())
# {'dp': 1, 'tp': 2, 'pp': 2, 'ep': 2, 'world': 8}
print(router.to_deepspeed(layout))
```

Megatron adapters now expose the same layout via `MegatronAdapterConfig.parallel_layout`,
ensuring EP/TP/PP values remain consistent regardless of the launch backend.

## Running the MoE demo

The `examples/moe_demo.py` module creates a tiny Mixtral-style policy with a
weighted routing layer.  It shares the synchronous rollout engine with the
existing tutorials and consumes the new advantage interface so that PPO and
GRPO may be swapped without touching the environment loop.

Launch it through the common controller:

```bash
python scripts/train.py mode=moe_demo \
  experiment.world_size=8 \
  experiment.tensor_parallel_size=2 \
  experiment.pipeline_parallel_size=2 \
  experiment.expert_parallel_size=2 \
  experiment.num_experts=4 \
  experiment.algorithm=grpo
```

The demo prints a checklist summarising the DP/TP/PP/EP factors and router
hyper-parameters before stepping through a handful of training iterations.

## Algorithm options: PPO and GRPO

Both PPO and GRPO are wired through a shared advantage/return estimator.  The
`GAEAdvantageEstimator` lives in `core.utils.advantages` and provides
standardised normalisation via `normalize_advantages`, eliminating duplicate
z-scoring logic across trainers.

Pick the optimiser from the Hydra config using `experiment.algorithm`.
GRPO additionally accepts `experiment.grpo_group_size` which controls the
relative baseline window.

## Weight-sync cadence for async rollouts

Asynchronous PPO experiments can now specify cadence in terms of update
interval, maximum policy staleness, and an optional timeout.  The new
`WeightSyncController` (defined in `examples/ppo_async.py`) replaces the raw
modulo logic and ensures resumed runs, worker restarts, and staleness spikes
force a broadcast when needed.  The relevant Hydra flags are:

- `experiment.weight_sync_interval`
- `experiment.weight_sync_max_staleness`
- `experiment.weight_sync_timeout_s`

Combine them according to your rollout topology—for example broadcast every
4 updates but also whenever any worker lags by 6 policy versions, or at least
every 30 seconds.
