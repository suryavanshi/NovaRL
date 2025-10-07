# Choosing a Distributed Training Backend

NovaRL interoperates with several distributed training strategies. This guide
summarises when to reach for Fully Sharded Data Parallel (FSDP), DeepSpeed ZeRO
and Megatron-LM style tensor-parallelism along with the practical trade-offs you
should consider when scaling PPO-style reinforcement learning pipelines.

## Overview

| Backend            | Memory Footprint                                                                                                             | Throughput Profile                                                                                               | Ideal For                                                                                                   |
|--------------------|------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------|
| **FSDP**           | Shards parameters, gradients and optimizer state across ranks.<br>Peak memory per device is roughly ``O(total_parameters / world_size)`` when using ZeRO-3 style sharding.<br>Optional CPU offload reduces GPU pressure further at the cost of PCIe latency. | Excellent for moderate batch sizes when activation checkpointing is enabled.<br>Sequential all-gather/reduce phases introduce latency but are amortised in large batches. | Research iterations that must fit a large language model on commodity accelerators, or workloads with highly variable sequence lengths. |
| **DeepSpeed ZeRO** | Stage 2 shards optimizer states; Stage 3 shards parameters and gradients as well.<br>Requires partitioning metadata but provides predictable memory savings.<br>CPU/NVMe offload supported. | Stage 2 performs similarly to DDP when activation checkpointing is disabled.<br>Stage 3 introduces extra communication but unlocks larger models. | Production fine-tuning with stable batch sizes where checkpoint/optimizer state partitioning is desirable. |
| **Megatron (TP/PP)** | Tensor parallelism splits weight matrices across devices while pipeline parallelism divides layers.<br>Memory savings come from splitting activation storage; optimizer state often remains replicated.<br>Requires careful model surgery. | Highest throughput at scale for autoregressive transformers once microbatching and pipeline balancing are tuned.<br>Communication overhead is dominated by intra-layer collectives. | Massive models (70B+) where multiple GPUs must cooperate on individual forward passes.<br>Works best in homogeneous clusters with NVLink or Infiniband interconnects. |

## Decision Checklist

1. **Model size vs. device memory** – If the tuned policy fits on a single GPU
   with headroom for activations, vanilla DDP may be sufficient. Otherwise
   prefer FSDP or ZeRO-3 to shard state across devices.
2. **Latency tolerance** – FSDP and ZeRO introduce additional collective ops.
   Latency-sensitive applications (e.g. RL with tight inference loops) often
   trade some memory efficiency for lower variance in step time.
3. **Communication fabric** – Megatron assumes high-bandwidth, low-latency links
   (NVLink, NVSwitch, Infiniband). On Ethernet clusters the synchronisation cost
   may outweigh throughput gains.
4. **Checkpoint portability** – ZeRO stores partitioned optimizer state while
   FSDP exposes composite checkpoints compatible with Hugging Face out of the
   box. Megatron checkpoints encode tensor-parallel layout and require dedicated
   loaders during inference/export.

## Recommended Configurations

- **FSDP**: Use the preset exposed via `core.distributed.get_default_fsdp_preset`
  and enable activation checkpointing for long sequences. Pair with
  gradient-accumulation to reach target batch sizes.
- **DeepSpeed ZeRO-3**: Start from `scripts/launch_zero3.sh`, enabling optimizer
  and parameter offload when GPUs are memory constrained. Keep gradient
  accumulation modest to avoid CPU/NVMe bottlenecks.
- **Megatron-style tensor parallelism**: Combine with RL policies that reuse
  inference-time tensor-parallel checkpoints (e.g. Llama or Falcon). Integrate
  with vLLM for serving by exporting shard metadata alongside weights.

## Throughput vs. Memory Rule of Thumb

- FSDP (ZeRO-3 equivalent) yields **~4× memory savings** over DDP with a
  **10–20% throughput hit** for medium batch sizes.
- DeepSpeed ZeRO-2 delivers **~2× memory reduction** with **near-DDP throughput**,
  while ZeRO-3 approaches FSDP memory efficiency at the cost of extra
  communication.
- Megatron tensor parallelism typically provides **linear throughput scaling up
  to 8 GPUs** per model replica provided microbatches keep pipelines busy. Memory
  savings stem from partitioning activations; optimizer state remains replicated
  unless combined with ZeRO.

In practice many large-scale systems blend these strategies: e.g. ZeRO-1 or
ZeRO-2 for optimizer sharding, tensor-parallel for intra-layer scaling and FSDP
for encoder/decoder stacks. NovaRL exposes the relevant knobs via the
configuration presets so you can compose the right mix for your hardware.

