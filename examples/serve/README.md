# NovaRL vLLM Serving Guide

This directory contains utilities for exporting RL-tuned checkpoints into the
Hugging Face format and hosting them behind a high-throughput
[vLLM](https://github.com/vllm-project/vllm) server.

## 1. Export the policy to Hugging Face format

NovaRL examples persist checkpoints as PyTorch state dictionaries by default.
Convert them into the Hugging Face layout expected by vLLM (requires the
`transformers` package):

```bash
python -m examples.export_policy_to_hf \
  --checkpoint checkpoints/ppo_rlhf_single_turn/policy.pt \
  --output-dir checkpoints/ppo_rlhf_single_turn_hf
```

The command infers the network dimensions from the saved weights and writes a
standard `config.json` + `pytorch_model.bin` pair alongside
`novarl_policy_metadata.json` for reference.

## 2. Launch the vLLM server

Install vLLM (e.g. `pip install vllm`) and run the OpenAI-compatible API server:

```bash
python -m examples.serve.serve_vllm \
  --model checkpoints/ppo_rlhf_single_turn_hf \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1
```

The script is a thin wrapper around `vllm serve` and accepts the same
performance-related flags such as `--max-model-len`, `--gpu-memory-utilization`
and `--max-num-seqs` for tuning throughput.

## 3. Measure throughput with batched rollouts

With the server running, benchmark generation latency and throughput using the
streaming rollout helper:

```bash
python -m examples.rollout_vllm_text \
  --model checkpoints/ppo_rlhf_single_turn_hf \
  --tensor-parallel-size 1 \
  --batch-size 32 \
  --num-batches 20 \
  --warmup-batches 2 \
  --perf-baseline-qps 1.0 \
  --perf-target 10.0
```

The summary emitted at the end of the run reports average QPS, latency
histograms and whether the measured throughput clears the configured speed-up
target relative to a baseline implementation. Adjust the batch size or
parallelism flags to explore the throughput vs. latency trade-off space. When
the server is active you can point the `--model` flag to the exported directory
for an in-process benchmark, or use the native `vllm serve` client tooling to
exercise the OpenAI endpoint.

