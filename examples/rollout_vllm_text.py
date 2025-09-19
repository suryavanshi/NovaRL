"""Run batched text rollouts against a vLLM backend."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from engines.vllm_engine import VLLMGenerationEngine, VLLMStreamResponse, meets_perf_target


def _load_prompts(path: Path | None, batch_size: int) -> List[str]:
    if path is None:
        prompt = "Summarise the benefits of paged attention for reinforcement learning."
        return [prompt for _ in range(batch_size)]
    raw: Sequence[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                raw.append(line)
    if not raw:
        raise ValueError(f"Prompt file '{path}' did not contain any text lines")
    return [raw[i % len(raw)] for i in range(batch_size)]


def _collect_stream(
    responses: Iterable[VLLMStreamResponse],
) -> tuple[float, List[float], List[VLLMStreamResponse]]:
    start_time = time.perf_counter()
    latencies: List[float] = []
    in_flight: dict[str, float] = defaultdict(lambda: start_time)
    captured: List[VLLMStreamResponse] = []
    for chunk in responses:
        captured.append(chunk)
        if chunk.finished:
            latency = time.perf_counter() - in_flight.pop(chunk.request_id, start_time)
            latencies.append(latency)
    elapsed = time.perf_counter() - start_time
    return elapsed, latencies, captured


def main() -> None:  # pragma: no cover - CLI utility
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=str, required=True, help="vLLM model identifier")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer path")
    parser.add_argument("--tensor-parallel-size", type=int, default=1, help="Tensor parallelism")
    parser.add_argument("--batch-size", type=int, default=8, help="Concurrent prompts per batch")
    parser.add_argument("--num-batches", type=int, default=10, help="Number of timed batches")
    parser.add_argument("--warmup-batches", type=int, default=1, help="Warmup batches to ignore")
    parser.add_argument("--max-tokens", type=int, default=128, help="Generation length cap")
    parser.add_argument("--top-p", type=float, default=1.0, help="Nucleus sampling parameter")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=None, help="Optional RNG seed")
    parser.add_argument("--prompt-file", type=Path, default=None, help="File containing seed prompts")
    parser.add_argument(
        "--perf-baseline-qps",
        type=float,
        default=None,
        help="Baseline throughput for the quick performance sanity check",
    )
    parser.add_argument(
        "--perf-target",
        type=float,
        default=10.0,
        help="Expected speedup ratio over the baseline implementation",
    )
    parser.add_argument(
        "--histogram-bins",
        type=int,
        default=10,
        help="Number of bins to use for the latency histogram",
    )
    args = parser.parse_args()

    engine = VLLMGenerationEngine(
        model=args.model,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        temperature=args.temperature,
        seed=args.seed,
    )

    prompts = _load_prompts(args.prompt_file, args.batch_size)

    if args.num_batches <= 0:
        raise ValueError("num_batches must be positive")

    warmup = max(args.warmup_batches, 0)
    total_batches = warmup + args.num_batches
    qps_samples: List[float] = []
    latency_samples: List[float] = []

    for batch_idx in range(total_batches):
        stream = engine.stream_generate(prompts)
        elapsed, latencies, _ = _collect_stream(stream)
        if batch_idx >= warmup:
            qps_samples.append(len(prompts) / max(elapsed, 1e-9))
            latency_samples.extend(latencies)

    avg_qps = statistics.fmean(qps_samples)
    std_qps = statistics.pstdev(qps_samples) if len(qps_samples) > 1 else 0.0
    latency_array = np.array(latency_samples, dtype=np.float64) if latency_samples else np.array([])
    histogram = None
    if latency_array.size:
        histogram = np.histogram(latency_array, bins=args.histogram_bins)

    print("=== vLLM Rollout Summary ===")
    print(f"Batches timed         : {len(qps_samples)}")
    print(f"Average QPS           : {avg_qps:.3f} ± {std_qps:.3f}")
    if latency_array.size:
        print(f"Latency p50 / p95     : {np.percentile(latency_array, [50, 95])}")
    if histogram is not None:
        buckets = {"edges": histogram[1].tolist(), "counts": histogram[0].tolist()}
        print("Latency histogram     :", json.dumps(buckets))
    print("PagedAttention        : enabled (vLLM default)")

    if args.perf_baseline_qps is not None:
        achieved = meets_perf_target(avg_qps, args.perf_baseline_qps, args.perf_target)
        ratio = avg_qps / max(args.perf_baseline_qps, 1e-9)
        status = "PASS" if achieved else "FAIL"
        print(f"Performance target    : {status} (ratio={ratio:.2f}x vs baseline)")


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

