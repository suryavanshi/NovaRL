"""Helper to run single-process training modules under torchrun."""

from __future__ import annotations

import argparse
import os
import runpy

import torch
import torch.distributed as dist


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help="Python module to execute on rank 0")
    return parser.parse_known_args()


def _maybe_init_process_group() -> None:
    if not dist.is_available():  # pragma: no cover - defensive branch
        return
    if dist.is_initialized():
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)


def main() -> None:  # pragma: no cover - CLI entry point
    args, extra = parse_args()
    _maybe_init_process_group()

    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = int(os.environ.get("RANK", "0"))

    if rank == 0:
        argv = [args.target, *extra]
        import sys

        sys.argv = argv
        runpy.run_module(args.target, run_name="__main__", alter_sys=True)
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
    else:
        if dist.is_available() and dist.is_initialized():
            dist.barrier()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

