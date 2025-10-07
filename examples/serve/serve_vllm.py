"""Launch a vLLM OpenAI-compatible server for NovaRL checkpoints."""

from __future__ import annotations

import argparse
import logging


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        required=True,
        help="Path or Hugging Face identifier for the RL-tuned checkpoint exported in HF format.",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Optional tokenizer path (defaults to the model identifier).",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to expose the OpenAI API")
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree passed through to vLLM",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Optional sequence length override for vLLM",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="Target fraction of GPU memory to utilise",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help="Optional cap on concurrent sequences",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Verbosity for the local logger as well as the vLLM server",
    )
    parser.add_argument(
        "--disable-log-requests",
        action="store_true",
        help="Skip per-request logging to keep the output concise",
    )
    parser.add_argument(
        "--served-model-name",
        default=None,
        help="Override the model name announced by the OpenAI-compatible endpoint",
    )
    parser.add_argument(
        "--ssl-keyfile",
        default=None,
        help="Path to SSL key file when serving over HTTPS",
    )
    parser.add_argument(
        "--ssl-certfile",
        default=None,
        help="Path to SSL certificate when serving over HTTPS",
    )
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI entry point
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")

    try:
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.entrypoints.openai.api_server import run_server
        from vllm.entrypoints.openai.cli_args import make_arg_parser as _make_arg_parser
        from vllm.entrypoints.openai.cli_args import ServerArgs
    except Exception as exc:  # pragma: no cover - import guard
        raise ImportError(
            "vLLM is required to serve checkpoints. Install it via `pip install vllm`."
        ) from exc

    # Reuse the upstream argument transformers so that behavioural parity with the
    # ``vllm serve`` CLI is maintained. We manually stitch together the namespace to
    # avoid re-parsing all command-line options.
    parser = _make_arg_parser(argparse.ArgumentParser(add_help=False))
    cli_args = parser.parse_args([])  # empty parse to get defaults

    for key, value in vars(args).items():
        setattr(cli_args, key.replace("-", "_"), value)

    engine_args = AsyncEngineArgs.from_cli_args(cli_args)
    server_args = ServerArgs.from_cli_args(cli_args)

    if args.disable_log_requests:
        server_args.log_requests = False

    if args.served_model_name is not None:
        server_args.served_model_name = args.served_model_name

    logger.info(
        "Starting vLLM server on %s:%s with model=%s tensor_parallel=%s",
        args.host,
        args.port,
        args.model,
        args.tensor_parallel_size,
    )
    run_server(engine_args, server_args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()

