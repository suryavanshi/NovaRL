"""Minimal Hugging Face fine-tuning entry point with FSDP/DeepSpeed presets."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from core.distributed import DeepSpeedZeROConfig, get_default_fsdp_preset

logger = logging.getLogger(__name__)

try:  # pragma: no cover - optional heavy deps
    from datasets import Dataset, load_dataset
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )
except Exception as exc:  # pragma: no cover - graceful error for optional deps
    raise ImportError(
        "The fine-tuning entry point requires the `datasets` and `transformers` packages."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-name", default="meta-llama/Llama-2-7b-hf", help="Model to fine-tune"
    )
    parser.add_argument(
        "--output-dir", default="runs/finetune", help="Directory for checkpoints and logs"
    )
    parser.add_argument(
        "--dataset-name",
        default=None,
        help="HF dataset identifier. If omitted a synthetic corpus is used",
    )
    parser.add_argument("--dataset-config-name", default=None, help="Optional dataset config")
    parser.add_argument("--dataset-split", default="train", help="Dataset split to consume")
    parser.add_argument("--text-field", default="text", help="Dataset column containing raw text")
    parser.add_argument(
        "--synthetic-samples",
        type=int,
        default=1024,
        help="Samples to synthesize when dataset-name is absent",
    )
    parser.add_argument(
        "--synthetic-text",
        default="NovaRL synthetic sample.",
        help="Template text for synthetic dataset",
    )
    parser.add_argument(
        "--max-length", type=int, default=1024, help="Sequence length used during tokenization"
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--strategy", choices=["fsdp", "deepspeed"], default="fsdp")
    parser.add_argument("--zero-stage", type=int, choices=[2, 3], default=3)
    parser.add_argument(
        "--deepspeed-config", default=None, help="Optional path to a custom DeepSpeed config"
    )
    parser.add_argument(
        "--optimizer-offload", action="store_true", help="Offload optimizer state to CPU"
    )
    parser.add_argument(
        "--param-offload", action="store_true", help="Offload parameters to CPU (ZeRO-3 only)"
    )
    parser.add_argument("--no-optimizer-sharding", dest="optimizer_sharding", action="store_false")
    parser.set_defaults(optimizer_sharding=True)
    parser.add_argument(
        "--cpu-offload", action="store_true", help="Enable FSDP CPU offload for activations/params"
    )
    parser.add_argument(
        "--fsdp-mixed-precision",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Mixed precision mode for FSDP",
    )
    parser.add_argument(
        "--no-gradient-checkpointing", dest="gradient_checkpointing", action="store_false"
    )
    parser.set_defaults(gradient_checkpointing=True)
    parser.add_argument(
        "--bf16", dest="bf16", action="store_true", help="Enable bf16 in training arguments"
    )
    parser.add_argument("--no-bf16", dest="bf16", action="store_false")
    parser.set_defaults(bf16=True)
    parser.add_argument(
        "--fp16", dest="fp16", action="store_true", help="Enable fp16 in training arguments"
    )
    parser.add_argument("--no-fp16", dest="fp16", action="store_false")
    parser.set_defaults(fp16=False)
    parser.add_argument("--save-strategy", default="no", help="Trainer save strategy")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_dataset(args: argparse.Namespace) -> Dataset:
    if args.dataset_name:
        ds = load_dataset(args.dataset_name, args.dataset_config_name, split=args.dataset_split)
    else:
        ds = Dataset.from_dict(
            {"text": [args.synthetic_text for _ in range(args.synthetic_samples)]}
        )
    if args.max_train_samples:
        ds = ds.select(range(min(args.max_train_samples, len(ds))))
    return ds


def _prepare_dataset(
    dataset: Dataset, tokenizer: AutoTokenizer, args: argparse.Namespace
) -> Dataset:
    column = args.text_field
    if column not in dataset.column_names:
        raise ValueError(f"Column {column!r} not found in dataset columns {dataset.column_names}")

    def tokenize_fn(example: Dict[str, Any]) -> Dict[str, Any]:
        text = example[column]
        if isinstance(text, list):
            text = "\n\n".join(str(t) for t in text)
        return tokenizer(text, truncation=True, max_length=args.max_length)

    return dataset.map(tokenize_fn, remove_columns=dataset.column_names)


def build_trainer(args: argparse.Namespace) -> Trainer:
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    raw_dataset = _load_dataset(args)
    logger.info("Loaded dataset with %s samples", len(raw_dataset))
    dataset = _prepare_dataset(raw_dataset, tokenizer, args)
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "config"):
            model.config.use_cache = False

    fsdp_config = None
    fsdp_argument = None
    deepspeed_config_path: Optional[str] = None

    if args.strategy == "fsdp":
        preset = get_default_fsdp_preset().with_overrides(
            gradient_checkpointing=args.gradient_checkpointing,
            mixed_precision=args.fsdp_mixed_precision,
            cpu_offload=args.cpu_offload,
        )
        fsdp_config = preset.to_hf_dict()
        fsdp_argument = preset.fsdp_argument
    else:
        if args.deepspeed_config:
            deepspeed_config_path = args.deepspeed_config
        else:
            zero = (
                DeepSpeedZeROConfig.stage2_default
                if args.zero_stage == 2
                else DeepSpeedZeROConfig.stage3_default
            )(
                bf16=args.bf16,
                fp16=args.fp16,
                offload_optimizer=args.optimizer_offload,
                offload_parameters=args.param_offload,
                optimizer_sharding=args.optimizer_sharding,
            )
            output_dir = Path(args.output_dir)
            deepspeed_config_path = str(
                zero.dump_json(output_dir / f"zero{args.zero_stage}_config.json")
            )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        seed=args.seed,
        report_to=[],
        deepspeed=deepspeed_config_path,
        fsdp=fsdp_argument,
        fsdp_config=fsdp_config,
    )

    return Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    logger.info("Starting fine-tuning with args: %s", args)
    trainer = build_trainer(args)
    trainer.train()


if __name__ == "__main__":
    main()
