#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

GPUS_PER_NODE=${GPUS_PER_NODE:-8}
MODEL_NAME=${MODEL_NAME:-meta-llama/Llama-2-7b-hf}
OUTPUT_DIR=${OUTPUT_DIR:-runs/deepspeed_zero}
DATASET_NAME=${DATASET_NAME:-tatsu-lab/alpaca}
DATASET_CONFIG=${DATASET_CONFIG:-}
DATASET_SPLIT=${DATASET_SPLIT:-train}
TEXT_FIELD=${TEXT_FIELD:-text}
MAX_LENGTH=${MAX_LENGTH:-1024}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
LOGGING_STEPS=${LOGGING_STEPS:-10}
ZERO_STAGE=${ZERO_STAGE:-3}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-1}
BF16=${BF16:-1}
FP16=${FP16:-0}
OPTIMIZER_OFFLOAD=${OPTIMIZER_OFFLOAD:-0}
PARAM_OFFLOAD=${PARAM_OFFLOAD:-0}
OPTIMIZER_SHARDING=${OPTIMIZER_SHARDING:-1}
USE_PRESET=${USE_PRESET:-1}
CUSTOM_DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-}

GC_FLAG=""
if [[ "${GRADIENT_CHECKPOINTING}" == "0" ]]; then
  GC_FLAG="--no-gradient-checkpointing"
fi

BF16_FLAG=""
if [[ "${BF16}" == "0" ]]; then
  BF16_FLAG="--no-bf16"
fi

FP16_FLAG=""
if [[ "${FP16}" == "1" ]]; then
  FP16_FLAG="--fp16"
else
  FP16_FLAG="--no-fp16"
fi

OPTIMIZER_OFFLOAD_FLAG=""
if [[ "${OPTIMIZER_OFFLOAD}" == "1" ]]; then
  OPTIMIZER_OFFLOAD_FLAG="--optimizer-offload"
fi

PARAM_OFFLOAD_FLAG=""
if [[ "${PARAM_OFFLOAD}" == "1" && "${ZERO_STAGE}" == "3" ]]; then
  PARAM_OFFLOAD_FLAG="--param-offload"
fi

SHARDING_FLAG=""
if [[ "${OPTIMIZER_SHARDING}" == "0" ]]; then
  SHARDING_FLAG="--no-optimizer-sharding"
fi

CONFIG_FLAG=()
if [[ -n "${CUSTOM_DEEPSPEED_CONFIG}" ]]; then
  CONFIG_FLAG=(--deepspeed-config "${CUSTOM_DEEPSPEED_CONFIG}")
else
  PRESET_PATH="${SCRIPT_DIR}/../configs/deepspeed/zero${ZERO_STAGE}.json"
  if [[ "${OPTIMIZER_OFFLOAD}" == "0" && "${PARAM_OFFLOAD}" == "0" && "${OPTIMIZER_SHARDING}" == "1" && "${USE_PRESET}" != "0" ]]; then
    CONFIG_FLAG=(--deepspeed-config "${PRESET_PATH}")
  fi
fi

CMD=(
  torchrun --standalone --nproc_per_node="${GPUS_PER_NODE}" scripts/finetune_hf.py
  --model-name "${MODEL_NAME}"
  --output-dir "${OUTPUT_DIR}"
  --dataset-name "${DATASET_NAME}"
  --dataset-split "${DATASET_SPLIT}"
  --text-field "${TEXT_FIELD}"
  --max-length "${MAX_LENGTH}"
  --per-device-train-batch-size "${PER_DEVICE_BATCH_SIZE}"
  --gradient-accumulation-steps "${GRADIENT_ACCUMULATION_STEPS}"
  --learning-rate "${LEARNING_RATE}"
  --warmup-ratio "${WARMUP_RATIO}"
  --num-train-epochs "${NUM_TRAIN_EPOCHS}"
  --logging-steps "${LOGGING_STEPS}"
  --strategy deepspeed
  --zero-stage "${ZERO_STAGE}"
)

if [[ -n "${DATASET_CONFIG}" ]]; then
  CMD+=(--dataset-config-name "${DATASET_CONFIG}")
fi

if [[ ${#CONFIG_FLAG[@]} -gt 0 ]]; then
  CMD+=(${CONFIG_FLAG[@]})
fi

if [[ -n "${GC_FLAG}" ]]; then
  CMD+=(${GC_FLAG})
fi
if [[ -n "${BF16_FLAG}" ]]; then
  CMD+=(${BF16_FLAG})
fi
if [[ -n "${FP16_FLAG}" ]]; then
  CMD+=(${FP16_FLAG})
fi
if [[ -n "${OPTIMIZER_OFFLOAD_FLAG}" ]]; then
  CMD+=(${OPTIMIZER_OFFLOAD_FLAG})
fi
if [[ -n "${PARAM_OFFLOAD_FLAG}" ]]; then
  CMD+=(${PARAM_OFFLOAD_FLAG})
fi
if [[ -n "${SHARDING_FLAG}" ]]; then
  CMD+=(${SHARDING_FLAG})
fi

"${CMD[@]}"
