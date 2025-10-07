#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "$SCRIPT_DIR/../.." && pwd)

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

cd "$REPO_ROOT"
python -m examples.ppo_vlm_vqa "$@"

