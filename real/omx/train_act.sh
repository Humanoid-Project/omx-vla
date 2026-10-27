#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

RUN_NAME="${RUN_NAME:-act_${DATASET_NAME}}"

lerobot-train \
  --dataset.repo_id="$DATASET" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir="$TRAIN_DIR/$RUN_NAME" \
  --job_name="$RUN_NAME" \
  --batch_size=8 \
  --steps=20000 \
  --save_freq=5000 \
  --wandb.enable=false \
  "$@"
