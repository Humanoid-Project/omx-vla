#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

if [ $# -lt 1 ]; then
  echo "usage: $0 <policy_path> [extra lerobot-rollout args]" >&2
  echo "  e.g. $0 $TRAIN_DIR/act_${DATASET_NAME}/checkpoints/last/pretrained_model" >&2
  exit 1
fi
POLICY_PATH="$1"
shift

RENAME_ARGS=()
if [ -f "$POLICY_PATH/config.json" ] && grep -qE '"type": *"smolvla"' "$POLICY_PATH/config.json"; then
  RENAME_ARGS=(--rename_map="$SMOLVLA_RENAME_MAP")
fi

lerobot-rollout \
  --strategy.type=base \
  --robot.type=omx_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS" \
  --policy.path="$POLICY_PATH" \
  --task="$TASK" \
  --fps="$FPS" \
  --display_data=true \
  "${RENAME_ARGS[@]}" \
  "$@"
