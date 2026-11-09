#!/usr/bin/env bash

OMX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$OMX_DIR/../.." && pwd)"

if [ -f "$OMX_DIR/env.local.sh" ]; then
  source "$OMX_DIR/env.local.sh"
fi

if [ -n "${TASK_ENV:-}" ]; then
  if [ -f "$TASK_ENV" ]; then
    source "$TASK_ENV"
  elif [ -f "$REPO_ROOT/$TASK_ENV" ]; then
    source "$REPO_ROOT/$TASK_ENV"
  else
    echo "TASK_ENV not found: $TASK_ENV" >&2
    exit 1
  fi
fi

: "${ROBOT_PORT:=/dev/ttyACM0}"
: "${TELEOP_PORT:=/dev/ttyACM1}"
: "${ROBOT_ID:=omx_follower}"
: "${TELEOP_ID:=omx_leader}"

: "${FPS:=30}"
: "${CAM_WIDTH:=640}"
: "${CAM_HEIGHT:=480}"
: "${WRIST_CAM:=0}"
: "${TOP_CAM:=2}"
: "${CAM_FOURCC:=MJPG}"
if [ -z "${CAMERAS:-}" ]; then
  CAMERAS="{ wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, fourcc: $CAM_FOURCC}, top: {type: opencv, index_or_path: $TOP_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS, fourcc: $CAM_FOURCC} }"
fi

: "${HF_USER:=}"
: "${DATASET_NAME:=omx_pickplace}"
: "${TASK:=}"
: "${NUM_EPISODES:=50}"
: "${EPISODE_TIME_S:=30}"
: "${RESET_TIME_S:=15}"
: "${PUSH_TO_HUB:=false}"

DATASET_BASE="${HF_USER:-local}/$DATASET_NAME"
LEROBOT_HOME="${HF_LEROBOT_HOME:-${HF_HOME:-$HOME/.cache/huggingface}/lerobot}"
TRAIN_DIR="$REPO_ROOT/outputs/train"

if [ -z "${SMOLVLA_RENAME_MAP:-}" ]; then
  SMOLVLA_RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
fi

require_dataset() {
  if [ -z "${DATASET:-}" ]; then
    echo "DATASET is empty. lerobot-record saves each recording as ${DATASET_BASE}_YYYYMMDD_HHMMSS; pick one:" >&2
    ls -1d "$LEROBOT_HOME/$DATASET_BASE"_[0-9]*_[0-9]* 2>/dev/null | sed "s|^$LEROBOT_HOME/|  DATASET=|" >&2
    exit 1
  fi
}
