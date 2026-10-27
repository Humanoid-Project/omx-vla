#!/usr/bin/env bash

OMX_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$OMX_DIR/../.." && pwd)"

if [ -f "$OMX_DIR/env.local.sh" ]; then
  source "$OMX_DIR/env.local.sh"
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
if [ -z "${CAMERAS:-}" ]; then
  CAMERAS="{ wrist: {type: opencv, index_or_path: $WRIST_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS}, top: {type: opencv, index_or_path: $TOP_CAM, width: $CAM_WIDTH, height: $CAM_HEIGHT, fps: $FPS} }"
fi

: "${HF_USER:=}"
: "${DATASET_NAME:=omx_pickplace}"
: "${TASK:=Pick the cube and place it in the blue square}"
: "${PUSH_TO_HUB:=false}"

DATASET="${DATASET:-${HF_USER:-local}/$DATASET_NAME}"
TRAIN_DIR="$REPO_ROOT/outputs/train"

if [ -z "${SMOLVLA_RENAME_MAP:-}" ]; then
  SMOLVLA_RENAME_MAP='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
fi
