#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

lerobot-record \
  --robot.type=omx_follower \
  --robot.port="$ROBOT_PORT" \
  --robot.id="$ROBOT_ID" \
  --robot.cameras="$CAMERAS" \
  --teleop.type=omx_leader \
  --teleop.port="$TELEOP_PORT" \
  --teleop.id="$TELEOP_ID" \
  --dataset.repo_id="$DATASET" \
  --dataset.single_task="$TASK" \
  --dataset.fps="$FPS" \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=30 \
  --dataset.reset_time_s=15 \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --display_data=true \
  "$@"
