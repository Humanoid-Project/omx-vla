#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${TASK:?TASK is empty, run with TASK_ENV=tasks/<name>/task.env}"

case " $* " in
  *" --resume=true "*) require_dataset ;;
  *) DATASET="${DATASET:-$DATASET_BASE}" ;;
esac

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
  --dataset.num_episodes="$NUM_EPISODES" \
  --dataset.episode_time_s="$EPISODE_TIME_S" \
  --dataset.reset_time_s="$RESET_TIME_S" \
  --dataset.push_to_hub="$PUSH_TO_HUB" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --display_data=true \
  "$@"
