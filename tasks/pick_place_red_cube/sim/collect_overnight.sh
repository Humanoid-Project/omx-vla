#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SIM_DIR="$REPO_ROOT/tasks/pick_place_red_cube/sim"

: "${EPISODES:=200}"
: "${SEED:=0}"
: "${RAW_DIR:=$REPO_ROOT/outputs/sim_raw/pick_place_red_cube_seed$SEED}"
: "${REPO_ID:=local/omx_pick_place_red_cube_sim}"
: "${DATASET_ROOT:=}"
: "${STALL_S:=300}"
: "${MAX_FAILED_ATTEMPTS:=5}"
: "${EXPORT:=1}"
: "${ISAAC_PYTHON:=$HOME/miniconda3/envs/isaacsim/bin/python}"
: "${LEROBOT_PYTHON:=$HOME/miniconda3/envs/lerobot-omx/bin/python}"

if [ -z "${COLLECT_INHIBITED:-}" ]; then
  export COLLECT_INHIBITED=1
  exec systemd-inhibit --what=idle:sleep:handle-lid-switch --mode=block --who=omx-collect --why="sim data collection" "$0" "$@"
fi

mkdir -p "$RAW_DIR"
LOG="$RAW_DIR/run.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%F %T')] $*"; }
done_count() { find "$RAW_DIR" -mindepth 2 -maxdepth 2 -path '*/episode_*/data.npz' | wc -l; }

"$ISAAC_PYTHON" -c 'import torch,time
x = torch.zeros(1, device="cuda")
while True:
    x += 1
    torch.cuda.synchronize()
    time.sleep(5)' &
KEEPALIVE=$!
COLLECT_PID=""
cleanup() {
  [ -n "$COLLECT_PID" ] && kill "$COLLECT_PID" 2>/dev/null
  kill "$KEEPALIVE" 2>/dev/null
}
trap cleanup EXIT
trap 'say "interrupted"; exit 130' INT TERM

sleep 10
if ! kill -0 "$KEEPALIVE" 2>/dev/null; then
  say "ABORT: no usable CUDA GPU (reboot, then run again)"
  exit 1
fi

say "start: $EPISODES episodes, seed $SEED, raw $RAW_DIR, $(done_count) already done"
failed=0
while [ "$(done_count)" -lt "$EPISODES" ] && [ "$failed" -lt "$MAX_FAILED_ATTEMPTS" ]; do
  before=$(done_count)
  ATTEMPT_LOG="$RAW_DIR/collect_$(date +%H%M%S).log"
  PREVIEW=()
  [ "$before" -eq 0 ] && PREVIEW=(--preview_video "$RAW_DIR/preview.mp4" --preview_episodes 5)
  bash "$REPO_ROOT/sim/omx/run.sh" "$SIM_DIR/collect.py" --episodes "$EPISODES" --seed "$SEED" --out "$RAW_DIR" "${PREVIEW[@]}" > "$ATTEMPT_LOG" 2>&1 &
  COLLECT_PID=$!
  last=$(date +%s)
  seen=$before
  while kill -0 "$COLLECT_PID" 2>/dev/null; do
    sleep 15
    now_done=$(done_count)
    if [ "$now_done" -gt "$seen" ]; then
      grep -E '^episode [0-9]+: success' "$ATTEMPT_LOG" | tail -n "$((now_done - seen))"
      seen=$now_done
      last=$(date +%s)
    elif [ $(( $(date +%s) - last )) -gt "$STALL_S" ]; then
      say "no new episode for $STALL_S s, killing pid $COLLECT_PID"
      kill -9 "$COLLECT_PID" 2>/dev/null
      break
    fi
  done
  wait "$COLLECT_PID" 2>/dev/null
  COLLECT_PID=""
  if [ "$(done_count)" -le "$before" ]; then
    failed=$((failed + 1))
    say "attempt made no progress ($failed/$MAX_FAILED_ATTEMPTS), log $ATTEMPT_LOG"
    grep -E 'Traceback|Error:|RuntimeError' "$ATTEMPT_LOG" | grep -v carb | tail -3
  else
    failed=0
  fi
  sleep 10
done

say "collected $(done_count)/$EPISODES"
"$LEROBOT_PYTHON" -I "$SIM_DIR/report.py" --raw "$RAW_DIR" | tee "$RAW_DIR/report.txt"

if [ "$(done_count)" -lt "$EPISODES" ]; then
  say "FAILED: stopped before reaching $EPISODES; run the same command again to resume"
  exit 1
fi

if [ "$EXPORT" = "1" ]; then
  say "exporting to $REPO_ID"
  ROOT_ARG=()
  [ -n "$DATASET_ROOT" ] && ROOT_ARG=(--root "$DATASET_ROOT")
  if "$LEROBOT_PYTHON" "$SIM_DIR/export_lerobot.py" --raw "$RAW_DIR" --repo_id "$REPO_ID" "${ROOT_ARG[@]}" --overwrite > "$RAW_DIR/export.log" 2>&1; then
    grep -E '^dataset ' "$RAW_DIR/export.log"
    say "DONE"
  else
    say "FAILED: export, see $RAW_DIR/export.log"
    tail -5 "$RAW_DIR/export.log"
    exit 1
  fi
fi
