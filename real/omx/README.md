# OMX (real robot)

Teleoperation, data recording, training and rollout on the ROBOTIS OMX-AI (OMX-F follower + OMX-L leader) with LeRobot.

| File | Purpose |
|---|---|
| `env.sh` | Shared settings (ports, cameras, dataset, task). Sourced by every script. |
| `env.local.sh.example` | Copy to `env.local.sh` (git-ignored) to override `env.sh` for this machine. |
| `teleoperate.sh` | Leader → follower teleoperation with camera view. |
| `record.sh` | Record demonstration episodes into a LeRobotDataset. |
| `train_act.sh` | Train an ACT policy. |
| `train_smolvla.sh` | Fine-tune SmolVLA from `lerobot/smolvla_base`. |
| `rollout.sh` | Run a trained policy on the robot. |
| `99-omx.rules.example` | udev rule template for stable `/dev/omx_follower` and `/dev/omx_leader` names. |

## Setup

```bash
conda create -n lerobot-omx python=3.12
conda activate lerobot-omx
cd lerobot
pip install -e ".[core_scripts,training,dynamixel,smolvla]" \
  --extra-index-url https://download.pytorch.org/whl/cu128
```

Pick the PyTorch CUDA build that matches the NVIDIA driver (`nvidia-smi`, "CUDA Version").

## Find ports and cameras

```bash
lerobot-find-port
lerobot-find-cameras opencv
ls -l /dev/serial/by-id/
```

```bash
cp real/omx/env.local.sh.example real/omx/env.local.sh
```

Edit `env.local.sh` with the ports, camera indices and `HF_USER`.

Optional stable port names:

```bash
cp real/omx/99-omx.rules.example real/omx/99-omx.rules
sudo cp real/omx/99-omx.rules /etc/udev/rules.d/99-omx.rules
sudo udevadm control --reload-rules && sudo udevadm trigger
```

Replace `FOLLOWER_SERIAL` / `LEADER_SERIAL` with the serials from `/dev/serial/by-id/` first.

## Run

The robot moves during `teleoperate.sh`, `record.sh` and `rollout.sh`.

Task settings (task sentence, dataset name, episode count and length) come from a `task.env` file under `tasks/`,
selected with `TASK_ENV`. `record.sh` and `rollout.sh` stop if no task sentence is set.

```bash
export TASK_ENV=tasks/pick_place_red_cube/task.env
./real/omx/teleoperate.sh
./real/omx/record.sh
```

`lerobot-record` appends the start time to the dataset name, e.g. `local/omx_pick_place_red_cube_20261108_234512`
under `~/.cache/huggingface/lerobot/`. Training and resuming take that full name as `DATASET`; without it the scripts
stop and list the recorded datasets.

```bash
ls ~/.cache/huggingface/lerobot/local/
DATASET=local/omx_pick_place_red_cube_20261108_234512 ./real/omx/train_act.sh
DATASET=local/omx_pick_place_red_cube_20261108_234512 ./real/omx/train_smolvla.sh
./real/omx/rollout.sh outputs/train/act_omx_pick_place_red_cube/checkpoints/last/pretrained_model
```

Every script forwards extra arguments to the underlying `lerobot-*` command, e.g.

```bash
DATASET=local/omx_pick_place_red_cube_20261108_234512 ./real/omx/record.sh --resume=true --dataset.num_episodes=10
DATASET=local/omx_pick_place_red_cube_20261108_234512 ./real/omx/train_smolvla.sh --batch_size=4
```

With `--resume=true`, `--dataset.num_episodes` is the number of episodes to add, not the total.

Cameras use MJPG (`CAM_FOURCC`): two uncompressed 640×480 streams on one USB 2.0 hub drop frames.
`record.sh` encodes video while recording (`--dataset.streaming_encoding=true --dataset.encoder_threads=2`).

Keep `FPS`, camera names, resolution and `TASK` identical between recording and rollout.

To add a task, create `tasks/<name>/task.env` with `TASK`, `DATASET_NAME` and, if needed, `NUM_EPISODES`,
`EPISODE_TIME_S` and `RESET_TIME_S`.
