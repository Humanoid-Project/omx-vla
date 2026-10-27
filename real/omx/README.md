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

```bash
./real/omx/teleoperate.sh
./real/omx/record.sh
./real/omx/train_act.sh
./real/omx/train_smolvla.sh
./real/omx/rollout.sh outputs/train/act_omx_pickplace/checkpoints/last/pretrained_model
```

Every script forwards extra arguments to the underlying `lerobot-*` command, e.g.

```bash
./real/omx/record.sh --dataset.num_episodes=10 --resume=true
./real/omx/train_smolvla.sh --batch_size=4
```

Keep `FPS`, camera names, resolution and `TASK` identical between recording and rollout.
