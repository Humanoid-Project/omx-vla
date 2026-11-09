# Pick and Place Red Cube

Pick up a cube wrapped in red electrical tape and place it inside a square outlined with black electrical tape, on the
real ROBOTIS OMX-AI arm.

| Item | Value |
|---|---|
| Task sentence | `Pick up the red cube and place it inside the black square.` |
| Dataset | `<HF_USER or local>/omx_pick_place_red_cube_<YYYYMMDD_HHMMSS>` (one per `record.sh` run) |
| Robot | OMX-F follower, OMX-L leader |
| Cameras | `top`, `wrist` (640×480, 30 fps) |
| Settings | [`task.env`](task.env) |

## Run

```bash
export TASK_ENV=tasks/pick_place_red_cube/task.env
./real/omx/teleoperate.sh
./real/omx/record.sh
ls ~/.cache/huggingface/lerobot/local/
DATASET=local/omx_pick_place_red_cube_<YYYYMMDD_HHMMSS> ./real/omx/train_act.sh
DATASET=local/omx_pick_place_red_cube_<YYYYMMDD_HHMMSS> ./real/omx/train_smolvla.sh
./real/omx/rollout.sh outputs/train/act_omx_pick_place_red_cube/checkpoints/last/pretrained_model
```

The task sentence, camera names, resolution and fps must stay identical between recording and rollout.

## Simulation

Isaac Lab environment `OMX-PickPlaceRedCube-v0`: blue OMX-F, white table, yellow wall behind the robot, 3 cm red cube,
7 cm filled black square, `top` and `wrist` cameras. It uses the `isaacsim` conda env for collection and `lerobot-omx`
for export; nothing is installed into either.

Once, after a fresh clone:

```bash
bash sim/omx/run.sh sim/omx/convert_urdf.py --force
```

Collect randomized scripted episodes and export them as a LeRobot dataset:

```bash
EPISODES=200 tasks/pick_place_red_cube/sim/collect_overnight.sh
```

| Variable | Default | Meaning |
|---|---|---|
| `EPISODES` | `200` | Total episodes, including ones that fail and are left out of the export |
| `SEED` | `0` | Each episode's randomization depends only on `SEED` and its index |
| `RAW_DIR` | `outputs/sim_raw/pick_place_red_cube_seed$SEED` | Frames, joint data, `run.log`, `report.txt`, `preview.mp4` |
| `REPO_ID` | `local/omx_pick_place_red_cube_sim` | Dataset id, stored under `$HF_LEROBOT_HOME` |
| `DATASET_ROOT` | empty | Store the dataset in this folder instead |
| `EXPORT` | `1` | `0` skips the export |
| `STALL_S` | `300` | Restart collection when no episode finishes for this long |

The script blocks sleep and lid-close suspend while it runs, restarts collection after a crash or hang, and resumes from
the last finished episode when run again with the same settings. The export replaces the existing dataset of that id.

Train on the result:

```bash
export TASK_ENV=tasks/pick_place_red_cube/task.env
DATASET=local/omx_pick_place_red_cube_sim RUN_NAME=smolvla_sim ./real/omx/train_smolvla.sh
DATASET=local/omx_pick_place_red_cube_sim RUN_NAME=act_sim ./real/omx/train_act.sh
```

Exported units follow the default `omx_follower` recording: arm joints in `RANGE_M100_100` (`rad × 100/π`), gripper
`60.3 − (0.7 − rad) × 100/(2π)`, larger is more open. The zero, sign and scale were checked against the public dataset
`maximellerbach/omx_pickandplace`, not against this lab's arm; measure the open gripper value on the real arm and pass it
as `--gripper_open_units` to `export_lerobot.py` before mixing sim and real data.
