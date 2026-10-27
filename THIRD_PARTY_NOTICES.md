# Third-Party Notices

This repository redistributes source code from the following projects. Each keeps its own license file and the
copyright and license headers in its source files.

## LeRobot

- Source: https://github.com/huggingface/lerobot
- Version: commit `79c68214070e392f04800ed092dd89e0b761f44e` (2026-06-10, package version 0.5.2)
- Location: `lerobot/`
- License: Apache License 2.0 — `lerobot/LICENSE`
- Copyright: The Hugging Face team and contributors
- Included files under other licenses (license text or reference kept in each file):
  - `lerobot/src/lerobot/policies/groot/eagle2_hg_model/modeling_eagle2_5_vl.py` and
    `image_processing_eagle2_5_vl_fast.py` — MIT License, Copyright (c) 2025 NVIDIA
  - `lerobot/src/lerobot/policies/vqbet/vqbet_utils.py` — contains code from vector-quantize-pytorch and nanoGPT,
    MIT License
  - `lerobot/src/lerobot/motors/damiao/damiao.py` — portions derived from DM_Control_Python by cmjang, MIT License
- Not redistributed here: `docs/`, `tests/`, `media/`, `docker/`, `.github/`, `scripts/`, and `examples/` except
  `examples/omx/`.
- Modifications: none.

## LeIsaac

- Source: https://github.com/LightwheelAI/leisaac
- Version: commit `24d3bcd3f1e4585740fc79921782c41617237812` (2026-04-15, version 0.4.0)
- Location: `leisaac/`
- License: Apache License 2.0 — `leisaac/LICENSE`
- Copyright: Lightwheel and the LeIsaac Project Developers
- Citation: `leisaac/CITATION.cff`
- Included files under other licenses (license header kept in each file):
  - Files derived from Isaac Lab — BSD-3-Clause, Copyright (c) 2022-2025 The Isaac Lab Project Developers
    (`leisaac/source/leisaac/leisaac/__init__.py`, `leisaac/source/leisaac/leisaac/tasks/__init__.py`,
    `leisaac/source/leisaac/leisaac/devices/device_base.py`, `leisaac/scripts/environments/list_envs.py`,
    `leisaac/scripts/environments/teleoperation/teleop_se3_agent.py`, `leisaac/.vscode/tools/setup_vscode.py`)
- Not redistributed here: `docs/`, `.github/`, `scripts/tutorials/`, and robot/scene assets under
  `assets/` (downloaded separately; they carry their own licenses).
- Modifications: none.

## Modifying these files

Apache License 2.0, section 4(b): a modified file must carry a prominent notice stating that it was changed. Add that
notice to any file under `lerobot/` or `leisaac/` that is edited, and update the "Modifications" lines above.
