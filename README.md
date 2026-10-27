# lerobot_isaac

[![LeRobot](https://img.shields.io/badge/LeRobot-HuggingFace-FFD21E?logo=github&logoColor=black)](https://github.com/huggingface/lerobot)
[![LeIsaac](https://img.shields.io/badge/LeIsaac-LightwheelAI-4A90D9?logo=github&logoColor=white)](https://github.com/LightwheelAI/leisaac)

Monorepo combining [LeRobot](https://github.com/huggingface/lerobot) and [LeIsaac](https://github.com/LightwheelAI/leisaac) for training and evaluating robot manipulation policies in Isaac Sim and on the real ROBOTIS OMX-AI arm, with the goal of sim-to-real transfer.


## Projects

### \#1 — Training and Testing SmolVLA in Isaac Sim

[![YouTube](https://img.shields.io/badge/YouTube-FF0000?logo=youtube&logoColor=white)](https://youtu.be/IMJx391fn_4?si=pU5dBo8fUevKBreY)

**Training and Testing SmolVLA in Isaac Sim with LeRobot and LeIsaac**

- Demonstration data collected in Isaac Sim using LeRobot and LeIsaac
- SmolVLA policy trained on the collected data
- Trained policy tested back in the LeIsaac Isaac Sim environment


### \#2 — OMX Real-Robot Imitation Learning

Teleoperation data collection, ACT / SmolVLA training and rollout on the ROBOTIS OMX-AI arm. See [`real/omx/`](real/omx/README.md).


## Repository Layout

```
lerobot/     LeRobot (upstream, unmodified) — datasets, training, real-robot drivers
leisaac/     LeIsaac (upstream, .gitignore only changed) — Isaac Sim tasks and teleoperation
real/omx/    Scripts and settings for the real OMX arm
```


## License

The code in this repository outside `lerobot/` and `leisaac/` (`real/`, root files) is licensed under the Apache License 2.0 — see [`LICENSE`](LICENSE).

`lerobot/` and `leisaac/` are third-party code under the Apache License 2.0, each with its own `LICENSE` file.
Versions, copyright holders, files under other licenses (MIT, BSD-3-Clause) and what is left out of this repository are listed in [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
