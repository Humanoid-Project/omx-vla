import argparse
import json
import time
from multiprocessing.connection import Listener

import numpy as np
import torch

from lerobot.common.control_utils import predict_action
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies import get_policy_class, make_pre_post_processors

SMOLVLA_RENAME_MAP = {
    "observation.images.top": "observation.images.camera1",
    "observation.images.wrist": "observation.images.camera2",
}


def parse_args():
    p = argparse.ArgumentParser(description="Serve a trained LeRobot policy to the Isaac Lab sim over a local socket.")
    p.add_argument("--policy_path", required=True, help="Folder with config.json and model.safetensors.")
    p.add_argument("--port", type=int, default=5555)
    p.add_argument("--authkey", default="omx-vla")
    p.add_argument("--device", default="cuda")
    p.add_argument("--rename_map", default=None, help="JSON rename map; SmolVLA gets the top/wrist -> camera1/2 map by default.")
    p.add_argument("--temporal_ensemble_coeff", type=float, default=None, help="ACT only: enable temporal ensembling (forces n_action_steps=1).")
    p.add_argument("--n_action_steps", type=int, default=None, help="Override how many actions of each chunk are executed before re-planning.")
    return p.parse_args()


def load(args):
    cfg = PreTrainedConfig.from_pretrained(args.policy_path)
    cfg.pretrained_path = args.policy_path
    cfg.device = args.device
    if args.temporal_ensemble_coeff is not None:
        cfg.temporal_ensemble_coeff = args.temporal_ensemble_coeff
        cfg.n_action_steps = 1
    elif args.n_action_steps is not None:
        cfg.n_action_steps = args.n_action_steps
    policy = get_policy_class(cfg.type).from_pretrained(args.policy_path, config=cfg)
    policy.to(args.device).eval()
    rename_map = json.loads(args.rename_map) if args.rename_map else (SMOLVLA_RENAME_MAP if cfg.type == "smolvla" else {})
    pre, post = make_pre_post_processors(
        policy_cfg=cfg,
        pretrained_path=args.policy_path,
        preprocessor_overrides={
            "device_processor": {"device": args.device},
            "rename_observations_processor": {"rename_map": rename_map},
        },
    )
    return cfg, policy, pre, post


def main():
    args = parse_args()
    cfg, policy, pre, post = load(args)
    device = torch.device(args.device)
    te = getattr(cfg, "temporal_ensemble_coeff", None)
    print(f"policy {cfg.type} from {args.policy_path}, inputs {list(cfg.input_features)}, "
          f"chunk_size {getattr(cfg, 'chunk_size', None)}, n_action_steps {cfg.n_action_steps}, temporal_ensemble_coeff {te}", flush=True)
    with Listener(("127.0.0.1", args.port), authkey=args.authkey.encode()) as listener:
        print(f"listening on 127.0.0.1:{args.port}", flush=True)
        while True:
            with listener.accept() as conn:
                print("client connected", flush=True)
                try:
                    while True:
                        msg = conn.recv()
                        if msg["cmd"] == "reset":
                            policy.reset()
                            pre.reset()
                            post.reset()
                            conn.send({"ok": True, "type": cfg.type, "policy_path": args.policy_path, "n_action_steps": cfg.n_action_steps,
                                       "temporal_ensemble_coeff": getattr(cfg, "temporal_ensemble_coeff", None)})
                        elif msg["cmd"] == "act":
                            t0 = time.perf_counter()
                            action = predict_action(
                                msg["obs"], policy, device, pre, post, use_amp=False, task=msg.get("task"), robot_type="omx_follower"
                            )
                            a = action.squeeze(0).float().cpu().tolist()
                            conn.send({"action": a, "ms": (time.perf_counter() - t0) * 1000})
                        elif msg["cmd"] == "close":
                            conn.send({"ok": True})
                            break
                except EOFError:
                    pass
                print("client disconnected", flush=True)


if __name__ == "__main__":
    main()
