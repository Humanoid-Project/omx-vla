#!/usr/bin/env python
import argparse
import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description="Check a recorded LeRobot v3 dataset before uploading it.")
    p.add_argument("--root", required=True, help="Local dataset folder (contains meta/, data/, videos/).")
    p.add_argument("--repo-id", default=None, help="Repo id used to load the dataset with LeRobotDataset.")
    p.add_argument("--expected-episodes", type=int, default=None)
    p.add_argument("--max-tail-idle-s", type=float, default=3.0)
    p.add_argument("--max-head-idle-s", type=float, default=3.0)
    p.add_argument("--max-start-dev", type=float, default=10.0)
    p.add_argument("--max-jump", type=float, default=10.0)
    p.add_argument("--hold-range", type=float, nargs=2, default=(50.5, 57.0))
    p.add_argument("--min-hold-s", type=float, default=0.5)
    p.add_argument("--skip-video", action="store_true")
    p.add_argument("--skip-load", action="store_true")
    return p.parse_args()


def read_parquets(folder: Path) -> pd.DataFrame:
    files = sorted(folder.glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files under {folder}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def count_and_check_video(path: Path) -> tuple[int, str | None]:
    try:
        with av.open(str(path)) as c:
            n = 0
            for _ in c.decode(c.streams.video[0]):
                n += 1
        return n, None
    except Exception as e:
        return -1, f"{type(e).__name__}: {e}"


def main():
    a = parse_args()
    root = Path(a.root).expanduser()
    errors, warns = [], []

    info = json.loads((root / "meta/info.json").read_text())
    fps = info["fps"]
    df = read_parquets(root / "data")
    ep = read_parquets(root / "meta/episodes").sort_values("episode_index").reset_index(drop=True)
    tasks = pd.read_parquet(root / "meta/tasks.parquet")
    video_keys = [k for k, v in info["features"].items() if v["dtype"] == "video"]
    names = info["features"]["observation.state"]["names"]

    print(f"dataset   {root}")
    print(f"episodes  {info['total_episodes']}   frames {info['total_frames']}   fps {fps}   "
          f"duration {info['total_frames'] / fps / 60:.1f} min")
    print(f"tasks     {list(tasks.index) if 'task' not in tasks.columns else list(tasks['task'])}")
    print(f"cameras   {video_keys}")

    n_ep = len(ep)
    if a.expected_episodes is not None and n_ep != a.expected_episodes:
        errors.append(f"episode count {n_ep} != expected {a.expected_episodes}")
    if info["total_episodes"] != n_ep:
        errors.append(f"info.json total_episodes {info['total_episodes']} != episodes meta rows {n_ep}")
    if info["total_frames"] != len(df):
        errors.append(f"info.json total_frames {info['total_frames']} != data rows {len(df)}")
    if (ep["episode_index"].values != np.arange(n_ep)).any():
        errors.append("episode_index in meta is not 0..N-1")
    if (df["index"].values != np.arange(len(df))).any():
        errors.append("global index is not contiguous 0..N-1")
    if len(tasks) != 1:
        warns.append(f"{len(tasks)} task sentences in one dataset")

    st = np.stack(df["observation.state"].values)
    ac = np.stack(df["action"].values)
    if not (np.isfinite(st).all() and np.isfinite(ac).all()):
        errors.append("NaN or inf in observation.state / action")

    lens = ep["length"].values
    cum = np.concatenate([[0], np.cumsum(lens)])
    for i, r in ep.iterrows():
        a0, b0 = int(r["dataset_from_index"]), int(r["dataset_to_index"])
        sub = df.iloc[a0:b0]
        if a0 != cum[i] or b0 - a0 != r["length"]:
            errors.append(f"ep {i}: from/to index {a0}-{b0} does not match length {r['length']}")
            continue
        if not (sub["episode_index"] == i).all():
            errors.append(f"ep {i}: rows carry another episode_index")
        if (sub["frame_index"].values != np.arange(b0 - a0)).any():
            errors.append(f"ep {i}: frame_index not 0..len-1")
        if np.abs(sub["timestamp"].values - np.arange(b0 - a0) / fps).max() > 1e-3:
            errors.append(f"ep {i}: timestamps are not frame_index / fps")
        for k in video_keys:
            f0, t0 = r[f"videos/{k}/from_timestamp"], r[f"videos/{k}/to_timestamp"]
            if abs((t0 - f0) * fps - (b0 - a0)) > 0.6:
                errors.append(f"ep {i}: {k} video span {(t0 - f0) * fps:.1f} frames != {b0 - a0}")

    if not a.skip_video:
        for k in video_keys:
            groups = ep.groupby([f"videos/{k}/chunk_index", f"videos/{k}/file_index"])
            for (ci, fi), g in groups:
                path = root / info["video_path"].format(video_key=k, chunk_index=ci, file_index=fi)
                n, err = count_and_check_video(path)
                need = int(round(g[f"videos/{k}/to_timestamp"].max() * fps))
                if err:
                    errors.append(f"{path.name} ({k}): cannot decode: {err}")
                elif abs(n - need) > 1:
                    errors.append(f"{k} chunk {ci} file {fi}: {n} decoded frames, episodes need {need}")
                else:
                    print(f"video     {k} chunk-{ci:03d}/file-{fi:03d}: {n} frames OK")

    g_idx = names.index("gripper.pos") if "gripper.pos" in names else None
    arm = [j for j in range(len(names)) if j != g_idx]
    home = np.median(np.stack([st[cum[i]] for i in range(n_ep)]), axis=0)
    rows = []
    for i in range(n_ep):
        s, u = st[cum[i]:cum[i + 1]], ac[cum[i]:cum[i + 1]]
        spd = np.abs(np.diff(s[:, arm], axis=0)).sum(1)
        mv = np.nonzero(spd > 0.3)[0]
        head = mv[0] / fps if len(mv) else np.nan
        tail = (len(spd) - 1 - mv[-1]) / fps if len(mv) else np.nan
        jump = float(np.abs(np.diff(s, axis=0)).max()) if len(s) > 1 else 0.0
        sdev = float(np.abs(s[0, arm] - home[arm]).max())
        lag = max(int(np.argmin([np.mean(np.abs(s[l:, j] - u[: len(u) - l, j])) for l in range(8)])) for j in arm)
        row = dict(ep=i, sec=round(len(s) / fps, 1), head_idle=round(head, 1), tail_idle=round(tail, 1),
                   start_dev=round(sdev, 1), max_jump=round(jump, 1), lag=lag)
        flags = []
        if not len(mv):
            flags.append("no arm motion")
        if len(mv) and tail > a.max_tail_idle_s:
            flags.append(f"tail idle {tail:.1f}s")
        if len(mv) and head > a.max_head_idle_s:
            flags.append(f"head idle {head:.1f}s")
        if sdev > a.max_start_dev:
            flags.append(f"start pose off by {sdev:.1f}")
        if jump > a.max_jump:
            flags.append(f"joint jump {jump:.1f}")
        if g_idx is not None:
            g = s[:, g_idx]
            lo, hi = a.hold_range
            inside = (g >= lo) & (g <= hi)
            best = run = 0
            for v in inside:
                run = run + 1 if v else 0
                best = max(best, run)
            row.update(g_start=round(float(g[0]), 1), g_end=round(float(g[-1]), 1), hold_s=round(best / fps, 1))
            if best / fps < a.min_hold_s:
                flags.append("no grasp hold")
        row["flags"] = "; ".join(flags)
        rows.append(row)
        for f in flags:
            warns.append(f"ep {i}: {f}")

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_rows", 10000)
    print()
    print(R.to_string(index=False))
    print()
    print(f"length    min {R.sec.min()}s  median {R.sec.median()}s  max {R.sec.max()}s")
    if g_idx is not None:
        gs, ge = R.g_start.round(0), R.g_end.round(0)
        print(f"gripper   start {gs.min()}..{gs.max()}   end {ge.min()}..{ge.max()}")
        if gs.max() - gs.min() > 5 or ge.max() - ge.min() > 5:
            warns.append("gripper start/end state is not the same in every episode")

    if not a.skip_load:
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset

            repo = a.repo_id or f"local/{root.name}"
            ds = LeRobotDataset(repo, root=root, delta_timestamps={"action": [i / fps for i in range(100)]})
            for i in sorted({0, n_ep // 2, n_ep - 1}):
                for idx in (int(cum[i]), int(cum[i + 1] - 1)):
                    item = ds[idx]
                    assert tuple(item["action"].shape) == (100, len(names))
                    for k in video_keys:
                        assert item[k].shape[0] == 3
            print(f"load      LeRobotDataset with a 100-step action chunk OK ({len(ds)} frames)")
        except Exception as e:
            errors.append(f"LeRobotDataset load failed: {type(e).__name__}: {e}")

    print()
    for w in warns:
        print(f"WARN  {w}")
    for e in errors:
        print(f"ERROR {e}")
    if errors:
        print(f"\nFAIL  {len(errors)} error(s), {len(warns)} warning(s)")
        sys.exit(1)
    print(f"\nPASS  {n_ep} episodes, {len(warns)} warning(s)")


if __name__ == "__main__":
    main()
