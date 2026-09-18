"""S5 host-side preparation: turn a real LeRobot dataset into what the sim side consumes.

Covers the parts of S5 §2 that do NOT need Isaac Sim, so they run on any laptop with `uv`:

  gap 1 (drive gains)   -> traj.npz (action/state in URDF radians) + real_tracking.json, i.e. how well
                           the REAL follower tracks its own action. That is the target the sim fit
                           (`sim/fit_drive_gains.py`) is compared against — a sim that tracks better
                           than the real arm is as wrong as one that tracks worse.
  gap 3 (attach/detach) -> grasp_segments.csv: per episode, the frame ranges where the gripper is
                           closed, and whether it stalled on an object (action - state stays offset).
                           This is what `--skip-grasp-frames` needs, whichever attach design is chosen.
  gap 2 (mimic), helper -> with --dump-gripper-frames, saves wrist-camera frames at the most-open and
                           most-closed gripper reading, so the gripper sign/offset in
                           `sim/joint_mapping.py` can be checked by eye against the sim finger gap.

Nothing here writes into the dataset. Outputs go to --out.

    uv run python scripts/s5_prepare_replay.py \
        --dataset-root .cache/lerobot/omx_pick_place_pilot_uvc_60 \
        --calibration calibration/2026-09-13_omx_follower.json \
        --out outputs/s5_prep/uvc_60
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import joint_mapping as JM  # noqa: E402

GRIPPER = 5


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True, help="LeRobot v3 dataset directory (has meta/ and data/)")
    p.add_argument("--calibration", required=True,
                   help="follower calibration used when the dataset was RECORDED (refused unless factory default)")
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument("--episodes", default=None, help="comma-separated episode indices (default: all)")
    p.add_argument("--open-window", type=int, default=15, help="frames at episode start that define 'open' (default 15 = 1 s)")
    p.add_argument("--close-drop", type=float, default=4.0,
                   help="gripper.pos below (open level - this) counts as closed (default 4.0; open~59, cup~50)")
    p.add_argument("--min-closed", type=int, default=8, help="shortest closed run kept as a grasp, frames (default 8)")
    p.add_argument("--stall", type=float, default=1.0,
                   help="|median(action - state)| of the gripper during a closed run above this = stalled. Weak evidence only: "
                        "on uvc_60 many closed runs hold the cup at ~49.6 with no stall because the leader stopped there too")
    p.add_argument("--max-lag", type=int, default=5, help="largest action->state lag searched, frames")
    p.add_argument("--dump-gripper-frames", action="store_true",
                   help="save wrist frames at the most-open / most-closed gripper reading (needs lerobot + video decode)")
    p.add_argument("--dry-run", action="store_true", help="read and report, write nothing")
    return p.parse_args()


def load_frames(root: Path, episodes: set[int] | None):
    import pandas as pd

    files = sorted((root / "data").glob("*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {root / 'data'}")
    df = pd.concat([pd.read_parquet(f, columns=["action", "observation.state", "episode_index", "frame_index", "index"])
                    for f in files])
    if episodes is not None:
        df = df[df.episode_index.isin(episodes)]
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)
    return df


def tracking_stats(action: np.ndarray, state: np.ndarray, ep: np.ndarray, max_lag: int) -> dict:
    """How the REAL follower tracks: |action[t] - state[t+lag]| per joint, in URDF degrees.

    Pairs never cross an episode boundary. lag=1 is the natural pairing (state is read, then the
    action is sent, then the next frame's state shows its effect); the best lag is reported too.
    """
    out = {}
    per_lag = {}
    for lag in range(0, max_lag + 1):
        same = ep[:-lag] == ep[lag:] if lag else np.ones(len(ep), bool)
        a = action[:-lag] if lag else action
        s = state[lag:] if lag else state
        err = np.degrees(np.abs(a[same] - s[same]))
        per_lag[lag] = err
    best = {}
    for j, name in enumerate(JM.URDF_NAMES):
        lag_p50 = {lag: float(np.percentile(e[:, j], 50)) for lag, e in per_lag.items()}
        best[name] = min(lag_p50, key=lag_p50.get)
    e1 = per_lag[1]
    signed = np.degrees(action[:-1] - state[1:])[ep[:-1] == ep[1:]]
    for j, name in enumerate(JM.URDF_NAMES):
        out[name] = {
            "lag1_abs_err_deg": {
                "p50": round(float(np.percentile(e1[:, j], 50)), 3),
                "p95": round(float(np.percentile(e1[:, j], 95)), 3),
                "max": round(float(e1[:, j].max()), 3),
            },
            "lag1_mean_signed_err_deg": round(float(signed[:, j].mean()), 3),
            "best_lag_frames_by_p50": best[name],
        }
    return out


def grasp_segments(df, args) -> list[dict]:
    rows = []
    for ep, g in df.groupby("episode_index"):
        s = np.stack(g["observation.state"].values)[:, GRIPPER]
        a = np.stack(g["action"].values)[:, GRIPPER]
        open_level = float(np.median(s[: args.open_window]))
        closed = s < open_level - args.close_drop
        runs, t = [], 0
        while t < len(closed):
            if closed[t]:
                start = t
                while t < len(closed) and closed[t]:
                    t += 1
                if t - start >= args.min_closed:
                    runs.append((start, t))  # [start, end)
            else:
                t += 1
        if not runs:
            rows.append({"episode_index": int(ep), "grasp_no": 0, "n_frames": len(g), "open_level": round(open_level, 2),
                         "close_start": "", "close_end": "", "min_gripper": round(float(s.min()), 2),
                         "median_action_minus_state": "", "stalled_on_object": "", "note": "no closed run found"})
            continue
        for k, (st, en) in enumerate(runs, start=1):
            # skip the closing transient: judge the stall on the settled second half of the run
            mid = st + (en - st) // 2
            med = float(np.median(a[mid:en] - s[mid:en]))
            rows.append({"episode_index": int(ep), "grasp_no": k, "n_frames": len(g), "open_level": round(open_level, 2),
                         "close_start": st, "close_end": en, "min_gripper": round(float(s[st:en].min()), 2),
                         "median_action_minus_state": round(med, 2), "stalled_on_object": abs(med) > args.stall,
                         "note": "regrasp" if len(runs) > 1 else ""})
    return rows


def dump_gripper_frames(root: Path, df, out: Path):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from PIL import Image

    ds = LeRobotDataset(repo_id=f"local/{root.name}", root=root)
    s = np.stack(df["observation.state"].values)[:, GRIPPER]
    picks = {"most_open": int(np.argmax(s)), "most_closed": int(np.argmin(s))}
    for label, i in picks.items():
        row = df.iloc[i]
        item = ds[int(row["index"])]
        img = item["observation.images.wrist"]  # (C, H, W) float in [0, 1]
        arr = (img.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)
        q = JM.lerobot_to_urdf_rad(np.stack([row["observation.state"]]))[0, GRIPPER]
        path = out / f"gripper_{label}_ep{int(row['episode_index'])}_f{int(row['frame_index'])}.png"
        Image.fromarray(arr).save(path)
        print(f"  {label:<12} gripper.pos={s[i]:.2f}  -> gripper_joint_1={np.degrees(q):.1f} deg [mapping 未確認]  {path}")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")  # Windows cp950 console cannot print the status marks
    args = parse_args()
    root = Path(args.dataset_root)
    info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
    fps = int(info["fps"])
    for key in ("action", "observation.state"):
        names = [n.removesuffix(".pos") for n in info["features"][key]["names"]]
        if tuple(names) != JM.LEROBOT_NAMES:
            raise SystemExit(f"🔴 {key} names {names} != {JM.LEROBOT_NAMES}")
    if info.get("robot_type") != "omx_follower":
        raise SystemExit(f"🔴 robot_type={info.get('robot_type')!r}, expected omx_follower")

    problems = JM.check_calibration(args.calibration)
    if problems:
        print("🔴 calibration is not factory default — joint_mapping's constant conversion does not apply:")
        for p in problems:
            print("   ", p)
        raise SystemExit(2)

    episodes = {int(e) for e in args.episodes.split(",")} if args.episodes else None
    df = load_frames(root, episodes)
    action_pos = np.stack(df["action"].values).astype(np.float64)
    state_pos = np.stack(df["observation.state"].values).astype(np.float64)
    ep = df["episode_index"].to_numpy()
    action_rad = JM.lerobot_to_urdf_rad(action_pos)
    state_rad = JM.lerobot_to_urdf_rad(state_pos)

    print(f"dataset   : {root}  ({info['total_episodes']} episodes, {fps} fps)")
    print(f"selected  : {len(np.unique(ep))} episodes, {len(df)} frames")
    print(f"calibration: {args.calibration}  (factory default ✅)")
    print("🔴 joint signs/zero offsets are [未確認] (sim/joint_mapping.py) — S4 §5-1 settles them.\n")

    q_deg = np.degrees(state_rad)
    print(f"{'urdf joint':<17}{'state range (deg)':<22}")
    for j, name in enumerate(JM.URDF_NAMES):
        print(f"{name:<17}{q_deg[:, j].min():7.1f} .. {q_deg[:, j].max():7.1f}")

    track = tracking_stats(action_rad, state_rad, ep, args.max_lag)
    print(f"\nREAL follower tracking, |action[t] - state[t+1]| (deg) — the sim gain fit's reference:")
    print(f"{'urdf joint':<17}{'p50':>7}{'p95':>8}{'max':>8}{'mean signed':>13}{'best lag':>10}")
    for name, r in track.items():
        e = r["lag1_abs_err_deg"]
        print(f"{name:<17}{e['p50']:7.2f}{e['p95']:8.2f}{e['max']:8.2f}{r['lag1_mean_signed_err_deg']:13.2f}"
              f"{r['best_lag_frames_by_p50']:>10}")
    print("  (gripper errors include stalling on the object — the sim fit scores the body joints only)")

    segs = grasp_segments(df, args)
    n_ep = len({r["episode_index"] for r in segs})
    no_grasp = [r["episode_index"] for r in segs if r["grasp_no"] == 0]
    multi = sorted({r["episode_index"] for r in segs if r["note"] == "regrasp"})
    stalled = [r for r in segs if r["stalled_on_object"] is True]
    closed_runs = [r for r in segs if r["grasp_no"]]
    print(f"\ngrasp segmentation ({n_ep} episodes):")
    print(f"  closed runs          : {len(closed_runs)}  (stalled on an object: {len(stalled)})")
    print(f"  episodes w/o a grasp : {no_grasp or 'none'}")
    print(f"  episodes w/ regrasp  : {multi or 'none'}")
    if closed_runs:
        lens = np.array([r["close_end"] - r["close_start"] for r in closed_runs])
        first = np.array([r["close_start"] for r in closed_runs if r["grasp_no"] == 1])
        print(f"  closed run length    : p50 {np.median(lens):.0f} / max {lens.max()} frames")
        print(f"  first close at frame : p50 {np.median(first):.0f} / min {first.min()} "
              f"-> frames before that are approach-only (usable without an attach design)")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "traj.npz",
        action_rad=action_rad.astype(np.float32),
        state_rad=state_rad.astype(np.float32),
        episode_index=ep.astype(np.int64),
        frame_index=df["frame_index"].to_numpy().astype(np.int64),
        fps=np.int64(fps),
        joint_names=np.array(JM.URDF_NAMES),
        source=np.array(str(root.resolve())),
        calibration=np.array(str(Path(args.calibration).resolve())),
        mapping_note=np.array("joint_mapping signs/zero offsets [未確認] at export time"),
    )
    (out / "real_tracking.json").write_text(json.dumps(
        {"dataset": str(root), "fps": fps, "episodes": sorted(int(e) for e in np.unique(ep)),
         "mapping": {"SIGN": dict(JM.SIGN), "BODY_ZERO_DEG": JM.BODY_ZERO_DEG.tolist(),
                     "GRIPPER_ZERO_DEG": JM.GRIPPER_ZERO_DEG, "status": "未確認"},
         "joints": track}, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "grasp_segments.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(segs[0].keys()))
        w.writeheader()
        w.writerows(segs)
    print(f"\nwrote {out / 'traj.npz'}, {out / 'real_tracking.json'}, {out / 'grasp_segments.csv'}")

    if args.dump_gripper_frames:
        print("wrist frames at the gripper extremes:")
        dump_gripper_frames(root, df, out)


if __name__ == "__main__":
    main()
