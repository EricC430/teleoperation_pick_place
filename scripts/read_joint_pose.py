#!/usr/bin/env python
"""Print the OMX follower's current joint positions (LeRobot .pos units); optionally move it to the training start pose.

Default (read-only): only the motor bus is opened -- no configure(), no torque change, no action sent.

--goto-home: enables torque, interpolates to HOME over --seconds, prints the residual, and exits with
torque LEFT ON so the arm holds the pose for the next lerobot-rollout (which captures its startup pose as
initial_position and returns there between episodes). Caveat: rollout's connect() runs configure() inside
torque_disabled(), so the arm may sag briefly before that capture -- check frame 0 of ep 0 against HOME.

    uv run python scripts/read_joint_pose.py
    uv run python scripts/read_joint_pose.py --goto-home              # alcan start pose
    uv run python scripts/read_joint_pose.py --goto-home --home cup   # paper-cup start pose
    uv run python scripts/read_joint_pose.py --goto-home --home-episode 1   # before replaying uvc_60 ep 1
"""

import argparse
import time
from pathlib import Path

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
HOMES = {
    # Median observation.state at frame 0 of omx_pick_place_pilot_60_alcan_fixed (all 60 eps; std 0.5-2.6).
    # 2026-10-05: the 10:39 rollout started at wrist_flex +1.5 / gripper 50.6, outside the training range.
    "alcan": [0.12, -57.22, 54.99, -10.82, -1.61, 59.16],
    # Median frame 0 of rollout_omx_b1_uvc60_100k_paper_cup_20260918_010912 (36 eps, spread < 1): the start pose
    # of the N=100 paper-cup run, so an N=30 run from here changes only N. Inside the uvc_60 training range
    # (wrist_flex -19.5..1.4); the training median is [0.2, -62.8, 54.6, -6.2, 1.4, 59.2].
    "cup": [-0.66, -62.59, 55.21, 0.71, 0.90, 58.27],
}
MAX_JUMP = 30.0  # refuse --goto-home if any joint is further than this from HOME (a wrong pose, not drift)

ap = argparse.ArgumentParser()
ap.add_argument("--port", default="COM8")
ap.add_argument("--id", default="2026-09-18_omx_follower")
ap.add_argument("--goto-home", action="store_true", help="move to HOME and leave torque on")
ap.add_argument("--home", choices=sorted(HOMES), default="alcan", help="which start pose HOME is")
ap.add_argument("--home-episode", type=int, default=None,
                help="HOME = frame-0 action of this episode of --dataset-root (use before lerobot-replay); overrides --home")
ap.add_argument("--dataset-root", type=Path, default=Path(".cache/lerobot/omx_pick_place_pilot_uvc_60"))
ap.add_argument("--seconds", type=float, default=3.0, help="duration of the move to HOME")
ap.add_argument("--dry-run", action="store_true", help="build the robot and print its calibration; no serial port")
args = ap.parse_args()
HOME = dict(zip(JOINTS, HOMES[args.home]))
if args.home_episode is not None:
    import pandas as pd

    data = pd.concat(pd.read_parquet(f) for f in sorted((args.dataset_root / "data").rglob("*.parquet")))
    first = data[(data.episode_index == args.home_episode) & (data.frame_index == 0)]
    if first.empty:
        raise SystemExit(f"episode {args.home_episode} not found in {args.dataset_root}")
    HOME = dict(zip(JOINTS, (float(v) for v in first["action"].iloc[0])))
    args.home = f"{args.dataset_root.name} ep{args.home_episode} frame-0 action"

robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.id, calibration_dir=Path("calibration")))
if args.dry_run:
    print("calibration loaded:", {k: (c.homing_offset, c.range_min, c.range_max) for k, c in robot.bus.calibration.items()})
    print(f"HOME ({args.home}):", HOME)
    raise SystemExit(0)


def read() -> dict[str, float]:
    return robot.bus.sync_read("Present_Position", num_retry=2)


def show(label: str, pos: dict[str, float]) -> None:
    print(f"{label:8s}", {k: round(v, 1) for k, v in pos.items()})
    print(f"{'- HOME':8s}", {k: round(pos[k] - HOME[k], 1) for k in HOME})


robot.bus.connect()
try:
    start = read()
    show("present", start)
    if args.goto_home:
        far = {k: round(start[k] - HOME[k], 1) for k in HOME if abs(start[k] - HOME[k]) > MAX_JUMP}
        if far:
            raise SystemExit(f"refusing: {far} further than {MAX_JUMP} from HOME -- move the arm closer by hand first")
        robot.bus.enable_torque(num_retry=2)
        steps = max(int(args.seconds * 50), 1)
        for i in range(1, steps + 1):
            t = i / steps
            robot.bus.sync_write("Goal_Position", {k: start[k] * (1 - t) + HOME[k] * t for k in HOME})
            time.sleep(1 / 50)
        time.sleep(0.5)
        show("after", read())
        print("torque left ON -- start lerobot-rollout now; check ep 0 frame 0 against HOME afterwards")
finally:
    robot.bus.disconnect(False)
