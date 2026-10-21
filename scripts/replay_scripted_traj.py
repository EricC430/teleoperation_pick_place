#!/usr/bin/env python
"""Replay a sim joint trajectory CSV (sim/scripted_pick_to_drop.py output) on the real OMX follower, open loop.

Only the motor bus is opened (no cameras). The CSV is sim radians (joint1-5 + gripper_joint_1); each row is
converted with sim/joint_mapping.py to LeRobot .pos units and sent as Goal_Position.

Safety order: convert -> check range and per-step jump -> move to row 0 slowly -> wait for Enter -> play.
Torque is LEFT ON at the end (like read_joint_pose.py --goto-home), so the arm does not drop; release it with
read_joint_pose.py --release while supporting the arm.

    uv run python scripts/replay_scripted_traj.py --dry-run                      # convert + check only, no serial port
    uv run python scripts/replay_scripted_traj.py --phases above descend         # first part only (recommended first run)
    uv run python scripts/replay_scripted_traj.py --speed 0.3                    # slower, full trajectory

Known limits (sim/scripted_pick_to_drop.py docstring): the cube is not the paper cup; the waypoint puts joint4
near 98 deg (limit 100); the descent leaves the TCP ~0.7 cm above the table; FK error grows far from the table.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "sim"))
import joint_mapping as jm  # noqa: E402

JOINTS = list(jm.LEROBOT_NAMES)
SIM_COLS = ["joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1"]
STEP_S = 0.02  # the CSV is one row per 20 ms policy step

ap = argparse.ArgumentParser()
ap.add_argument("--csv", type=Path, default=REPO / "sim/trajectories/scripted_c1_traj.csv")
ap.add_argument("--port", default="COM8")
ap.add_argument("--id", default="2026-10-08_omx_follower")
ap.add_argument("--phases", nargs="*", default=None, help="only these phases, in CSV order (above descend close lift over_arm drop_point)")
ap.add_argument("--speed", type=float, default=0.5, help="1.0 = the CSV's 20 ms per row; 0.5 = twice as slow")
ap.add_argument("--approach-seconds", type=float, default=4.0, help="slow move from the present pose to row 0")
ap.add_argument("--max-approach", type=float, default=40.0, help="refuse if any joint is further than this from row 0")
ap.add_argument("--max-step", type=float, default=3.0, help="refuse if (row-to-row change x speed) exceeds this (.pos units per 20 ms of CSV time)")
ap.add_argument("--dry-run", action="store_true", help="convert and check only; no serial port")
args = ap.parse_args()

with args.csv.open(encoding="utf-8") as f:
    rows = [r for r in csv.DictReader(f)]
if args.phases:
    rows = [r for r in rows if r["phase"] in args.phases]
if not rows:
    raise SystemExit(f"no rows (phases={args.phases}) in {args.csv}")
sim = np.array([[float(r[c]) for c in SIM_COLS] for r in rows])
traj = jm.urdf_rad_to_lerobot(sim)  # (N, 6) in JOINTS order

lo = np.array([-100.0] * 5 + [0.0])
hi = np.array([100.0] * 6)
problems = []
for j, name in enumerate(JOINTS):
    if traj[:, j].min() < lo[j] or traj[:, j].max() > hi[j]:
        problems.append(f"{name}: {traj[:, j].min():.1f}..{traj[:, j].max():.1f} outside {lo[j]:.0f}..{hi[j]:.0f}")
step = (np.abs(np.diff(traj, axis=0)).max(axis=0) if len(traj) > 1 else np.zeros(6)) * args.speed
for j, name in enumerate(JOINTS):
    if step[j] > args.max_step:
        problems.append(f"{name}: step x speed {step[j]:.2f} > {args.max_step}")

print(f"{len(traj)} rows, {len(traj) * STEP_S / args.speed:.1f} s at speed {args.speed}  ({args.csv.name})")
print(f"{'':14s}{'row0':>8s}{'min':>8s}{'max':>8s}{'step*spd':>10s}")
for j, name in enumerate(JOINTS):
    print(f"{name:14s}{traj[0, j]:8.1f}{traj[:, j].min():8.1f}{traj[:, j].max():8.1f}{step[j]:10.2f}")
if problems:
    raise SystemExit("refusing:\n  " + "\n  ".join(problems))
if args.dry_run:
    raise SystemExit(0)

from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig  # noqa: E402

robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.id, calibration_dir=Path("calibration")))
robot.bus.connect()
try:
    start = robot.bus.sync_read("Present_Position", num_retry=2)
    first = dict(zip(JOINTS, traj[0]))
    far = {k: round(start[k] - first[k], 1) for k in JOINTS if abs(start[k] - first[k]) > args.max_approach}
    if far:
        raise SystemExit(f"refusing: {far} further than {args.max_approach} from row 0 -- move the arm closer by hand first")
    print("present", {k: round(v, 1) for k, v in start.items()})
    print("row 0  ", {k: round(v, 1) for k, v in first.items()})
    input("Hand on the arm, area clear. Enter = move to row 0 (torque ON) ... ")
    robot.bus.sync_write("Goal_Position", start)  # goal = present BEFORE torque, or the servos lunge to a stale goal
    robot.bus.enable_torque(num_retry=2)
    n = max(int(args.approach_seconds * 50), 1)
    for i in range(1, n + 1):
        t = i / n
        robot.bus.sync_write("Goal_Position", {k: start[k] * (1 - t) + first[k] * t for k in JOINTS})
        time.sleep(1 / 50)
    time.sleep(0.5)
    print("at row 0", {k: round(v, 1) for k, v in robot.bus.sync_read("Present_Position", num_retry=2).items()})
    input("Enter = play (Ctrl+C stops; torque stays ON at the last goal) ... ")
    t0 = time.perf_counter()
    for i, row in enumerate(traj):
        robot.bus.sync_write("Goal_Position", dict(zip(JOINTS, (float(v) for v in row))))
        wait = t0 + (i + 1) * STEP_S / args.speed - time.perf_counter()
        if wait > 0:
            time.sleep(wait)
    time.sleep(0.5)
    end = robot.bus.sync_read("Present_Position", num_retry=2)
    print("last goal", {k: round(float(v), 1) for k, v in zip(JOINTS, traj[-1])})
    print("present  ", {k: round(v, 1) for k, v in end.items()})
    print("torque left ON -- support the arm before read_joint_pose.py --release")
finally:
    robot.bus.disconnect(False)
