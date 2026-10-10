#!/usr/bin/env python
"""Replay a sim joint trajectory (URDF radians, one row per control step) on the real OMX follower and log
what the encoders report, for a sim-to-real comparison.

Input: a CSV with `step, phase, joint1..joint5, gripper_joint_1` (extra columns are ignored), e.g.
`sim/trajectories/scripted_c1_traj.csv` -- the scripted pick-and-carry from Template-Pickup-Place-OMX-Bin-v0
(pickup_place_omx_bin_env_cfg.py: sim dt 0.01 x decimation 2 = 0.02 s per row, drop point (0, -25, 45) cm).
The joint values there are the sim's MEASURED joint positions, not its targets -- so the real arm is asked
to follow where the sim arm went, and "real minus sim" below is real tracking + whatever sim got wrong.

Units: rows go URDF rad -> LeRobot `.pos` through sim/joint_mapping.py (`urdf_rad_to_lerobot`).
🔴 --offsets is REQUIRED because the two candidates differ by 24 deg on wrist_flex:
  * joint_mapping  the file as it is now. shoulder_lift and wrist_flex carry +2.0 / +24.0 deg added in
                   62ab0ca (2026-09-23) to make ONE rendered frame (ep 0 frame 226) put the gripper in the cup.
  * touch_0922     the same table before that commit = the 2026-09-22 touch calibration solve
                   (calibration/2026-09-22_touch_calibration.csv), all five offsets measured.
Which one is the real arm's zero is exactly what a sim-to-real test is sensitive to; pick one on purpose.

What this does and does NOT measure:
  * encoders -> the joint-space gap: servo lag, overshoot, the gripper stalling elsewhere than in sim.
  * NOT the kinematic gap. The real TCP is never observed here; FK of the real joints through the same
    joint_mapping would just reproduce the sim TCP. To see where the real gripper actually ends up, use
    --pause-at <phase> and measure the pinch point against the CSV's tcp_*_pan / tcp_z (pan-axis frame).

Safety:
  * the arm first moves from wherever it is to row 0 over --approach-seconds (refused if any joint is
    further than --max-jump away -- put it near the start pose by hand, or use read_joint_pose.py --goto);
  * torque stays ON at the end (the arm holds the last pose). --release-at-end drops it -- support the arm;
  * Ctrl+C stops sending, holds the present pose, and still writes the log.

    uv run python scripts/replay_sim_traj.py sim/trajectories/scripted_c1_traj.csv --offsets touch_0922 --dry-run
    uv run python scripts/replay_sim_traj.py sim/trajectories/scripted_c1_traj.csv --offsets touch_0922 --speed 0.5 --until-phase above
    uv run python scripts/replay_sim_traj.py sim/trajectories/scripted_c1_traj.csv --offsets touch_0922 --pause-at close --pause-at drop_point
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "sim"))
import joint_mapping as jm  # noqa: E402

JOINTS = list(jm.LEROBOT_NAMES)
SIM_DT = 0.02  # pickup_place_omx_latched_obs_env_cfg.py: sim dt 0.01, decimation 2
OFFSETS = {
    "joint_mapping": dict(jm.OFFSET_RAD),
    # 62ab0ca^ -- the touch_calibrate solve before the render-tuned +2.0 / +24.0 deg
    "touch_0922": {**jm.OFFSET_RAD, "shoulder_lift": 0.08942211, "wrist_flex": 1.53990733},
}
BODY_LIMIT, GRIPPER_LIMIT = (-100.0, 100.0), (0.0, 100.0)

ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
ap.add_argument("csv", type=Path)
ap.add_argument("--offsets", choices=sorted(OFFSETS), required=True, help="joint zero table, see the docstring")
ap.add_argument("--port", default="COM8")
ap.add_argument("--id", default="2026-09-18_omx_follower")
ap.add_argument("--speed", type=float, default=1.0, help="time scale: 1.0 = sim speed (50 Hz), 0.5 = half speed")
ap.add_argument("--approach-seconds", type=float, default=4.0, help="duration of the move to row 0")
ap.add_argument("--max-jump", type=float, default=40.0, help="refuse if any joint starts further than this (.pos units) from row 0")
ap.add_argument("--until-phase", default=None, help="stop after the last row of this phase")
ap.add_argument("--pause-at", action="append", default=[], metavar="PHASE",
                help="hold at the last row of PHASE and wait for Enter (to measure the real TCP); repeatable")
ap.add_argument("--settle-seconds", type=float, default=1.0, help="keep logging this long after the last row")
ap.add_argument("--release-at-end", action="store_true", help="torque OFF at the end (the arm goes limp)")
ap.add_argument("--out", type=Path, default=None, help="log CSV (default outputs/sim_traj_replay/<csv stem>_<offsets>_<time>.csv)")
ap.add_argument("--dry-run", action="store_true", help="convert and check only; no serial port")
args = ap.parse_args()

# --- load + convert -------------------------------------------------------------------------------------
rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
phases = list(dict.fromkeys(r["phase"] for r in rows))
for p in args.pause_at + ([args.until_phase] if args.until_phase else []):
    if p not in phases:
        ap.error(f"phase {p!r} not in {phases}")
if args.until_phase:
    last = max(i for i, r in enumerate(rows) if r["phase"] == args.until_phase)
    rows = rows[: last + 1]
pause_rows = {max(i for i, r in enumerate(rows) if r["phase"] == p) for p in args.pause_at if any(r["phase"] == p for r in rows)}

q_sim = np.array([[float(r[n]) for n in jm.URDF_NAMES] for r in rows])
jm.OFFSET_RAD.update(OFFSETS[args.offsets])
cmd = jm.urdf_rad_to_lerobot(q_sim)

bad = [(JOINTS[j], round(float(cmd[:, j].min()), 1), round(float(cmd[:, j].max()), 1))
       for j, (lo, hi) in enumerate([BODY_LIMIT] * 5 + [GRIPPER_LIMIT])
       if cmd[:, j].min() < lo or cmd[:, j].max() > hi]
if bad:
    raise SystemExit(f"refusing: commands outside the normalised range: {bad}")

np.set_printoptions(precision=1, suppress=True, linewidth=150)
period = SIM_DT / args.speed
print(f"{args.csv.name}: {len(rows)} rows, phases {list(dict.fromkeys(r['phase'] for r in rows))}, offsets={args.offsets}")
print(f"  playback {1 / period:.0f} Hz ({len(rows) * period:.1f} s), speed x{args.speed}")
print(f"  {'':10s}{JOINTS}")
print(f"  row 0     {cmd[0]}")
print(f"  min       {cmd.min(0)}")
print(f"  max       {cmd.max(0)}")
print(f"  last      {cmd[-1]}")
print(f"  max |d|/row {np.abs(np.diff(cmd, axis=0)).max(0)}  (.pos units; 1 unit = 1.8 deg on body joints)")
if args.dry_run:
    raise SystemExit(0)

# --- robot ----------------------------------------------------------------------------------------------
from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig  # noqa: E402

cal_path = _REPO / "calibration" / f"{args.id}.json"
problems = jm.check_calibration(cal_path)
if problems:
    raise SystemExit(f"{cal_path.name}: joint_mapping's constant conversion does not apply: {problems}")

robot = OmxFollower(OmxFollowerConfig(port=args.port, id=args.id, calibration_dir=_REPO / "calibration"))
bus = robot.bus


def read() -> np.ndarray:
    pos = bus.sync_read("Present_Position", num_retry=2)
    return np.array([pos[j] for j in JOINTS], dtype=np.float64)


def write(goal: np.ndarray) -> None:
    bus.sync_write("Goal_Position", dict(zip(JOINTS, goal.tolist())))


out = args.out or _REPO / "outputs" / "sim_traj_replay" / (
    f"{args.csv.stem}_{args.offsets}_{dt.datetime.now():%Y%m%d_%H%M%S}.csv")
out.parent.mkdir(parents=True, exist_ok=True)
log: list[list] = []  # t, row, phase, cmd x6, meas x6

bus.connect()
try:
    start = read()
    far = {j: round(float(start[k] - cmd[0, k]), 1) for k, j in enumerate(JOINTS) if abs(start[k] - cmd[0, k]) > args.max_jump}
    if far:
        raise SystemExit(f"refusing: {far} further than {args.max_jump} from row 0 -- move the arm closer first, or raise --max-jump")
    write(start)  # goal = present BEFORE torque, or the servos lunge to a stale goal
    # Same operating modes / elbow PID as lerobot-record and -rollout (configure() runs with torque off: support the arm)
    robot.configure()
    start = read()

    n = max(int(args.approach_seconds / SIM_DT), 1)
    for i in range(1, n + 1):
        write(start + (cmd[0] - start) * i / n)
        time.sleep(SIM_DT)
    time.sleep(0.5)
    print("at row 0, residual", read() - cmd[0])
    input("Enter to start the replay (Ctrl+C to abort) ")

    t0 = time.perf_counter()
    tick = t0
    for i, r in enumerate(rows):
        write(cmd[i])
        log.append([time.perf_counter() - t0, int(r["step"]), r["phase"], *cmd[i], *read()])
        if i in pause_rows:
            time.sleep(1.0)
            log.append([time.perf_counter() - t0, int(r["step"]), r["phase"] + "#paused", *cmd[i], *read()])
            sim_tcp = [float(r[k]) * 100 for k in ("tcp_x_pan", "tcp_y_pan", "tcp_z") if k in r]
            print(f"paused at end of {r['phase']} (step {r['step']}); sim TCP (x, y, z) cm = {sim_tcp}")
            input("measure the real pinch point, then Enter to continue ")
            t0 += time.perf_counter() - tick  # the pause is not part of the trajectory's time axis
            tick = time.perf_counter()
            continue
        tick += period
        time.sleep(max(tick - time.perf_counter(), 0.0))
    t_end = time.perf_counter() + args.settle_seconds
    while time.perf_counter() < t_end:
        log.append([time.perf_counter() - t0, int(rows[-1]["step"]), "settle", *cmd[-1], *read()])
        time.sleep(SIM_DT)
except KeyboardInterrupt:
    print("\ninterrupted -- holding the present pose")
    write(read())
finally:
    if args.release_at_end:
        print("torque OFF -- support the arm")
    bus.disconnect(args.release_at_end)

# --- save + summarise -----------------------------------------------------------------------------------
with out.open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["t", "step", "phase", *(f"cmd_{j}" for j in JOINTS), *(f"meas_{j}" for j in JOINTS)])
    w.writerows([[round(x, 5) if isinstance(x, float) else x for x in row] for row in log])
print(f"wrote {len(log)} rows -> {out}")

if log:
    phase_col = [row[2] for row in log]
    err = np.array([row[9:15] for row in log]) - np.array([row[3:9] for row in log])  # meas - cmd, .pos units
    err_deg = err * np.array([np.degrees(jm.SCALE_RAD_PER_UNIT[j]) for j in JOINTS[:5]] + [1.0])
    print("meas - cmd: body joints in DEG, gripper in .pos units; mean |err| / max |err| per phase")
    print(f"  {'':16s}" + "".join(f"{j[:12]:>14s}" for j in JOINTS))
    for p in dict.fromkeys(phase_col):
        m = np.array([pc == p for pc in phase_col])
        cells = "".join(f"{a:6.1f} /{b:6.1f}" for a, b in zip(np.abs(err_deg[m]).mean(0), np.abs(err_deg[m]).max(0)))
        print(f"  {p:16s}{cells}")
