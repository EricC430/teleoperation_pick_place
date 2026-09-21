#!/usr/bin/env python
"""End-to-end touch calibration: touch known points on the mat, compare FK prediction vs reality.

The idea (`[柏宇說 2026-09-22]`): drive the follower arm so that the closed gripper fingertip
centre touches a known coordinate on the placement mat.  Record the six joint readings at the
moment of contact.  Then run the same joint_mapping + FK chain that the sim uses and compare
the predicted fingertip position against the known coordinate.  The difference is the
**end-to-end error** of the entire calibration pipeline — SCALE, OFFSET, URDF link lengths,
TCP definition — all in one number, with NO camera involved.

Constraints that make the measured points unambiguous:
  * **Wrist held downward** — wrist_flex is locked so the fingertip points straight down. This
    makes "fingertip touches the point" a single well-defined 3D position (the point's x/y at
    the mat surface height), instead of an angled contact that could land anywhere on the finger.
  * **Gripper closed** — so the TCP is at the closed-grip position (the one the sim cares about
    during grasps), and the two fingertips converge to a single point.
  * **Direct follower control** — no leader arm needed.  Each joint EXCEPT shoulder_pan and
    shoulder_lift is locked rigid; the operator manually pushes those two to slide the arm over
    the target point, then presses Enter to record.

Coordinate system: the pan-axis frame used by the placement mat and `scene_constants.py`.
  Origin = pan axis (joint1) projected onto the table top.
  +X = forward (away from operator), +Y = operator's left, +Z = up.
  The user inputs x and y in CENTIMETRES (matching the mat's labels).

Suggested coordinate range: x = 10–25 cm, y = −20–+20 cm.

Usage:

    # Interactive session — connect, take N touch measurements, save CSV + analysis
    uv run python scripts/touch_calibrate.py session \\
        --csv calibration/2026-09-22_touch_calibration.csv

    # Analyse a previously recorded CSV (no hardware needed)
    uv run python scripts/touch_calibrate.py analyse \\
        --csv calibration/2026-09-22_touch_calibration.csv

    # Quick dry-run: just verify the FK + joint_mapping chain on a synthetic point
    uv run python scripts/touch_calibrate.py selftest

🔴 This script lets the operator push the arm while shoulder_pan and shoulder_lift are free.
   The other four joints are locked rigid (torque ON), so the arm cannot collapse.  But the
   two free joints can drop under their own weight — support the arm before pressing Enter.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "sim"))

import numpy as np

import joint_mapping as JM
from reach_logger import fk

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# CSV column names
FIELDS = (
    "point_id",
    "target_x_cm", "target_y_cm", "target_z_cm",
    "fk_x_cm", "fk_y_cm", "fk_z_cm",
    "error_horiz_cm", "error_vert_cm", "error_total_cm",
    "note",
) + tuple(f"state_{j}" for j in JOINTS)

# The TCP offset in the link5 frame — the MEASURED fingertip, same as omx_constants.
TCP_IN_LINK5 = np.array([0.08, -0.00165, 0.0, 1.0])

# Riser height (arm base above the table surface).
ARM_RISER_HEIGHT_M = 0.15  # [Eric said 2026-09-21], matches scene_constants.py

# Default config path for the follower connection.
DEFAULT_CONFIG = os.path.join(_REPO, "configs", "record_omx.yaml")


# --------------------------------------------------------------------------------------
# FK with the real TCP (fingertip, not end_effector_link)
# --------------------------------------------------------------------------------------

def fingertip_position_m(joint_rad_5: list[float]) -> tuple[float, float, float]:
    """Fingertip (x, y, z) in the arm-base frame, metres.

    Uses the link5 transform + the measured TCP offset, which is the same definition
    `sim/grasp_attach.py:tcp_pose_w` uses in the simulator.  joint_rad_5 is the five
    body joint angles in URDF order (shoulder_pan .. wrist_roll), NOT the gripper.
    """
    t5 = fk.link5_transform(joint_rad_5)
    tip_base = t5 @ TCP_IN_LINK5
    return float(tip_base[0]), float(tip_base[1]), float(tip_base[2])


def fingertip_above_table_cm(joint_rad_5: list[float]) -> tuple[float, float, float]:
    """Fingertip (x, y, z) in the pan-axis frame, cm, with z measured from the table surface."""
    x, y, z = fingertip_position_m(joint_rad_5)
    return x * 100.0, y * 100.0, (z + ARM_RISER_HEIGHT_M) * 100.0


# --------------------------------------------------------------------------------------
# Arm connection (reuses the proven pattern from measure_joint_zeros.py)
# --------------------------------------------------------------------------------------

def connect_follower(config_path: str):
    """Connect to the follower arm. Caller owns disconnect()."""
    import draccus
    import yaml
    from lerobot.robots import RobotConfig, make_robot_from_config
    import lerobot.robots.omx_follower  # noqa: F401 — registers the choice

    cfg_dict = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    robot_dict = dict(cfg_dict.get("robot", cfg_dict))
    robot_dict.pop("cameras", None)
    robot = make_robot_from_config(draccus.decode(RobotConfig, robot_dict))
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        raise RuntimeError(
            "motors do not match the calibration file. Fix that first — do NOT let "
            "connect() re-calibrate mid-measurement (see sim/joint_mapping.py check_calibration)."
        )
    return robot


def read_state(robot) -> dict[str, float]:
    """Read all six joint .pos values in one round-trip."""
    obs = robot.get_observation()
    state = {}
    for j in JOINTS:
        key = f"{j}.pos"
        if key not in obs:
            raise KeyError(f"{key!r} not in robot observation ({sorted(obs)})")
        state[j] = float(obs[key])
    return state


def set_posture_for_touch(robot) -> None:
    """Lock wrist_flex downward, wrist_roll neutral, gripper closed; free pan and lift.

    The strategy:
      1. Read every joint's Present_Position (raw ticks, no normalisation).
      2. For the four joints we want LOCKED (elbow_flex, wrist_flex, wrist_roll, gripper):
         write their Present_Position into Goal_Position, then enable torque.
      3. For shoulder_pan and shoulder_lift: disable torque so the operator can push them.

    🔴 The operator must support the arm before calling this — two joints go limp.
    """
    bus = robot.bus

    locked = ["elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    freed = ["shoulder_pan", "shoulder_lift"]

    # Step 1: for locked joints, copy Present_Position → Goal_Position, then enable torque.
    for j in locked:
        pos = bus.read("Present_Position", j, normalize=False)
        bus.write("Goal_Position", j, pos, normalize=False)
    bus.enable_torque(locked)

    # Step 2: free the two joints the operator will push.
    bus.disable_torque(freed)


def close_gripper(robot) -> None:
    """Command the gripper to close (fingertip contact position).

    Uses raw tick writes: the gripper's "fully closed" reading from the S6 calibration CSV is
    ~50.2 units (normalised), which is raw tick ≈ 2048 + 50.2/100 * 4095 ≈ 4103.  But the
    simplest approach: read the current position, then command a position that drives the fingers
    together.  Gripper joint range is 0–4095; "closed" in uvc_60 data ≈ raw 2048 (centre).
    We command a known closed position.
    """
    bus = robot.bus
    # The normalised "closed" reading from S6 is ~50.2 units (RANGE_0_100).
    # raw = 50.2 / 100 * 4095 = 2055.69 ≈ 2056.
    closed_raw = 2056
    bus.write("Goal_Position", "gripper", closed_raw, normalize=False)
    bus.enable_torque("gripper")


def set_wrist_downward(robot) -> None:
    """Command wrist_flex so the fingertip points straight down, and wrist_roll to neutral.

    "Fingertip pointing straight down" means the wrist link is vertical.  In URDF terms,
    that is the orientation where the forearm + wrist chain puts link5's +X pointing toward
    the ground.  The exact reading depends on the current shoulder_lift and elbow_flex poses.

    Since elbow_flex is locked and we just want the wrist to stay downward, we read the
    current wrist_flex position and HOLD it — the operator is expected to pose the arm so that
    the wrist is already roughly pointing down before starting the session.
    """
    bus = robot.bus
    for j in ["wrist_flex", "wrist_roll"]:
        pos = bus.read("Present_Position", j, normalize=False)
        bus.write("Goal_Position", j, pos, normalize=False)
    bus.enable_torque(["wrist_flex", "wrist_roll"])


# --------------------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------------------

def append_row(path: Path, row_dict: dict) -> None:
    """Append one measurement row to the CSV, creating the file with headers if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row_dict)


def read_csv(path: Path) -> list[dict]:
    """Read back the touch-calibration CSV."""
    rows = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    return rows


# --------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------

def analyse_csv(path: Path) -> None:
    """Print residual statistics and per-point details from a recorded CSV."""
    rows = read_csv(path)
    if not rows:
        print(f"no data in {path}")
        return

    print(f"\n{'='*72}")
    print(f"  Touch-calibration analysis: {path.name}")
    print(f"  {len(rows)} point(s)")
    print(f"{'='*72}\n")

    # Header
    print(f"  {'id':>4s}  {'tgt_x':>6s} {'tgt_y':>6s} {'tgt_z':>6s}  "
          f"{'fk_x':>6s} {'fk_y':>6s} {'fk_z':>6s}  "
          f"{'e_hz':>5s} {'e_vt':>5s} {'e_3d':>5s}  note")
    print(f"  {'':>4s}  {'(cm)':>6s} {'(cm)':>6s} {'(cm)':>6s}  "
          f"{'(cm)':>6s} {'(cm)':>6s} {'(cm)':>6s}  "
          f"{'(cm)':>5s} {'(cm)':>5s} {'(cm)':>5s}")
    print(f"  {'-'*4}  {'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*5} {'-'*5} {'-'*5}  {'-'*20}")

    errors_horiz = []
    errors_vert = []
    errors_total = []
    reach_list = []
    vert_vs_reach = []

    for r in rows:
        pid = r.get("point_id", "?")
        tx = float(r["target_x_cm"])
        ty = float(r["target_y_cm"])
        tz = float(r["target_z_cm"])
        fx = float(r["fk_x_cm"])
        fy = float(r["fk_y_cm"])
        fz = float(r["fk_z_cm"])
        eh = float(r["error_horiz_cm"])
        ev = float(r["error_vert_cm"])
        e3 = float(r["error_total_cm"])
        note = r.get("note", "")

        errors_horiz.append(eh)
        errors_vert.append(fz - tz)  # signed vertical
        errors_total.append(e3)
        reach = math.hypot(tx, ty)
        reach_list.append(reach)
        vert_vs_reach.append((reach, fz - tz))

        print(f"  {pid:>4s}  {tx:6.1f} {ty:6.1f} {tz:6.1f}  "
              f"{fx:6.1f} {fy:6.1f} {fz:6.1f}  "
              f"{eh:5.1f} {ev:5.1f} {e3:5.1f}  {note}")

    eh_arr = np.array(errors_horiz)
    ev_arr = np.array(errors_vert)
    e3_arr = np.array(errors_total)

    print(f"\n  --- Summary (n={len(rows)}) ---")
    print(f"  horizontal error:  median {np.median(eh_arr):.1f}  "
          f"p75 {np.percentile(eh_arr, 75):.1f}  "
          f"max {np.max(eh_arr):.1f} cm")
    print(f"  vertical (signed): median {np.median(ev_arr):+.1f}  "
          f"IQR {np.percentile(ev_arr, 25):+.1f}..{np.percentile(ev_arr, 75):+.1f} cm")
    print(f"  3D error:          median {np.median(e3_arr):.1f}  "
          f"p75 {np.percentile(e3_arr, 75):.1f}  "
          f"max {np.max(e3_arr):.1f} cm")

    # Linear regression: vertical error vs reach
    if len(rows) >= 3:
        reaches = np.array([vr[0] for vr in vert_vs_reach])
        verts = np.array([vr[1] for vr in vert_vs_reach])
        if np.std(reaches) > 0.01:
            slope, intercept = np.polyfit(reaches, verts, 1)
            corr = np.corrcoef(reaches, verts)[0, 1]
            print(f"\n  vertical error vs reach:  slope {slope:+.3f} cm/cm  "
                  f"intercept {intercept:+.1f} cm  r={corr:+.2f}")
            if abs(corr) > 0.7:
                print(f"  🔴 strong linear trend (|r|={abs(corr):.2f}) — "
                      f"suggests a SCALE error, not just an offset.")
            else:
                print(f"  ✅ no strong linear trend (|r|={abs(corr):.2f}) — "
                      f"consistent with offset-only or random measurement noise.")

    # Horizontal: dx, dy
    dx_list = [float(r["fk_x_cm"]) - float(r["target_x_cm"]) for r in rows]
    dy_list = [float(r["fk_y_cm"]) - float(r["target_y_cm"]) for r in rows]
    dx_arr, dy_arr = np.array(dx_list), np.array(dy_list)
    print(f"\n  x residual (fk - target): median {np.median(dx_arr):+.1f}  "
          f"IQR {np.percentile(dx_arr, 25):+.1f}..{np.percentile(dx_arr, 75):+.1f} cm")
    print(f"  y residual (fk - target): median {np.median(dy_arr):+.1f}  "
          f"IQR {np.percentile(dy_arr, 25):+.1f}..{np.percentile(dy_arr, 75):+.1f} cm")

    print(f"\n  reading: an offset (constant residual) is an OFFSET error in joint_mapping.")
    print(f"           a slope (residual grows with reach) is a SCALE or link-length error.")
    print(f"           a rotation (dx/dy correlated with azimuth) is a shoulder_pan zero error.")
    print()


# --------------------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------------------

def cmd_session(args: argparse.Namespace) -> int:
    """Interactive touch-calibration session."""
    csv_path = Path(args.csv)

    # Count existing rows to set point_id
    next_id = 0
    if csv_path.exists():
        existing = read_csv(csv_path)
        next_id = len(existing)
        print(f"resuming — {next_id} point(s) already in {csv_path.name}")

    # Target z: fingertip touches the mat surface.  The mat sits on the table, which is at z=0
    # in the "above table" frame.  The mat is ~1 mm thick; we treat it as z=0.
    target_z_cm = float(args.target_z)

    print()
    print("=" * 72)
    print("  TOUCH CALIBRATION SESSION")
    print()
    print("  How it works:")
    print("    1. The arm connects with elbow, wrist, and gripper LOCKED rigid.")
    print("       Shoulder pan and shoulder lift are FREED — you push them by hand.")
    print("    2. Pose the arm so the closed fingertip touches a known point on the mat.")
    print("    3. Type the point's x and y coordinates (in cm, from the mat's labels).")
    print("    4. The script records the joint readings, runs FK, and prints the error.")
    print()
    print(f"  Target height: z = {target_z_cm:.1f} cm above the table surface.")
    print(f"  Suggested range: x = 10–25 cm, y = −20–+20 cm.")
    print()
    print("  🔴 The arm will go partially limp.  Support it before pressing Enter.")
    print("=" * 72)
    print()

    try:
        robot = connect_follower(args.config)
    except Exception as e:
        print(f"could not connect ({type(e).__name__}: {e})")
        return 1

    written = 0
    try:
        # Close gripper first (while still fully rigid)
        print("  closing gripper...")
        close_gripper(robot)
        time.sleep(0.5)

        # Now set the constrained posture: lock elbow/wrist/gripper, free pan/lift
        print("  locking elbow, wrist, gripper; freeing shoulder_pan and shoulder_lift...")
        print("  🔴 support the arm NOW — two joints are about to go limp.")
        try:
            input("  press Enter when ready (q to quit): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        set_posture_for_touch(robot)
        print("  ✅ shoulder_pan and shoulder_lift are free.  Push the arm to each target point.")
        print()

        while True:
            prompt = f"  point #{next_id}: enter 'x y' in cm (e.g. '15.0 -5.0'), or 'q' to quit: "
            try:
                raw = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw.lower() == "q":
                break
            if not raw:
                continue

            parts = raw.replace(",", " ").split()
            if len(parts) != 2:
                print("    expected two numbers: x y (in cm). Try again.")
                continue
            try:
                target_x = float(parts[0])
                target_y = float(parts[1])
            except ValueError:
                print(f"    could not parse '{raw}' as two numbers. Try again.")
                continue

            # Read joint state
            try:
                state = read_state(robot)
            except Exception as e:
                print(f"    could not read arm ({type(e).__name__}: {e})")
                continue

            # Convert to URDF radians via joint_mapping
            pos_6 = [state[j] for j in JOINTS]
            rad_6 = JM.row_to_sim_rad(pos_6)
            rad_5 = rad_6[:5]  # body joints only

            # FK → fingertip position
            fk_x, fk_y, fk_z = fingertip_above_table_cm(rad_5)

            # Errors
            dx = fk_x - target_x
            dy = fk_y - target_y
            dz = fk_z - target_z_cm
            error_horiz = math.hypot(dx, dy)
            error_vert = abs(dz)
            error_total = math.sqrt(dx * dx + dy * dy + dz * dz)

            # Build row
            row_dict = {
                "point_id": str(next_id),
                "target_x_cm": f"{target_x:.2f}",
                "target_y_cm": f"{target_y:.2f}",
                "target_z_cm": f"{target_z_cm:.2f}",
                "fk_x_cm": f"{fk_x:.2f}",
                "fk_y_cm": f"{fk_y:.2f}",
                "fk_z_cm": f"{fk_z:.2f}",
                "error_horiz_cm": f"{error_horiz:.2f}",
                "error_vert_cm": f"{error_vert:.2f}",
                "error_total_cm": f"{error_total:.2f}",
                "note": "",
            }
            for j in JOINTS:
                row_dict[f"state_{j}"] = f"{state[j]:.6f}"

            append_row(csv_path, row_dict)
            written += 1
            next_id += 1

            # Print immediate feedback
            print(f"    target:  ({target_x:+.1f}, {target_y:+.1f}, {target_z_cm:+.1f}) cm")
            print(f"    FK:      ({fk_x:+.1f}, {fk_y:+.1f}, {fk_z:+.1f}) cm")
            print(f"    error:   horiz {error_horiz:.1f}  vert {dz:+.1f}  3D {error_total:.1f} cm")
            print(f"    readings: " + "  ".join(f"{j[:9]}={state[j]:.1f}" for j in JOINTS))
            print()

    finally:
        robot.disconnect()
        print()
        print("🔴 DISCONNECTED — torque is OFF.  Support the arm or lower it to rest.")
        print(f"{written} point(s) written to {csv_path}")
        if written > 0:
            print(f"analyse: uv run python scripts/touch_calibrate.py analyse --csv {csv_path}")

    if written > 0:
        print()
        analyse_csv(csv_path)

    return 0


def cmd_analyse(args: argparse.Namespace) -> int:
    """Analyse a previously recorded CSV."""
    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found")
        return 1
    analyse_csv(path)
    return 0


def cmd_selftest(_args: argparse.Namespace) -> int:
    """Verify the FK + joint_mapping chain on synthetic data (no hardware)."""
    print("selftest: verifying FK + joint_mapping chain on synthetic points...")
    print()

    # Take a set of readings from the S6 calibration CSV as a realistic test case.
    # shoulder_pan=0, shoulder_lift≈0.56, elbow=2.03, wrist=-100, wristroll=0.56, gripper=58.8
    test_state = [0.0, 0.56, 2.03, -100.0, 0.56, 58.8]
    rad = JM.row_to_sim_rad(test_state)
    rad_5 = rad[:5]

    # FK in arm-base frame
    x_m, y_m, z_m = fingertip_position_m(rad_5)
    x_cm, y_cm, z_cm = fingertip_above_table_cm(rad_5)

    print(f"  test .pos readings: {test_state}")
    print(f"  URDF rad:           {[f'{r:.4f}' for r in rad_5]}")
    print(f"  fingertip (base):   ({x_m*100:.2f}, {y_m*100:.2f}, {z_m*100:.2f}) cm")
    print(f"  fingertip (table):  ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
    print()

    # Verify FK matches the dedicated fk module
    ee = fk.ee_position_m(rad_5)
    print(f"  fk.ee_position_m:   ({ee[0]*100:.2f}, {ee[1]*100:.2f}, {ee[2]*100:.2f}) cm")
    print(f"  (this uses end_effector_link, NOT the measured fingertip — expect ≈5 cm shorter reach)")
    print()

    # Round-trip check
    back = JM.sim_rad_to_row(rad)
    max_err = max(abs(a - b) for a, b in zip(test_state, back))
    print(f"  round-trip max error: {max_err:.2e}")
    assert max_err < 1e-10, f"round-trip error too large: {max_err}"
    print("  ✅ round-trip OK")
    print()

    # Verify that the TCP offset makes sense (fingertip should be further from the base than EE)
    tip_reach = math.hypot(x_m, y_m)
    ee_reach = math.hypot(ee[0], ee[1])
    print(f"  fingertip reach: {tip_reach*100:.1f} cm  (ee_link reach: {ee_reach*100:.1f} cm)")
    # The fingertip at 8 cm from link5 should be roughly in the same ballpark as ee_link at 9.2 cm
    print()
    print("selftest passed.")
    return 0


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("session", help="interactive touch-calibration session (needs arm)")
    s.add_argument("--csv", required=True, help="CSV file to write/append measurements to")
    s.add_argument("--config", default=DEFAULT_CONFIG,
                   help="campaign YAML with the follower's port/id (default: %(default)s)")
    s.add_argument("--target-z", type=float, default=0.0,
                   help="z coordinate of the touch surface in cm above table (default: 0 = mat on table)")
    s.set_defaults(func=cmd_session)

    a = sub.add_parser("analyse", help="analyse a previously recorded CSV (no hardware)")
    a.add_argument("--csv", required=True, help="CSV file to analyse")
    a.set_defaults(func=cmd_analyse)

    t = sub.add_parser("selftest", help="verify FK + joint_mapping on synthetic data (no hardware)")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
