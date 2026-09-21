#!/usr/bin/env python
"""End-to-end touch calibration: touch known points on the mat, compare FK prediction vs reality.

The idea (`[柏宇說 2026-09-22]`): drive the follower arm so that the closed gripper fingertip
centre touches a known coordinate on the placement mat.  Record the six joint readings at the
moment of contact.  Then run the same joint_mapping + FK chain that the sim uses and compare
the predicted fingertip position against the known coordinate.  The difference is the
**end-to-end error** of the entire calibration pipeline — SCALE, OFFSET, URDF link lengths,
TCP definition — all in one number, with NO camera involved.

Constraints and joint control:
  * **Elbow flex & Shoulder lift freed** — to reach different radial distances on the table
    (e.g. x = 10–25 cm), the 2D planar arm MUST flex both shoulder_lift and elbow_flex.
    (If elbow_flex were locked, the arm could only reach a single fixed-radius circle!)
  * **Wrist flex freed, held vertical** — the operator guides the gripper to point straight down
    (perpendicular to the table) at contact.  This ensures:
      1. Fingertip touches the point at a well-defined single 3D position (TCP fingertip centre),
         without tilt-dependent contact shifts.
      2. The physical ground-truth pitch is known to be 90° downward.  The script calculates
         the FK-predicted wrist pitch, immediately revealing any pitch-chain offset or scale error!
  * **Gripper locked closed** — so the TCP is at the closed-grip position (the one the sim cares
    about during grasps), and the two fingertips converge to a single contact point.
  * **Wrist roll locked neutral** — keeps the fingers aligned along the central axis.
  * **Direct follower control** — no leader arm needed.  The operator holds the wrist/gripper
    by hand (supporting the arm so it does not collapse), touches the designated coordinate on
    the mat, and presses Enter.

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

🔴 Support the arm by holding the wrist/gripper before starting the session.  The arm
   will have torque disabled on pan, lift, elbow, and wrist_flex.
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
    "pitch_deg", "pitch_err_deg",
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


def fingertip_pitch_deg(joint_rad_5: list[float]) -> float:
    """Pitch angle of the fingertip link (link5) below horizontal, degrees.

    +90 deg means pointing straight down toward the table (-Z in base frame).
    0 deg means pointing horizontally forward.
    """
    t5 = fk.link5_transform(joint_rad_5)
    v = t5[:3, 0]  # link5's +X axis (finger extension direction)
    return math.degrees(math.asin(float(np.clip(-v[2], -1.0, 1.0))))


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
    """Free pan, lift, elbow, and wrist_flex; lock wrist_roll neutral and gripper closed.

    Why these joints:
      - 3 positioning joints (pan, lift, elbow) MUST move to reach arbitrary (x, y) on the table.
      - wrist_flex is freed so the operator keeps the gripper pointing vertically down at each point.
      - gripper is locked closed so the TCP is a single point of contact.
      - wrist_roll is locked neutral so the fingers do not spin.

    🔴 The operator must support the arm before calling this — four joints go limp.
    """
    bus = robot.bus

    locked = ["wrist_roll", "gripper"]
    freed = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex"]

    # Step 1: for locked joints, copy Present_Position → Goal_Position, then enable torque.
    for j in locked:
        pos = bus.read("Present_Position", j, normalize=False)
        bus.write("Goal_Position", j, pos, normalize=False)
    bus.enable_torque(locked)

    # Step 2: free the positioning and pitch joints so operator guides the arm.
    bus.disable_torque(freed)


def close_gripper(robot) -> None:
    """Command the gripper to close (fingertip contact position)."""
    bus = robot.bus
    closed_raw = 2056
    bus.write("Goal_Position", "gripper", closed_raw, normalize=False)
    bus.enable_torque("gripper")


# --------------------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------------------

def append_row(path: Path, row_dict: dict) -> None:
    """Append one measurement row to the CSV, creating the file with headers if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fieldnames = FIELDS
    if not is_new:
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fieldnames = tuple(header)

    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
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
    """Print residual statistics, pitch alignment, and per-point details from a recorded CSV."""
    rows = read_csv(path)
    if not rows:
        print(f"no data in {path}")
        return

    print(f"\n{'='*76}")
    print(f"  Touch-calibration analysis: {path.name}")
    print(f"  {len(rows)} point(s)")
    print(f"{'='*76}\n")

    # Header
    print(f"  {'id':>4s}  {'tgt_x':>6s} {'tgt_y':>6s} {'tgt_z':>6s}  "
          f"{'fk_x':>6s} {'fk_y':>6s} {'fk_z':>6s}  "
          f"{'pitch':>5s} {'e_hz':>5s} {'e_vt':>5s} {'e_3d':>5s}  note")
    print(f"  {'':>4s}  {'(cm)':>6s} {'(cm)':>6s} {'(cm)':>6s}  "
          f"{'(cm)':>6s} {'(cm)':>6s} {'(cm)':>6s}  "
          f"{'(deg)':>5s} {'(cm)':>5s} {'(cm)':>5s} {'(cm)':>5s}")
    print(f"  {'-'*4}  {'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*6} {'-'*6} {'-'*6}  "
          f"{'-'*5} {'-'*5} {'-'*5} {'-'*5}  {'-'*20}")

    errors_horiz = []
    errors_vert = []
    errors_total = []
    reach_list = []
    vert_vs_reach = []
    pitch_list = []

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

        # Compute pitch if available in row or reconstruct from state
        pitch = float("nan")
        if "pitch_deg" in r and r["pitch_deg"]:
            pitch = float(r["pitch_deg"])
        elif all(f"state_{j}" in r for j in JOINTS):
            pos_6 = [float(r[f"state_{j}"]) for j in JOINTS]
            rad_6 = JM.row_to_sim_rad(pos_6)
            pitch = fingertip_pitch_deg(rad_6[:5])

        errors_horiz.append(eh)
        errors_vert.append(fz - tz)  # signed vertical
        errors_total.append(e3)
        reach = math.hypot(tx, ty)
        reach_list.append(reach)
        vert_vs_reach.append((reach, fz - tz))
        pitch_list.append(pitch)

        p_str = f"{pitch:5.1f}" if not math.isnan(pitch) else "    -"
        print(f"  {pid:>4s}  {tx:6.1f} {ty:6.1f} {tz:6.1f}  "
              f"{fx:6.1f} {fy:6.1f} {fz:6.1f}  "
              f"{p_str} {eh:5.1f} {ev:5.1f} {e3:5.1f}  {note}")

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

    # Wrist pitch check
    pitch_valid = [p for p in pitch_list if not math.isnan(p)]
    if pitch_valid:
        pitch_arr = np.array(pitch_valid)
        pitch_err = pitch_arr - 90.0  # +90 deg is straight down
        print(f"\n  wrist pitch (ground truth ≈ 90.0° vertical):")
        print(f"    median pitch: {np.median(pitch_arr):.1f}°  "
              f"(median error: {np.median(pitch_err):+.1f}° from vertical)")
        if abs(np.median(pitch_err)) > 5.0:
            print(f"    🔴 pitch bias {np.median(pitch_err):+.1f}° > 5° — "
                  f"confirms a joint offset/scale issue in the pitch chain (lift/elbow/wrist)!")
        else:
            print(f"    ✅ pitch bias within ±5° — pitch chain alignment looks consistent.")

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

    print(f"\n  Diagnostics Guide:")
    print(f"    - Constant vertical/horizontal offset -> OFFSET error in joint_mapping.")
    print(f"    - Vertical error grows with reach    -> SCALE error in lift/elbow, or link length.")
    print(f"    - Pitch error from 90°               -> OFFSET in shoulder_lift/elbow_flex/wrist_flex.")
    print(f"    - Residual rotates with azimuth      -> shoulder_pan zero error.")
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

    target_z_cm = float(args.target_z)

    print()
    print("=" * 72)
    print("  TOUCH CALIBRATION SESSION")
    print()
    print("  How it works:")
    print("    1. Gripper is CLOSED and wrist_roll is LOCKED neutral.")
    print("       Shoulder pan, shoulder lift, elbow flex, and wrist flex are FREED.")
    print("    2. Hold the arm by the wrist/gripper (supporting its weight).")
    print("    3. Guide the fingertip so it touches a known point on the mat,")
    print("       keeping the gripper pointing straight down (vertical).")
    print("    4. Type the point's x and y coordinates (in cm, e.g. '15.0 -5.0').")
    print("    5. The script records joint readings, runs FK, and prints the error.")
    print()
    print(f"  Target height: z = {target_z_cm:.1f} cm above the table surface.")
    print(f"  Suggested range: x = 10–25 cm, y = −20–+20 cm.")
    print()
    print("  🔴 The arm will go limp! Hold the wrist/gripper before pressing Enter.")
    print("=" * 72)
    print()

    try:
        robot = connect_follower(args.config)
    except Exception as e:
        print(f"could not connect ({type(e).__name__}: {e})")
        return 1

    written = 0
    try:
        print("  closing gripper...")
        close_gripper(robot)
        time.sleep(0.5)

        print("  locking wrist_roll & gripper; freeing pan, lift, elbow, wrist_flex...")
        print("  🔴 HOLD THE ARM / WRIST NOW — four joints are about to go limp.")
        try:
            input("  press Enter when ready (q to quit): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0

        set_posture_for_touch(robot)
        print("  ✅ Joints freed.  Guide the fingertip (pointing down) to each target point.")
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

            # FK -> fingertip position and pitch
            fk_x, fk_y, fk_z = fingertip_above_table_cm(rad_5)
            pitch_deg = fingertip_pitch_deg(rad_5)
            pitch_err_deg = pitch_deg - 90.0

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
                "pitch_deg": f"{pitch_deg:.1f}",
                "pitch_err_deg": f"{pitch_err_deg:.1f}",
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
            print(f"    pitch:   {pitch_deg:.1f}° (offset from vertical: {pitch_err_deg:+.1f}°)")
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
    test_state = [0.0, 0.56, 2.03, -100.0, 0.56, 58.8]
    rad = JM.row_to_sim_rad(test_state)
    rad_5 = rad[:5]

    # FK in arm-base frame
    x_m, y_m, z_m = fingertip_position_m(rad_5)
    x_cm, y_cm, z_cm = fingertip_above_table_cm(rad_5)
    pitch = fingertip_pitch_deg(rad_5)

    print(f"  test .pos readings: {test_state}")
    print(f"  URDF rad:           {[f'{r:.4f}' for r in rad_5]}")
    print(f"  fingertip (base):   ({x_m*100:.2f}, {y_m*100:.2f}, {z_m*100:.2f}) cm")
    print(f"  fingertip (table):  ({x_cm:.2f}, {y_cm:.2f}, {z_cm:.2f}) cm")
    print(f"  fingertip pitch:    {pitch:.1f}° (relative to vertical: {pitch - 90.0:+.1f}°)")
    print()

    # Verify FK matches the dedicated fk module
    ee = fk.ee_position_m(rad_5)
    print(f"  fk.ee_position_m:   ({ee[0]*100:.2f}, {ee[1]*100:.2f}, {ee[2]*100:.2f}) cm")
    print()

    # Round-trip check
    back = JM.sim_rad_to_row(rad)
    max_err = max(abs(a - b) for a, b in zip(test_state, back))
    print(f"  round-trip max error: {max_err:.2e}")
    assert max_err < 1e-10, f"round-trip error too large: {max_err}"
    print("  ✅ round-trip OK")
    print()

    tip_reach = math.hypot(x_m, y_m)
    ee_reach = math.hypot(ee[0], ee[1])
    print(f"  fingertip reach: {tip_reach*100:.1f} cm  (ee_link reach: {ee_reach*100:.1f} cm)")
    print()
    print("selftest passed.")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve optimal joint offsets and scales from touch calibration CSV using least squares."""
    from scipy.optimize import least_squares

    path = Path(args.csv)
    if not path.exists():
        print(f"{path} not found")
        return 1
    rows = read_csv(path)
    if len(rows) < 4:
        print(f"need at least 4 points to solve, got {len(rows)}")
        return 1

    states = np.array([[float(r[f"state_{j}"]) for j in JOINTS] for r in rows])
    targets = np.array([[float(r["target_x_cm"]), float(r["target_y_cm"]), float(r["target_z_cm"])] for r in rows])

    nom_scale = 1.8 * np.pi / 180.0
    nom_scales_4 = np.array([nom_scale, nom_scale, nom_scale, nom_scale])

    init_scales = np.array([JM.SCALE_RAD_PER_UNIT[j] for j in JOINTS[:4]])
    init_offsets = np.array([JM.OFFSET_RAD[j] for j in JOINTS[:4]])
    init_wrist_roll_scale = nom_scale
    init_wrist_roll_offset = JM.OFFSET_RAD["wrist_roll"]

    fixed_riser = float(args.riser) if args.riser is not None else None

    def eval_points(scales_4, offsets_4, riser):
        tips, pitches = [], []
        for s in states:
            rad_4 = scales_4 * s[:4] + offsets_4
            rad_roll = init_wrist_roll_scale * s[4] + init_wrist_roll_offset
            rad_5 = list(rad_4) + [rad_roll]
            t5 = fk.link5_transform(rad_5)
            tip_m = t5 @ TCP_IN_LINK5
            tip_cm = np.array([tip_m[0] * 100.0, tip_m[1] * 100.0, tip_m[2] * 100.0 + riser])
            v = t5[:3, 0]
            tips.append(tip_cm)
            pitches.append(float(np.degrees(np.arcsin(np.clip(-v[2], -1.0, 1.0)))))
        return np.array(tips), np.array(pitches)

    # Initial baseline
    tips_init, pitches_init = eval_points(init_scales, init_offsets, 15.0)
    err_3d_init = np.linalg.norm(tips_init - targets, axis=1)

    fit_scales = args.fit_scales

    if not fit_scales:
        def obj_fn(param):
            offs = param[:4]
            r = param[4] if fixed_riser is None else fixed_riser
            tips, pitches = eval_points(nom_scales_4, offs, r)
            err_pos = (tips - targets).ravel()
            err_pitch = (pitches - 90.0) * 0.1
            res = [err_pos, err_pitch]
            if fixed_riser is None:
                res.append([(r - 14.0) * 0.2])
            return np.concatenate(res)

        p0 = np.concatenate([init_offsets, [14.0] if fixed_riser is None else []])
        res = least_squares(obj_fn, p0)
        solved_offsets = res.x[:4]
        solved_riser = float(res.x[4]) if fixed_riser is None else fixed_riser
        solved_scales = nom_scales_4
    else:
        def obj_fn(param):
            sc = param[:4]
            offs = param[4:8]
            r = param[8] if fixed_riser is None else fixed_riser
            tips, pitches = eval_points(sc, offs, r)
            err_pos = (tips - targets).ravel()
            err_pitch = (pitches - 90.0) * 0.1
            reg_scales = (sc - nom_scales_4) / nom_scales_4 * 0.5
            res = [err_pos, err_pitch, reg_scales]
            if fixed_riser is None:
                res.append([(r - 14.0) * 0.2])
            return np.concatenate(res)

        p0 = np.concatenate([nom_scales_4, init_offsets, [14.0] if fixed_riser is None else []])
        res = least_squares(obj_fn, p0)
        solved_scales = res.x[:4]
        solved_offsets = res.x[4:8]
        solved_riser = float(res.x[8]) if fixed_riser is None else fixed_riser

    tips_opt, pitches_opt = eval_points(solved_scales, solved_offsets, solved_riser)
    err_3d_opt = np.linalg.norm(tips_opt - targets, axis=1)

    print()
    print("=" * 76)
    print("  TOUCH CALIBRATION SOLVER")
    print(f"  Source CSV: {path.name} ({len(rows)} points)")
    print(f"  Mode: {'Fit SCALES + OFFSETS' if fit_scales else 'Nominal SCALES (1.80°/unit fixed) + Fit OFFSETS'}")
    print(f"  Arm Riser Height: {solved_riser:.2f} cm")
    print("=" * 76)
    print()
    print(f"  Overall 3D Error:  median {np.median(err_3d_init):.2f} cm -> {np.median(err_3d_opt):.2f} cm  (max: {np.max(err_3d_init):.2f} -> {np.max(err_3d_opt):.2f} cm)")
    print(f"  Pitch Error:       median {np.median(pitches_init - 90.0):+.1f}° -> {np.median(pitches_opt - 90.0):+.1f}°")
    print()
    print(f"  {'id':>3s}  {'tgt_x':>6s} {'tgt_y':>6s}  {'opt_x':>6s} {'opt_y':>6s} {'opt_z':>6s}  {'pitch':>5s}  {'err_init':>8s} -> {'err_opt':>7s}")
    print(f"  {'-'*3}  {'-'*6} {'-'*6}  {'-'*6} {'-'*6} {'-'*6}  {'-'*5}  {'-'*8}    {'-'*7}")
    for i, r in enumerate(rows):
        pid = r.get("point_id", str(i))
        tx, ty = targets[i, 0], targets[i, 1]
        ox, oy, oz = tips_opt[i]
        pv = pitches_opt[i]
        ei = err_3d_init[i]
        eo = err_3d_opt[i]
        print(f"  {pid:>3s}  {tx:6.1f} {ty:6.1f}  {ox:6.2f} {oy:6.2f} {oz:6.2f}  {pv:5.1f}°  {ei:6.2f} cm -> {eo:5.2f} cm")

    print()
    print("  --- Solved Constants (ready to paste into sim/joint_mapping.py) ---")
    print()
    print("SCALE_RAD_PER_UNIT: dict[str, float] = {")
    for i, j in enumerate(JOINTS[:4]):
        print(f'    "{j}": {solved_scales[i]:.8f},')
    print('    "wrist_roll": 0.03141593,')
    print("}")
    print()
    print("OFFSET_RAD: dict[str, float] = {")
    for i, j in enumerate(JOINTS[:4]):
        print(f'    "{j}": {solved_offsets[i]:.8f},')
    print(f'    "wrist_roll": {init_wrist_roll_offset:.8f},')
    print("}")
    print()
    print(f"# NOTE: ARM_RISER_HEIGHT solved as {solved_riser:.2f} cm (scene_constants.py)")
    print()
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

    v = sub.add_parser("solve", help="solve optimal joint offsets and scales from touch calibration CSV")
    v.add_argument("--csv", required=True, help="CSV file to solve from")
    v.add_argument("--fit-scales", action="store_true", help="also fit joint scales (default: fixed nominal 1.80 deg/unit)")
    v.add_argument("--riser", type=float, default=None, help="fixed riser height in cm (default: auto-solve near 14-15 cm)")
    v.set_defaults(func=cmd_solve)

    t = sub.add_parser("selftest", help="verify FK + joint_mapping on synthetic data (no hardware)")
    t.set_defaults(func=cmd_selftest)

    args = p.parse_args()
    if not hasattr(args, "func"):
        p.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

