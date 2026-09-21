#!/usr/bin/env python
"""S6: solve each joint's reading->radians affine map from two physically unambiguous poses.

Spec: `docs/specs/S6_joint_zero_calibration.md`. Read that first -- it says which two poses to put
each joint in and how to verify them. This script is the arithmetic half and the recording helper.

    true_rad = SCALE * reading + OFFSET

Two poses per joint give two equations, so BOTH unknowns come out. One pose only solves OFFSET and
only if SCALE is already known -- which is exactly the assumption that turned out to be wrong for
the gripper, where the recorded channel spans ~10 units across a visibly full open/close.

Four subcommands. `record` and `session` need the arm; `solve` and `selftest` run anywhere.
For a real 12-row run prefer `session`: it holds ONE connection, so the arm keeps torque
between poses (spec §4). `record` is the per-row form, useful with --reading for hand entry:

    # 1. on the machine wired to the arm: append a measurement row
    uv run python scripts/measure_joint_zeros.py record \\
        --csv calibration/2026-09-21_joint_zeros.csv \\
        --joint shoulder_lift --pose A --physical-deg 90.0 --note "upper arm vertical"

    # the gripper is measured as jaw opening in mm, not as an angle (spec §4)
    uv run python scripts/measure_joint_zeros.py record \\
        --csv calibration/2026-09-21_joint_zeros.csv \\
        --joint gripper --pose B --physical-mm 34.0 --note "fully open, at the mechanical stop"

    # 2. anywhere, no hardware: solve and print the constants to paste
    uv run python scripts/measure_joint_zeros.py solve --csv calibration/2026-09-21_joint_zeros.csv

    # 2b. the whole 12-row walk on one connection, resumable (prefer this on a lab day)
    uv run python scripts/measure_joint_zeros.py session --csv calibration/2026-09-21_joint_zeros.csv

    # 3. anywhere: check the arithmetic against a synthetic case
    uv run python scripts/measure_joint_zeros.py selftest

`record` reads the arm through lerobot using the same config the campaign records with, so the
channel it logs is the same `observation.state` channel the sim consumes. If that import fails you
can still fill the CSV by hand -- the columns are documented in the spec.

🔴 This never edits `sim/joint_mapping.py`. It prints a block for a human to paste, so that "who
changed the mapping, from which measurement" stays answerable.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
# The six `state_*` columns hold the WHOLE arm's reading at the moment of the measurement, not
# just the joint being measured. They cost nothing to capture and they are what makes a wrong
# OFFSET recoverable: a protractor on the forearm measures its ABSOLUTE pitch, which is
# joint2 + joint3 (+ a constant) because joint2/3/4 all turn about +Y, so the number typed in for
# `elbow_flex` or `wrist_flex` is not that joint's own angle unless the upstream joints happen to
# sit at their own zero. With the full state on every row that correction can be computed later
# from the same CSV instead of remeasuring. See the spec's §4-b.
STATE_FIELDS = tuple(f"state_{j}" for j in JOINTS)
FIELDS = ("joint", "pose", "physical_deg", "physical_mm", "reading", "note") + STATE_FIELDS
MIN_READING_SPAN = 40.0          # spec §7; below this the division amplifies measurement error
DEG2RAD = math.pi / 180.0

# Mirrors the pose table in docs/specs/S6_joint_zero_calibration.md §4, so a `session` can prompt
# without the doc open. 🔴 The SPEC is the source of truth: if these one-liners ever disagree with
# it, the spec wins and this table is the bug. They are cues, not the verification method -- §4
# still says how to check each pose (protractor, ruler, mat lines).
POSE_CUES = {
    ("shoulder_pan", "A"): "手臂指向墊子 0° 線",
    ("shoulder_pan", "B"): "手臂指向墊子 −90° 線",
    ("shoulder_lift", "A"): "上臂完全垂直",
    ("shoulder_lift", "B"): "上臂完全水平",
    ("elbow_flex", "A"): "前臂完全垂直",
    ("elbow_flex", "B"): "前臂完全水平",
    ("wrist_flex", "A"): "腕段完全水平",
    ("wrist_flex", "B"): "腕段與前臂共線",
    ("wrist_roll", "A"): "兩指連線水平",
    ("wrist_roll", "B"): "兩指連線垂直",
    ("gripper", "A"): "完全閉合（兩指接觸）",
    ("gripper", "B"): "完全張開到機械停點",
}


def append_row(path: Path, joint: str, pose: str, physical_deg, physical_mm, reading, note,
               state: dict[str, float] | None = None) -> None:
    """One measurement row, in the column order of FIELDS (spec §6).

    An existing CSV keeps ITS header: a file written before the `state_*` columns existed stays
    readable and gets the extra values dropped, rather than being silently corrupted by writing 12
    values under a 6-column header.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    fieldnames = FIELDS
    if not is_new:
        with path.open(newline="", encoding="utf-8") as fh:
            header = next(csv.reader(fh), None)
        if header:
            fieldnames = tuple(header)
            missing = [c for c in STATE_FIELDS if c not in fieldnames]
            if missing:
                print(f"    ⚠️  {path.name} predates the state_* columns -- the whole-arm state for "
                      f"this row is NOT being saved. Start a new CSV to capture it.")
    row = {"joint": joint, "pose": pose,
           "physical_deg": "" if physical_deg is None else physical_deg,
           "physical_mm": "" if physical_mm is None else physical_mm,
           "reading": reading, "note": note}
    for j in JOINTS:
        row[f"state_{j}"] = "" if state is None else state[j]
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        if is_new:
            w.writeheader()
        w.writerow(row)


def cmd_record(args: argparse.Namespace) -> int:
    reading, state = args.reading, None
    if reading is None:
        try:
            state = state_from_arm(args.config)
        except Exception as e:  # noqa: BLE001
            print(f"could not read the arm ({type(e).__name__}: {e}).")
            print("Pass --reading explicitly, or fill the CSV by hand -- see the spec's §6.")
            return 1
        reading = state[args.joint]
    else:
        # A hand-supplied --reading has no whole-arm state behind it, so the state_* columns stay
        # blank. That is a real loss (see FIELDS): prefer letting the script read the arm.
        print("note: --reading given by hand, so the state_* columns stay empty for this row.")
    append_row(Path(args.csv), args.joint, args.pose, args.physical_deg, args.physical_mm,
               reading, args.note, state)
    print(f"appended {args.joint} pose {args.pose}: physical={args.physical_deg} reading={reading}")
    return 0


def free_only(robot, joint: str) -> None:
    """Hold every joint but `joint` rigid; release `joint` so it can be posed by hand.

    This is what makes spec §4-b's "the other joints must not move between A and B" a mechanical
    guarantee instead of an instruction to be careful. Only one link's worth of weight is ever
    loose, so the arm cannot collapse the way --hand-pose lets it.

    🔴 The Goal_Position write is not optional. Enabling torque on a servo whose Goal_Position
    still holds an older target makes it SNAP to that target -- the arm lurches. Copying
    Present_Position into Goal_Position first means "hold exactly where you are". Raw ticks
    (`normalize=False`) on both sides, so no calibration round-trip can shift the value.
    """
    bus = robot.bus
    others = [j for j in JOINTS if j != joint]
    for j in others:
        pos = bus.read("Present_Position", j, normalize=False)
        bus.write("Goal_Position", j, pos, normalize=False)
    bus.enable_torque(others)
    bus.disable_torque(joint)


def release_torque_interactive(robot) -> bool:
    """Drop torque so the arm can be posed by hand. Returns False if the operator backs out.

    Why this exists: `connect()` ends with torque ENABLED -- OmxFollower.configure() wraps its
    writes in `bus.torque_disabled()`, whose contract is "guarantees torque is re-enabled". With a
    leader attached you pose the follower by driving it and never need this. With the follower
    alone the arm is rigid and there is no way to reach spec §4's twelve poses, so torque has to go.

    🔴 This is a physical hazard, not a software one: the arm drops under its own weight the
    instant torque goes, and §4's warning about a sagging joint is about exactly that. Support the
    arm FIRST, then confirm.
    """
    print()
    print("=" * 72)
    print("🔴 --hand-pose: about to DISABLE TORQUE on every joint.")
    print()
    print("   The arm will go limp IMMEDIATELY and fall under its own weight.")
    print("   Before answering: put one hand under the forearm and take its weight, or lower")
    print("   the arm to a pose where it already rests on the table.")
    print()
    print("   Then, for each measurement: hold the joint in the pose, verify it with the")
    print("   protractor / ruler per spec §4, type the value, and KEEP HOLDING until the")
    print("   reading is echoed back -- the encoder is read at that moment, so a joint that")
    print("   sags before then records a pose you did not measure.")
    print("=" * 72)
    try:
        answer = input("   type 'yes' to release torque, anything else to abort: ").strip()
    except (EOFError, KeyboardInterrupt):
        print()
        answer = ""
    if answer.lower() != "yes":
        print("   aborted -- torque untouched, arm still rigid.")
        return False
    robot.bus.disable_torque()
    print("   torque OFF. The arm is limp.")
    return True


def cmd_session(args: argparse.Namespace) -> int:
    """Walk every un-recorded (joint, pose) on ONE held connection.

    Why this exists: `record` connects and disconnects per row, and OmxFollowerConfig's
    `disable_torque_on_disconnect` defaults to True -- so a 12-row run by `record` drops torque 12
    times, and spec §4 requires the joint to be HOLDING the pose you just measured. One connection
    for the whole walk keeps the arm powered between poses.
    """
    path = Path(args.csv)
    done = set()
    if path.exists():
        for r in csv.DictReader(path.open(encoding="utf-8")):
            done.add((r["joint"], r["pose"]))
    only = [j.strip() for j in args.only.split(",") if j.strip()]
    todo = [(j, po) for j in JOINTS for po in ("A", "B")
            if (j, po) not in done and (not only or j in only)]
    if not todo:
        print(f"every (joint, pose) asked for is already in {path} -- nothing to do. Run `solve`.")
        return 0

    try:
        robot = connect_follower(args.config)
    except Exception as e:  # noqa: BLE001
        print(f"could not connect ({type(e).__name__}: {e}).")
        print("Fall back to `record --reading <value>` and fill the rows by hand (spec §6).")
        return 1

    if args.hand_pose:
        if not release_torque_interactive(robot):
            robot.disconnect()
            return 1
    if args.lock_others:
        print()
        print("=" * 72)
        print("🔒 --lock-others: every joint stays rigid except the one being measured.")
        print("   Before each joint, that ONE joint is released and drops under its own weight;")
        print("   you will be asked to support it first. The rest of the arm holds, so it cannot")
        print("   collapse, and spec §4-b's \"others must not move between A and B\" is enforced")
        print("   by the servos rather than by being careful.")
        print("=" * 72)

    print(f"connected. {len(todo)} measurement(s) to take on ONE held connection --")
    print("the arm stays powered between poses. Blank skips, 'q' or Ctrl-C stops; rows already")
    print("written are kept, so you can resume this same CSV later.")
    written = 0
    freed = None       # which joint is currently torque-off under --lock-others
    try:
        for j, po in todo:
            unit = "mm (jaw opening)" if j == "gripper" else "deg"
            cue = POSE_CUES.get((j, po), "see the spec's §4")
            print()
            print(f"--- {j} pose {po}: {cue}")
            if args.lock_others and j != freed:
                print(f"    🔴 about to release {j} -- it will drop. Every other joint stays rigid.")
                try:
                    go = input(f"    support that link, then press Enter (q to quit): ").strip()
                except (EOFError, KeyboardInterrupt):
                    print()
                    break
                if go.lower() == "q":
                    break
                try:
                    free_only(robot, j)
                except Exception as e:  # noqa: BLE001
                    print(f"    could not switch torque ({type(e).__name__}: {e}) -- stopping.")
                    break
                freed = j
                print(f"    {j} is loose; the rest is holding.")
            try:
                raw = input(f"    measured {unit} (blank=skip, q=quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if raw.lower() == "q":
                break
            if not raw:
                print("    skipped")
                continue
            try:
                value = float(raw)
            except ValueError:
                print(f"    {raw!r} is not a number -- skipped")
                continue
            try:
                state = read_state(robot)
            except Exception as e:  # noqa: BLE001
                print(f"    could not read the arm ({type(e).__name__}: {e}) -- skipped")
                continue
            reading = state[j]
            append_row(path, j, po,
                       None if j == "gripper" else value,
                       value if j == "gripper" else None,
                       reading, cue, state)
            written += 1
            print(f"    recorded: physical={value} reading={reading}")
            print("      whole-arm state: "
                  + "  ".join(f"{n[:9]}={state[n]:.1f}" for n in JOINTS))
    finally:
        robot.disconnect()
        print()
        print("🔴 DISCONNECTED -- torque is OFF and the arm will fall if nothing holds it.")
        print("   (`disable_torque_on_disconnect` defaults to True; this happens on every exit,")
        print("    hand-pose mode or not.) Support it or lower it to a rest pose before letting go.")
        print(f"{written} row(s) written to {path}.")
        print("Next: measure_joint_zeros.py solve --csv " + str(path))
    return 0


def connect_follower(config_path: str):
    """Build and connect the follower from a campaign YAML. The caller owns disconnect().

    2026-09-21 fix: this used `lerobot.configs.parser.load_config`, which does not exist in the
    pinned lerobot -- `lerobot/src/lerobot/configs/parser.py` has no such function, so EVERY
    `record` call fell through to "pass --reading yourself". The loading below is the pattern
    already proven on this machine by `scripts/measure_teleop_offset.py`: read the YAML, decode the
    robot section with draccus, and never build or connect the leader.
    """
    import draccus  # noqa: PLC0415
    import yaml  # noqa: PLC0415
    from lerobot.robots import RobotConfig, make_robot_from_config  # noqa: PLC0415
    import lerobot.robots.omx_follower  # noqa: F401, PLC0415  -- registers the "omx_follower" choice

    cfg_dict = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    robot_dict = dict(cfg_dict.get("robot", cfg_dict))
    # S6 reads joint values only. Dropping the cameras skips the OpenCV-index trap documented in
    # `configs/record_omx.yaml` and makes connect() near-instant. It cannot change a `.pos` value.
    robot_dict.pop("cameras", None)
    robot = make_robot_from_config(draccus.decode(RobotConfig, robot_dict))

    # 🔴 calibrate=False on purpose. OmxFollower.calibrate() starts with bus.disable_torque() and
    # then WRITES a new calibration file -- during S6 that would (a) let the arm sag out of the pose
    # you just measured and (b) redefine the very normalisation these readings are supposed to
    # pin down. If the motors disagree with the calibration file, S6 must stop, not silently re-zero.
    robot.connect(calibrate=False)
    if not robot.is_calibrated:
        robot.disconnect()
        raise RuntimeError(
            "the motors do not match the calibration file. S6 readings are only meaningful "
            "under a known calibration -- fix that first (see sim/joint_mapping.py's "
            "check_calibration), do NOT let connect() re-calibrate mid-measurement")
    return robot


def read_state(robot) -> dict[str, float]:
    """Every joint's current `.pos`, off an already-connected follower, in one round trip."""
    obs = robot.get_observation()
    state = {}
    for j in JOINTS:
        key = f"{j}.pos"
        if key not in obs:
            raise KeyError(f"{key!r} not in the robot's observation ({sorted(obs)})")
        state[j] = float(obs[key])
    return state



def state_from_arm(config_path: str) -> dict[str, float]:
    """One-shot connect -> read the whole arm -> disconnect, for a single `record` call.

    ⚠️ `disable_torque_on_disconnect` defaults to True, so this drops torque on the way out. Use
    the `session` subcommand for a full 12-row run: it holds one connection so the arm stays
    powered between poses.
    """
    robot = connect_follower(config_path)
    try:
        return read_state(robot)
    finally:
        robot.disconnect()


# [已查證 2026-09-21] from assets/omx_f/omx_f.urdf. Every <origin rpy> in that file is "0 0 0",
# so the zero pose is fixed entirely by these link vectors (parent joint -> child joint, metres,
# (x, z) in the parent frame). They are what turns a measured ABSOLUTE link angle into a per-joint
# URDF angle -- see the spec's §4-a.
URDF_UPPER_ARM_XZ = (0.0415, 0.11315)   # joint2 -> joint3
URDF_FOREARM_XZ = (0.162, 0.0)          # joint3 -> joint4
URDF_WRIST_XZ = (0.0287, 0.0)           # joint4 -> joint5


def _tilt_from_up(xz: tuple[float, float]) -> float:
    """Degrees this link leans from straight up at URDF zero, positive leaning forward (+X)."""
    return math.degrees(math.atan2(xz[0], xz[1]))


def fit_two(rows: list[dict], true_deg) -> dict:
    """Two rows + a function giving each row's TRUE joint angle in degrees -> scale/offset in rad."""
    a, b = rows[:2]
    dr = float(b["reading"]) - float(a["reading"])
    if dr == 0:
        raise ValueError("the two poses report the SAME reading -- nothing to solve")
    ta, tb = true_deg(a) * DEG2RAD, true_deg(b) * DEG2RAD
    scale = (tb - ta) / dr
    return {"scale": scale, "offset": ta - scale * float(a["reading"]), "reading_span": abs(dr)}


def solve_chain(by_joint: dict[str, list[dict]]) -> tuple[dict, list[str]]:
    """Per-joint URDF angles, walking DOWN the arm and subtracting what the upstream joints add.

    A protractor reads a link's ABSOLUTE orientation. joint2/3/4 all turn about +Y, so that angle
    is a running sum: upper arm = j2 + tilt, forearm = j2 + j3 + tilt, wrist = j2 + j3 + j4 + tilt.
    `solve_one` treats the typed number as the joint's own angle, which is right for the SCALE
    (the two poses differ only in that joint, enforced by --lock-others) and wrong for the OFFSET
    by exactly the upstream contribution. This recovers it from the `state_*` columns, so nothing
    has to be remeasured. Spec §4-a/§4-b.
    """
    out, problems = {}, []

    def usable(j):
        rs = [r for r in by_joint.get(j, []) if r.get("physical_deg") not in (None, "")]
        if len(rs) < 2:
            problems.append(f"{j}: needs 2 poses with physical_deg for the chain solve")
            return None
        if any(r.get("state_shoulder_lift") in (None, "") for r in rs):
            problems.append(f"{j}: no state_* columns -- cannot subtract the upstream joints")
            return None
        return rs[:2]

    # joint1 (yaw) and joint5 (roll) are measured in their own plane; no upstream term.
    # [AI推論] joint1 assumes the mat's 0° line is the URDF's +X; joint5 assumes the finger line is
    # along +Y at joint5=0, which the URDF's finger pivots (y=+0.0075 / -0.0108) support.
    for j in ("shoulder_pan", "wrist_roll"):
        rs = usable(j)
        if rs:
            out[j] = fit_two(rs, lambda r: float(r["physical_deg"]))

    rs = usable("shoulder_lift")
    if rs:
        tilt = _tilt_from_up(URDF_UPPER_ARM_XZ)
        out["shoulder_lift"] = fit_two(rs, lambda r: float(r["physical_deg"]) - tilt)

    def j_deg(name, reading):
        v = out[name]
        return math.degrees(v["scale"] * float(reading) + v["offset"])

    rs = usable("elbow_flex")
    if rs and "shoulder_lift" in out:
        tilt = _tilt_from_up(URDF_FOREARM_XZ)
        out["elbow_flex"] = fit_two(rs, lambda r: float(r["physical_deg"]) - tilt
                                    - j_deg("shoulder_lift", r["state_shoulder_lift"]))

    rs = usable("wrist_flex")
    if rs and "shoulder_lift" in out and "elbow_flex" in out:
        tilt = _tilt_from_up(URDF_WRIST_XZ)
        out["wrist_flex"] = fit_two(rs, lambda r: float(r["physical_deg"]) - tilt
                                    - j_deg("shoulder_lift", r["state_shoulder_lift"])
                                    - j_deg("elbow_flex", r["state_elbow_flex"]))
    return out, problems


def solve_one(rows: list[dict]) -> dict:
    """Two rows for one joint -> SCALE (rad per reading unit) and OFFSET (rad)."""
    a, b = rows
    dr = float(b["reading"]) - float(a["reading"])
    if dr == 0:
        raise ValueError("the two poses report the SAME reading -- nothing to solve")
    ta = float(a["physical_deg"]) * DEG2RAD
    tb = float(b["physical_deg"]) * DEG2RAD
    scale = (tb - ta) / dr
    offset = ta - scale * float(a["reading"])
    return {"scale": scale, "offset": offset, "reading_span": abs(dr)}


def cmd_solve(args: argparse.Namespace) -> int:
    """Solve every joint and print a pasteable block.

    Exit code: 1 only when something is genuinely UNRESOLVED (a joint missing a pose, a zero span,
    a span too small to trust). The gripper's "mm, not radians" result is a NOTE, not a problem --
    it is the correct, expected outcome of measuring jaw opening, and used to make a fully
    successful run exit 1, which read as failure.
    """
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    by_joint: dict[str, list[dict]] = {}
    for r in rows:
        by_joint.setdefault(r["joint"], []).append(r)

    print(f"{'joint':<15}{'scale(rad/unit)':>18}{'vs pi/180':>12}{'offset(deg)':>14}{'span':>8}  note")
    results: dict[str, dict] = {}
    gripper_mm: dict | None = None
    problems: list[str] = []   # blocking -- exit 1
    notes: list[str] = []      # informational -- exit stays 0
    for j in JOINTS:
        rs = by_joint.get(j, [])
        mm_rows = [r for r in rs if r.get("physical_mm") not in (None, "")]
        if j == "gripper" and len(mm_rows) >= 2:
            # The gripper's measurable truth is jaw opening in mm, not a joint angle. Solve
            # reading->mm and report it separately: turning mm into gripper_joint_1 radians needs
            # the finger linkage geometry, which nothing in this repo has measured. Emitting a
            # radian constant from a mm measurement would be inventing that geometry silently.
            a, b = mm_rows[:2]
            dr = float(b["reading"]) - float(a["reading"])
            if dr == 0:
                problems.append("gripper: both poses report the same reading")
                continue
            mm_per_unit = (float(b["physical_mm"]) - float(a["physical_mm"])) / dr
            mm_at_zero = float(a["physical_mm"]) - mm_per_unit * float(a["reading"])
            gripper_mm = {"mm_per_unit": mm_per_unit, "mm_at_zero": mm_at_zero, "span": abs(dr)}
            print(f"{j:<15}{'(mm, not rad)':>18}{'':>12}{'':>14}{abs(dr):>8.0f}"
                  f"  jaw = {mm_per_unit:.3f} mm/unit + {mm_at_zero:.1f} mm")
            notes.append("gripper: solved as reading->mm (this is the expected result, not a "
                         "failure). Converting it to gripper_joint_1 radians needs the finger "
                         "linkage geometry, which is NOT measured -- see the spec's §4 and gap 2's "
                         "amplitude residual in sim/README.md")
            continue

        usable = [r for r in rs if r.get("physical_deg") not in (None, "")]
        if len(usable) < 2:
            problems.append(f"{j}: {len(usable)} usable pose(s), need 2")
            continue
        try:
            out = solve_one(usable[:2])
        except ValueError as e:
            problems.append(f"{j}: {e}")
            continue
        ratio = out["scale"] / DEG2RAD
        flag = ""
        if out["reading_span"] < MIN_READING_SPAN:
            flag = f"  ⚠️ span {out['reading_span']:.0f} < {MIN_READING_SPAN:.0f}, error amplified"
            problems.append(f"{j}: reading span only {out['reading_span']:.1f}")
        if abs(ratio - 1.0) > 0.05:
            flag += f"  🔴 scale is {ratio:.2f}x degrees -- NOT a plain deg->rad channel"
        results[j] = out
        print(f"{j:<15}{out['scale']:>18.6f}{ratio:>11.2f}x{out['offset']/DEG2RAD:>14.2f}"
              f"{out['reading_span']:>8.0f}{flag}")

    if notes:
        print("\nℹ️  notes (not failures):")
        for n in notes:
            print(f"   {n}")
    if problems:
        print("\n⚠️ unresolved:")
        for pr in problems:
            print(f"   {pr}")

    chained = {}
    if args.chain:
        chained, chain_problems = solve_chain(by_joint)
        problems.extend(chain_problems)
        if chained:
            print("\n" + "-" * 72)
            print("--chain: the same fits with the upstream joints subtracted (spec 4-a)")
            print(f"{'joint':<15}{'scale(rad/unit)':>18}{'offset(deg)':>14}{'offset shift':>15}")
            for j in JOINTS:
                v = chained.get(j)
                if not v:
                    continue
                was = results.get(j)
                shift = "" if not was else f"{(v['offset'] - was['offset']) / DEG2RAD:+.2f}"
                print(f"{j:<15}{v['scale']:>18.6f}{v['offset'] / DEG2RAD:>14.2f}{shift:>15}")
            results = {**results, **chained}

    if results or gripper_mm:
        print("\n" + "=" * 72)
        print("paste into sim/joint_mapping.py (replacing the placeholder tables):\n")
        print(f"# [已查證 {args.date}] measured per S6, from {args.csv}")
        print("SCALE_RAD_PER_UNIT: dict[str, float] = {")
        for j in JOINTS:
            v = results.get(j)
            print(f'    "{j}": {v["scale"]:.8f},' if v else f'    "{j}": math.pi / 180.0,   # NOT measured')
        print("}")
        print("OFFSET_RAD: dict[str, float] = {")
        for j in JOINTS:
            v = results.get(j)
            print(f'    "{j}": {v["offset"]:.8f},' if v else f'    "{j}": 0.0,   # NOT measured')
        print("}")
        if gripper_mm:
            print("# The gripper was measured as JAW OPENING, so it is absent from the two tables")
            print("# above (they are radians). These two numbers are mm and are the honest result;")
            print("# turning them into gripper_joint_1 radians needs the finger linkage geometry,")
            print("# which is NOT measured -- see docs/specs/S6_joint_zero_calibration.md §4.")
            print(f"GRIPPER_MM_PER_UNIT = {gripper_mm['mm_per_unit']:.8f}")
            print(f"GRIPPER_MM_AT_ZERO = {gripper_mm['mm_at_zero']:.8f}")
    return 1 if problems else 0


def cmd_selftest(_: argparse.Namespace) -> int:
    """Round-trip the solver on a known answer -- runs anywhere, no arm, no CSV."""
    true_scale, true_offset = 0.9 * DEG2RAD, 12.0 * DEG2RAD
    rows = [{"reading": r, "physical_deg": (true_scale * r + true_offset) / DEG2RAD}
            for r in (-60.0, 40.0)]
    out = solve_one(rows)
    ok = abs(out["scale"] - true_scale) < 1e-12 and abs(out["offset"] - true_offset) < 1e-12
    print(f"scale  solved {out['scale']:.8f}  expected {true_scale:.8f}")
    print(f"offset solved {out['offset']:.8f}  expected {true_offset:.8f}")
    print("✅ solver round-trips" if ok else "🔴 solver is WRONG")
    return 0 if ok else 1


def force_utf8_console() -> None:
    """Make the Chinese pose cues and the flag emoji render, instead of cp950 mojibake.

    Two halves, and BOTH are needed on Windows. Reconfiguring the stream only decides which bytes
    Python emits; the console still decodes them with its own code page, so UTF-8 bytes into a
    cp950 console are mojibake of a different flavour. SetConsoleOutputCP(65001) is what makes the
    console read them as UTF-8. When stdout is a pipe or a file the console call is a no-op and the
    stream half is the one that matters. `errors="replace"` so a font gap never kills a session
    mid-measurement.
    """
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001 -- a redirected or absent console is fine
            pass
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:  # noqa: BLE001
                pass


def main() -> int:
    force_utf8_console()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("record", help="append one measurement row (needs the arm, or --reading)")
    r.add_argument("--csv", required=True)
    r.add_argument("--joint", required=True, choices=JOINTS)
    r.add_argument("--pose", required=True, choices=["A", "B"])
    r.add_argument("--physical-deg", type=float, default=None,
                   help="the angle you MEASURED with a protractor / phone app. Omit for the gripper "
                        "and pass --physical-mm instead (spec §4/§6)")
    r.add_argument("--physical-mm", type=float, default=None,
                   help="for the GRIPPER: the jaw opening you measured with a ruler, in mm")
    r.add_argument("--reading", type=float, default=None, help="skip reading the arm, supply it yourself")
    r.add_argument("--config", default="configs/record_omx.yaml")
    r.add_argument("--note", default="")
    r.set_defaults(func=cmd_record)

    n = sub.add_parser("session",
                       help="walk every un-recorded pose on ONE held connection (needs the arm)")
    n.add_argument("--csv", required=True)
    n.add_argument("--config", default="configs/record_omx.yaml")
    n.add_argument("--only", default="",
                   help="comma-separated joints to limit the walk to, e.g. wrist_roll,gripper")
    torque = n.add_mutually_exclusive_group()
    torque.add_argument("--lock-others", action="store_true",
                        help="RECOMMENDED when no leader is attached: release only the joint being "
                             "measured and hold every other joint rigid, so the arm cannot collapse "
                             "and the others cannot drift between poses A and B (spec §4-b)")
    torque.add_argument("--hand-pose", action="store_true",
                        help="release torque on EVERY joint at once so the whole arm can be posed by "
                             "hand. Blunter than --lock-others: the arm goes fully limp")
    n.set_defaults(func=cmd_session)

    s = sub.add_parser("solve", help="solve SCALE and OFFSET per joint (no hardware needed)")
    s.add_argument("--csv", required=True)
    s.add_argument("--date", default="YYYY-MM-DD", help="stamped into the pasteable block")
    s.add_argument("--chain", action="store_true",
                   help="subtract the upstream joints' contribution from each OFFSET, using the "
                        "state_* columns and the URDF link geometry (spec §4-a). Needed whenever "
                        "the typed angles are ABSOLUTE link orientations, which is what a "
                        "protractor gives. Leaves every SCALE unchanged")
    s.set_defaults(func=cmd_solve)

    t = sub.add_parser("selftest", help="check the arithmetic against a synthetic case")
    t.set_defaults(func=cmd_selftest)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
