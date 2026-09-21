#!/usr/bin/env python
"""S6: solve each joint's reading->radians affine map from two physically unambiguous poses.

Spec: `docs/specs/S6_joint_zero_calibration.md`. Read that first -- it says which two poses to put
each joint in and how to verify them. This script is the arithmetic half and the recording helper.

    true_rad = SCALE * reading + OFFSET

Two poses per joint give two equations, so BOTH unknowns come out. One pose only solves OFFSET and
only if SCALE is already known -- which is exactly the assumption that turned out to be wrong for
the gripper, where the recorded channel spans ~10 units across a visibly full open/close.

Three subcommands, and only `record` needs the arm:

    # 1. on the machine wired to the arm: append a measurement row
    uv run python scripts/measure_joint_zeros.py record \\
        --csv calibration/2026-09-21_joint_zeros.csv \\
        --joint shoulder_lift --pose A --physical-deg 90.0 --note "upper arm vertical"

    # 2. anywhere, no hardware: solve and print the constants to paste
    uv run python scripts/measure_joint_zeros.py solve --csv calibration/2026-09-21_joint_zeros.csv

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
FIELDS = ("joint", "pose", "physical_deg", "physical_mm", "reading", "note")
MIN_READING_SPAN = 40.0          # spec §7; below this the division amplifies measurement error
DEG2RAD = math.pi / 180.0


def cmd_record(args: argparse.Namespace) -> int:
    reading = args.reading
    if reading is None:
        try:
            reading = read_from_arm(args.config, args.joint)
        except Exception as e:  # noqa: BLE001
            print(f"could not read the arm ({type(e).__name__}: {e}).")
            print("Pass --reading explicitly, or fill the CSV by hand -- see the spec's §6.")
            return 1
    path = Path(args.csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({"joint": args.joint, "pose": args.pose,
                    "physical_deg": "" if args.physical_deg is None else args.physical_deg,
                    "physical_mm": "" if args.physical_mm is None else args.physical_mm,
                    "reading": reading, "note": args.note})
    print(f"appended {args.joint} pose {args.pose}: physical={args.physical_deg} reading={reading}")
    return 0


def read_from_arm(config_path: str, joint: str) -> float:
    """Current value of one joint, straight off the follower, via lerobot."""
    from lerobot.robots.utils import make_robot_from_config  # noqa: PLC0415
    from lerobot.configs.parser import load_config  # noqa: PLC0415

    cfg = load_config(config_path)
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    try:
        obs = robot.get_observation()
        key = f"{joint}.pos"
        if key not in obs:
            raise KeyError(f"{key!r} not in the robot's observation ({sorted(obs)})")
        return float(obs[key])
    finally:
        robot.disconnect()


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
    rows = list(csv.DictReader(Path(args.csv).open(encoding="utf-8")))
    by_joint: dict[str, list[dict]] = {}
    for r in rows:
        by_joint.setdefault(r["joint"], []).append(r)

    print(f"{'joint':<15}{'scale(rad/unit)':>18}{'vs pi/180':>12}{'offset(deg)':>14}{'span':>8}  note")
    results, problems = {}, []
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
            print(f"{j:<15}{'(mm, not rad)':>18}{'':>12}{'':>14}{abs(dr):>8}"
                  f"  jaw = {mm_per_unit:.3f} mm/unit + {mm_at_zero:.1f} mm")
            problems.append("gripper: solved as reading->mm. Converting that to gripper_joint_1 "
                            "radians needs the finger linkage geometry, which is NOT measured -- "
                            "see the spec's §4 and gap 2's amplitude residual in sim/README.md")
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

    if problems:
        print("\n⚠️ unresolved:")
        for p in problems:
            print(f"   {p}")

    if results:
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


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(required=True)

    r = sub.add_parser("record", help="append one measurement row (needs the arm, or --reading)")
    r.add_argument("--csv", required=True)
    r.add_argument("--joint", required=True, choices=JOINTS)
    r.add_argument("--pose", required=True, choices=["A", "B"])
    r.add_argument("--physical-deg", type=float, default=None,
                   help="the angle you MEASURED with a protractor / phone app. Omit for the gripper "
                        "and record the jaw opening in --note instead (spec §4)")
    r.add_argument("--physical-mm", type=float, default=None,
                   help="for the GRIPPER: the jaw opening you measured with a ruler, in mm")
    r.add_argument("--reading", type=float, default=None, help="skip reading the arm, supply it yourself")
    r.add_argument("--config", default="configs/record_omx.yaml")
    r.add_argument("--note", default="")
    r.set_defaults(func=cmd_record)

    s = sub.add_parser("solve", help="solve SCALE and OFFSET per joint (no hardware needed)")
    s.add_argument("--csv", required=True)
    s.add_argument("--date", default="YYYY-MM-DD", help="stamped into the pasteable block")
    s.set_defaults(func=cmd_solve)

    t = sub.add_parser("selftest", help="check the arithmetic against a synthetic case")
    t.set_defaults(func=cmd_selftest)

    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
