#!/usr/bin/env python
"""Link-marker capture: park the arm on recorded poses and photograph the ArUco markers taped to its links.

Why (2026-10-14): the physics replay of far_wrist_A1 t61-t70 grasped 2/10 -- every miss had the sim gripper 1-2 cm
short of the cup, and the cup placements check out in the images. The arm model is extrapolated where the cups
are: S7 (phone, hand-touched links) and S8 (circles of 12-19 cm) never measured the far reach or the free home pose.
A marker on each link, seen by the calibrated front-left camera, measures every link's pose with nobody touching the
arm, at as many recorded poses as we like.

Setup (once):
  1. All 8 markers on the paper: take the world photo and solve the camera (a new camera era; the camera moved on
     10-13). Then move markers 1, 5, 6 to the arm, each glued flat on stiff card, taped to the LEFT side of the upper
     arm, the forearm, and the gripper servo body (not the lattice finger). They must not move during the session;
     where exactly they sit on the link is solved from the data. 0, 2, 3, 4, 7 stay on the paper.
  2. Arm at the plan's start pose, which is the closed-loop HOME (`read_joint_pose.py --goto-home --home-episode 10`).

    uv run python scripts/capture_link_markers.py session --plan configs/link_marker_plan_farwrist.json \\
        --csv calibration/<date>_link_markers.csv

Each pose: Enter -> the arm eases from HOME onto the start of the RECORDED path (each episode began a few units away
from HOME), follows it there (slowed down), settles, joint states are averaged, one front-left still (and a wrist
still) is taken, then the arm retraces the path and eases back to HOME. s = skip, q = quit;
rerunning with the same CSV skips poses already captured. --retake P09,B0 captures those again: the new
stills get a '_retake' suffix (nothing is overwritten) and their rows are appended, so a pose_id can appear more than
once in the index -- the LAST row per (pose_id, camera) is the one to use. Stills go to calibration/shots/<csv stem>/ and the index to
<csv stem>_captures.csv (same format as S7's, so scripts/click_fingertips.py and the overlay tools read it).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import measure_link_tilt as S7  # noqa: E402  (Arm with auto-reconnect, capture_stills, prompts)
from measure_link_tilt import JOINTS  # noqa: E402


def cmd_session(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    csv_path = Path(args.csv)
    log = csv_path.with_name(csv_path.stem + "_captures.csv")
    done = set()
    if log.exists():
        done = {r["pose_id"] for r in csv.DictReader(log.open(encoding="utf-8"))}
    retake = {r for r in (args.retake or "").split(",") if r}
    done -= retake
    cams = [c for c in args.cameras.split(",") if c]
    start = dict(zip(JOINTS, plan["start"]))
    print("=" * 76)
    print(f"link-marker capture: {len(plan['poses'])} poses from {plan['dataset']}, {len(done)} already captured.")
    print("移動走的是錄影時真實走過的路徑（放慢），桌上不要放杯子；手臂上的標記紙板不能碰到任何東西。")
    print("拍照時手和身體都離開畫面；整個過程不要碰手臂。")
    print("=" * 76)
    with S7.Arm(args.port, args.id) as arm:
        now = arm.read()
        far = {j: round(now[j] - start[j], 1) for j in JOINTS if j != "gripper" and abs(now[j] - start[j]) > args.max_start_jump}
        if far:
            print(f"🔴 手臂不在計畫的起始姿勢（差 {far}）。先執行：")
            print("   uv run python scripts/read_joint_pose.py --goto-home --home-episode 10")
            return 1
        if "B0" not in done:
            st = arm.read_avg(10)
            S7.capture_stills(csv_path, "B0", "start_retake" if "B0" in retake else "", st, cams)
        for ps in plan["poses"]:
            if ps["id"] in done:
                continue
            print(f"\n--- {ps['id']}  ({ps['source']})")
            go = S7._ask("    Enter = 沿錄影路徑移過去並拍照，s = 跳過，q = 結束: ").lower()
            if go == "q":
                return 0
            if go == "s":
                continue
            arm.follow(ps["path"][:1], args.ease)
            arm.follow(ps["path"], args.seconds_per_point)
            time.sleep(args.settle)
            st = arm.read_avg(10)
            S7.capture_stills(csv_path, ps["id"], "traj_retake" if ps["id"] in retake else "traj", st, cams)
            print("    沿原路徑退回起始姿勢……")
            arm.follow(list(reversed(ps["path"])), args.seconds_per_point)
            arm.move_to(start, args.ease)
        print(f"\n完成。手臂停在起始姿勢、扭力開著。照片在 {csv_path.parent / 'shots' / csv_path.stem}/，索引 {log}")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    se = sub.add_parser("session", help="move through the plan and photograph the link markers (needs the arm)")
    se.add_argument("--plan", required=True)
    se.add_argument("--csv", required=True, help="calibration/<date>_link_markers.csv (only its name is used)")
    se.add_argument("--cameras", default="front-left,wrist")
    se.add_argument("--port", default="COM8")
    se.add_argument("--id", default="2026-09-18_omx_follower")
    se.add_argument("--seconds-per-point", type=float, default=0.2,
                    help="per kept path point (every 3rd recorded frame = 0.2 s), so 0.2 is the recorded speed "
                         "(10-14: 0.3 was too slow on site)")
    se.add_argument("--settle", type=float, default=2.0)
    se.add_argument("--retake", default="", help="comma-separated pose ids (B0, P09, ...) to capture again")
    se.add_argument("--ease", type=float, default=1.5, help="seconds for HOME <-> the recorded path's first point")
    se.add_argument("--max-start-jump", type=float, default=8.0)
    se.set_defaults(func=cmd_session)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
