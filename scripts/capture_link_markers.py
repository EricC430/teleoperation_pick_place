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

walk (2026-10-14, replaces `session` for this job): `session`'s plan came from S7's pose picker, which maximises
pitch-chain information for the PHONE measurement -- it knows nothing about which link faces the camera. On site
P00/P01/P21/P22/P29 were bin-transport poses (gripper closed, pan -48..-59, markers turned away) and ~10 were within
a few units of HOME. `walk` instead walks each chosen episode's recorded APPROACH (frame 0 -> the frame the gripper
starts closing, so never the bin), stops every --stop-every path points once the arm is --min-from-home away from
HOME and --min-spacing from the previous stop, photographs, and reports which markers it saw. Then it retraces to
HOME. Episodes are picked to spread the grasp poses (farthest-point in pan/lift/elbow/wrist_flex at the grasp frame).

    uv run python scripts/capture_link_markers.py walk --csv calibration/2026-10-14_link_markers.csv --dry-run
    uv run python scripts/capture_link_markers.py walk --csv calibration/2026-10-14_link_markers.csv

Stop ids are E<ep>_F<frame>; rerunning resumes, --retake works the same way. Same CSV as `session`, so the stills
already taken stay in one index.

scan (2026-10-14): the walk saw the gripper marker (ID6) in only 24 of 101 stills -- the gripper face is turned away
from the camera most of the approach, and ID6 is what carries the chain out to the fingertip. scan walks each episode
to its LAST stop whose fingertip is still --min-tip-cm above the table (nominal model; margin for the sweep and for a
model that is 1-2 cm off), then sweeps wrist_roll x wrist_flex around that pose (serpentine), photographing each,
and retraces. Grasp-region poses, not "in front of the camera": that is where the accuracy matters (link play and sag
depend on the pose), and it stays inside the recorded pan range (the camera now sits left-REAR of the pan axis).
Ids S<ep>_F<frame>_r<droll>_f<dflex>.

    uv run python scripts/capture_link_markers.py scan --csv calibration/2026-10-14_link_markers.csv --dry-run
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


PAPER_IDS = (0, 2, 3, 4, 7)   # calibration/2026-10-14_aruco_points.csv
ARM_IDS = (1, 5, 6)           # upper arm, forearm, gripper servo body


def _load_episodes(root: str) -> dict[int, list[tuple[int, list[float]]]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    by_ep: dict[int, list[tuple[int, list[float]]]] = {}
    for f in sorted(Path(root, "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for e, fi, st in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            by_ep.setdefault(int(e), []).append((int(fi), [float(v) for v in st]))
    for e in by_ep:
        by_ep[e].sort()
    return by_ep


def _grasp_frame(frames: list[tuple[int, list[float]]], drop: float = 3.0) -> int:
    """Index of the first frame where the gripper has closed `drop` below its open value (the grasp)."""
    g = [st[5] for _, st in frames]
    opened = max(g[: max(30, len(g) // 4)])
    for k, v in enumerate(g):
        if k > 15 and v < opened - drop:
            return k
    return len(g) - 1


def plan_walk(by_ep, home: list[float], n_episodes: int, stop_every: int, path_every: int,
              min_from_home: float, min_spacing: float, episodes: list[int] | None, exclude: set[int] = frozenset()):
    grasp = {e: _grasp_frame(fr) for e, fr in by_ep.items()}
    # No grasp found -> _grasp_frame returns the last frame, back at HOME (seen: ep46). Not a walk.
    by_ep = {e: fr for e, fr in by_ep.items() if e not in exclude
             and max(abs(fr[grasp[e]][1][i] - home[i]) for i in range(5)) >= min_from_home}
    if episodes is None:
        # Farthest-point pick over the grasp pose (pan, lift, elbow, wrist_flex): spreads the walks over the
        # placements instead of stacking near-identical ones.
        keys = sorted(by_ep)
        X = {e: by_ep[e][grasp[e]][1][:4] for e in keys}
        cen = [sum(X[e][i] for e in keys) / len(keys) for i in range(4)]
        dist = lambda a, b: max(abs(a[i] - b[i]) for i in range(4))  # noqa: E731
        episodes = [max(keys, key=lambda e: dist(X[e], cen))]
        while len(episodes) < min(n_episodes, len(keys)):
            episodes.append(max((e for e in keys if e not in episodes),
                                key=lambda e: min(dist(X[e], X[c]) for c in episodes)))
        episodes.sort()
    walks = []
    for e in episodes:
        fr = by_ep[e][: grasp[e] + 1]
        idx = list(range(0, len(fr), path_every))
        if idx[-1] != len(fr) - 1:
            idx.append(len(fr) - 1)
        stops, last = [], None
        for n, k in enumerate(idx):
            st = fr[k][1]
            if n % stop_every and k != idx[-1]:
                continue
            if max(abs(st[i] - home[i]) for i in range(5)) < min_from_home:
                continue
            if last is not None and max(abs(st[i] - last[i]) for i in range(5)) < min_spacing and k != idx[-1]:
                continue
            stops.append(n)
            last = st
        walks.append({"episode": e, "path": [fr[k][1] for k in idx], "frames": [fr[k][0] for k in idx],
                      "stops": stops})
    return walks


def _detect(img: Path) -> list[int]:
    import cv2  # noqa: PLC0415

    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                  cv2.aruco.DetectorParameters())
    _, ids, _ = det.detectMarkers(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))
    return sorted(int(i) for i in ids.ravel()) if ids is not None else []


def cmd_walk(args) -> int:
    csv_path = Path(args.csv)
    log = csv_path.with_name(csv_path.stem + "_captures.csv")
    done = {r["pose_id"] for r in csv.DictReader(log.open(encoding="utf-8"))} if log.exists() else set()
    retake = {r for r in (args.retake or "").split(",") if r}
    done -= retake
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    home = plan["start"]
    walks = plan_walk(_load_episodes(args.dataset_root), home, args.n_episodes, args.stop_every, args.path_every,
                      args.min_from_home, args.min_spacing,
                      [int(e) for e in args.episodes.split(",")] if args.episodes else None,
                      {int(e) for e in args.exclude_episodes.split(",") if e})
    n_stops = sum(len(w["stops"]) for w in walks)
    secs = sum(2 * args.ease + 2 * len(w["path"]) * args.seconds_per_point + len(w["stops"]) * (args.settle + 4)
               for w in walks)
    print("=" * 76)
    print(f"walk: {len(walks)} episodes, {n_stops} stops, about {secs / 60:.0f} min of arm time "
          f"(capture counted as 4 s/stop; real time is usually 2-3x this).")
    for w in walks:
        last = w["path"][-1]
        print(f"  ep{w['episode']:02d}: {len(w['stops'])} stops, grasp at frame {w['frames'][-1]}  "
              f"pan {last[0]:.1f} lift {last[1]:.1f} elbow {last[2]:.1f} wrist_flex {last[3]:.1f}")
    print("走的是每一集錄影的『接近杯子』那段（到夾爪開始閉合為止，不會去收納桶）；桌上不要放杯子。")
    print("拍照時手和身體都離開畫面；整個過程不要碰手臂。")
    print("=" * 76)
    if args.dry_run:
        return 0
    cams = [c for c in args.cameras.split(",") if c]
    shot_dir = csv_path.parent / "shots" / csv_path.stem
    with S7.Arm(args.port, args.id) as arm:
        start = dict(zip(JOINTS, home))
        now = arm.read()
        far = {j: round(now[j] - start[j], 1) for j in JOINTS if j != "gripper" and abs(now[j] - start[j]) > args.max_start_jump}
        if far:
            print(f"🔴 手臂不在 HOME（差 {far}）。先執行：")
            print("   uv run python scripts/read_joint_pose.py --goto-home --home-episode 10")
            return 1
        for w in walks:
            ids = [f"E{w['episode']:02d}_F{w['frames'][n]:04d}" for n in w["stops"]]
            if all(i in done for i in ids):
                continue
            print(f"\n--- ep{w['episode']:02d}: {len(ids)} stops")
            go = S7._ask("    Enter = 走這一集並沿途停下來拍照，s = 跳過這一集，q = 結束: ").lower()
            if go == "q":
                return 0
            if go == "s":
                continue
            arm.follow(w["path"][:1], args.ease)
            pos = 0
            for n, pid in zip(w["stops"], ids):
                arm.follow(w["path"][pos:n + 1], args.seconds_per_point)
                pos = n + 1
                if pid in done:
                    continue
                time.sleep(args.settle)
                st = arm.read_avg(10)
                appr = "traj_retake" if pid in retake else "traj"
                S7.capture_stills(csv_path, pid, appr, st, cams)
                seen = _detect(shot_dir / f"{pid}_{appr}_front-left.png") if "front-left" in cams else []
                arm_seen = [i for i in seen if i in ARM_IDS]
                paper_seen = [i for i in seen if i in PAPER_IDS]
                flag = "" if arm_seen else "   ⚠️ 手臂上的標記一個都沒偵測到"
                print(f"      {pid}: 手臂標記 {arm_seen}，紙上標記 {paper_seen}{flag}")
            print("    沿原路徑退回 HOME……")
            arm.follow(list(reversed(w["path"][:pos])), args.seconds_per_point)
            arm.move_to(start, args.ease)
        print(f"\n完成。手臂停在 HOME、扭力開著。照片在 {shot_dir}/，索引 {log}")
    return 0


def _serpentine(rolls: list[float], flexes: list[float]) -> list[tuple[float, float]]:
    out = []
    for n, f in enumerate(flexes):
        out += [(r, f) for r in (rolls if n % 2 == 0 else list(reversed(rolls)))]
    return out


def cmd_scan(args) -> int:
    csv_path = Path(args.csv)
    log = csv_path.with_name(csv_path.stem + "_captures.csv")
    done = {r["pose_id"] for r in csv.DictReader(log.open(encoding="utf-8"))} if log.exists() else set()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    home = plan["start"]
    walks = plan_walk(_load_episodes(args.dataset_root), home, args.n_episodes, args.stop_every, args.path_every,
                      args.min_from_home, args.min_spacing,
                      [int(e) for e in args.episodes.split(",")] if args.episodes else None,
                      {int(e) for e in args.exclude_episodes.split(",") if e})
    p = S7.nominal_params()
    grid = _serpentine([float(v) for v in args.rolls.split(",")], [float(v) for v in args.flexes.split(",")])
    bases = []
    for w in walks:
        ok = [n for n in w["stops"]
              if S7._fingertip_above_table_m(dict(zip(JOINTS, w["path"][n])), p) * 100 >= args.min_tip_cm]
        if ok:
            bases.append((w, ok[-1]))
    print("=" * 76)
    print(f"scan: {len(bases)} base poses x {len(grid)} (roll, flex) = {len(bases) * len(grid)} stills; "
          f"roll {args.rolls}, flex {args.flexes} (units)")
    for w, n in bases:
        st = w["path"][n]
        tip = S7._fingertip_above_table_m(dict(zip(JOINTS, st)), p) * 100
        print(f"  ep{w['episode']:02d} f{w['frames'][n]}: fingertip ~{tip:.0f} cm above the table (nominal), "
              f"pan {st[0]:.1f} roll {st[4]:.1f} flex {st[3]:.1f}")
    print("夾爪會在每個基準姿勢上小幅轉手腕（roll）和彎手腕（flex），每個角度停下來拍一張。")
    print("第一個基準姿勢請盯著看：夾爪或紙板若會碰到桌面、紙或線材，立刻按 Ctrl+C。")
    print("=" * 76)
    if args.dry_run:
        return 0
    cams = [c for c in args.cameras.split(",") if c]
    shot_dir = csv_path.parent / "shots" / csv_path.stem
    with S7.Arm(args.port, args.id) as arm:
        start = dict(zip(JOINTS, home))
        now = arm.read()
        far = {j: round(now[j] - start[j], 1) for j in JOINTS if j != "gripper" and abs(now[j] - start[j]) > args.max_start_jump}
        if far:
            print(f"🔴 手臂不在 HOME（差 {far}）。先執行：")
            print("   uv run python scripts/read_joint_pose.py --goto-home --home-episode 10")
            return 1
        for w, n in bases:
            base = dict(zip(JOINTS, w["path"][n]))
            base["gripper"] = max(base["gripper"], S7.GRIPPER_MIN_GOAL)
            ids = [f"S{w['episode']:02d}_F{w['frames'][n]:04d}_r{dr:+.0f}_f{df:+.0f}" for dr, df in grid]
            if all(i in done for i in ids):
                continue
            print(f"\n--- ep{w['episode']:02d} f{w['frames'][n]}: {len(ids)} stills")
            go = S7._ask("    Enter = 走到基準姿勢並轉動拍照，s = 跳過，q = 結束: ").lower()
            if go == "q":
                return 0
            if go == "s":
                continue
            arm.follow(w["path"][:1], args.ease)
            arm.follow(w["path"][: n + 1], args.seconds_per_point)
            hits = 0
            for (dr, df), pid in zip(grid, ids):
                if pid in done:
                    continue
                tgt = dict(base)
                tgt["wrist_roll"] += dr
                tgt["wrist_flex"] += df
                jump = max(abs(tgt[j] - arm.goal[j]) for j in ("wrist_roll", "wrist_flex"))
                arm.move_to(tgt, max(args.step_seconds, jump / 20.0))  # <= 20 units/s
                time.sleep(args.settle)
                st = arm.read_avg(10)
                S7.capture_stills(csv_path, pid, "scan", st, cams)
                seen = _detect(shot_dir / f"{pid}_scan_front-left.png") if "front-left" in cams else []
                hits += 6 in seen
                print(f"      {pid}: 手臂標記 {[i for i in seen if i in ARM_IDS]}")
            print(f"    這個基準姿勢 ID6 偵測到 {hits}/{len(ids)} 張。回基準姿勢，沿原路徑退回 HOME……")
            arm.move_to(base, max(args.step_seconds, max(abs(base[j] - arm.goal[j]) for j in JOINTS) / 20.0))
            arm.follow(list(reversed(w["path"][: n + 1])), args.seconds_per_point)
            arm.move_to(start, args.ease)
        print(f"\n完成。手臂停在 HOME、扭力開著。照片在 {shot_dir}/，索引 {log}")
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
    walks = argparse.ArgumentParser(add_help=False)  # what walk and scan share: which walks, how to move
    walks.add_argument("--csv", required=True, help="calibration/<date>_link_markers.csv (only its name is used)")
    walks.add_argument("--plan", default="configs/link_marker_plan_farwrist.json", help="only its HOME ('start') is used")
    walks.add_argument("--dataset-root", default=".cache/lerobot/omx_pick_place_pilot_paper_cup_far_wrist_A1")
    walks.add_argument("--episodes", default="", help="comma-separated episode indices; default = --n-episodes spread ones")
    walks.add_argument("--n-episodes", type=int, default=10)
    walks.add_argument("--exclude-episodes", default="17",
                       help="never walk these (17: the 10-14 session's P14 = ep17 f237 hit an obstacle)")
    walks.add_argument("--path-every", type=int, default=3, help="keep every Nth recorded frame as a path point")
    walks.add_argument("--stop-every", type=int, default=5, help="stop every N path points (5 x 3 frames = 1 s recorded)")
    walks.add_argument("--min-from-home", type=float, default=10.0, help="no stops until some joint is this far from HOME")
    walks.add_argument("--min-spacing", type=float, default=6.0, help="units in some joint between consecutive stops")
    walks.add_argument("--cameras", default="front-left")
    walks.add_argument("--port", default="COM8")
    walks.add_argument("--id", default="2026-09-18_omx_follower")
    walks.add_argument("--seconds-per-point", type=float, default=0.2)
    walks.add_argument("--settle", type=float, default=1.5)
    walks.add_argument("--ease", type=float, default=1.5)
    walks.add_argument("--max-start-jump", type=float, default=8.0)
    walks.add_argument("--dry-run", action="store_true", help="print what would be walked; no arm")
    wk = sub.add_parser("walk", parents=[walks],
                        help="walk recorded approaches, stop and photograph along the way (needs the arm)")
    wk.add_argument("--retake", default="")
    wk.set_defaults(func=cmd_walk)
    sc = sub.add_parser("scan", parents=[walks],
                        help="at a pre-grasp pose per episode, sweep wrist roll x flex and photograph (needs the arm)")
    sc.add_argument("--min-tip-cm", type=float, default=11.0,
                    help="base pose = the episode's last walk stop with the fingertip at least this high (nominal)")
    sc.add_argument("--rolls", default="-30,-20,-10,0",
                    help="wrist_roll offsets from the base pose (units). Toward lower roll: in the 10-14 walk ID6 was "
                         "seen at roll median 3.5 (p90 26.5), missed up to 43.5, and the bases sit at roll 15-48")
    sc.add_argument("--flexes", default="-8,0,8", help="wrist_flex offsets from the base pose (units)")
    sc.add_argument("--step-seconds", type=float, default=1.0, help="move time between grid points")
    sc.set_defaults(func=cmd_scan)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
