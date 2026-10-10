#!/usr/bin/env python
"""S7: measure the pitch-chain joints' reading->angle map with a phone inclinometer on flat link faces.

Spec: `docs/specs/S7_link_tilt_calibration.md` -- read §2 (what a reading means) before a lab day.

    true_rad = SCALE * reading + OFFSET        (shoulder_lift, elbow_flex, wrist_flex, wrist_roll)

Why it exists: the current OFFSETs came from fitting fingertip touches on the table (9/22) plus two
hand-tuned deltas, so they are only pinned down near the table. 10/08's overlay fits show the arm is
right near the table and wrong up high (home needs wrist_flex -22..-23 deg, grasp needs ~0). A phone
lying on a flat face of a link reads that face's angle against GRAVITY, which needs neither the
table nor a camera -- so every pose the arm actually visits can be measured, high ones included.

Each measurement is one angle of one face direction `d` (a unit vector in a link frame, read off the
CAD mesh -- FACES below). The model is the same FK the sim renders with (reach_logger/fk.py), so
the constants that come out make the sim's link MESHES tilt like the real ones, which is the thing a
render is judged on.

Five subcommands. Only `session` needs the arm.

    # 0. once, anywhere: build the pose plan from a recorded dataset (poses the arm really visited,
    #    and the recorded path to each -- so the session never invents a trajectory)
    python scripts/measure_link_tilt.py plan \\
        --dataset-root data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_normal_A1 \\
        --out configs/s7_link_tilt_plan_normalA1.json

    # 1. lab day, arm at the plan's start pose first (the session refuses otherwise):
    uv run python scripts/read_joint_pose.py --goto <the six numbers `plan` printed>
    uv run python scripts/measure_link_tilt.py session \\
        --plan configs/s7_link_tilt_plan_normalA1.json --csv calibration/<date>_link_tilt.csv

    # 2. anywhere: fit, diagnose, print the block to paste into sim/joint_mapping.py
    python scripts/measure_link_tilt.py solve --csv calibration/<date>_link_tilt.csv \\
        --touch-csv calibration/2026-09-22_touch_calibration.csv

    # 3. anywhere: the faces this script knows, and the arithmetic against a synthetic arm
    python scripts/measure_link_tilt.py faces
    python scripts/measure_link_tilt.py selftest

🔴 Never edits `sim/joint_mapping.py` (same rule as S6): `solve` prints a block for a human to paste.
🔴 `shoulder_pan` is NOT measured here -- a level cannot see a rotation about the vertical. That is S8.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "sim"))

import numpy as np  # noqa: E402

import joint_mapping as JM  # noqa: E402
from reach_logger import fk  # noqa: E402

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
FIT_JOINTS = ("shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")   # pan: S8
STATE_FIELDS = tuple(f"state_{j}" for j in JOINTS)
FIELDS = ("pose_id", "source", "face", "kind", "reading_deg", "reading_rev_deg", "lean", "value_deg",
          "approach", "support", "note") + STATE_FIELDS

ARM_RISER_HEIGHT_M = 0.146   # [柏宇 measured 2026-10-07]; only used for the clearance print and --touch-csv
LEAN_ASK_DEG = 60.0          # |elevation| at or above this: ask which way the arrow end leans
REVERSAL_WARN_DEG = 2.0      # phone read forward vs turned 180 deg disagree by more -> re-seat it
GRIPPER_MIN_GOAL = 50.5      # never command the fingers past touching while replaying a recorded path


# --------------------------------------------------------------------------------------------------
# Faces. [已查證 2026-10-08] from the STL meshes the URDF draws each link with
# (~/isaaclab_volume/assets/open_manipulator_description/meshes/omx_f/, scale 0.001, visual origin 0):
# planar triangle clusters by area, normals exact to 0.00 deg. `faces --mesh-dir` re-derives them.
# ONLY `d` enters the model; the normal just says which face to put the phone on, and the two
# parallel faces of a plate give the same reading, so either side works.
# --------------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Face:
    link: int                    # fk.link_transform index: 0 base plate, 2 upper arm, 3 forearm, 5 gripper body
    d: tuple[float, float, float]
    kind: str                    # "pitch": signed angle in the arm's vertical plane; "elev": plain elevation
    where: str                   # which face, in words
    arrow: str                   # which end of the phone's long edge is "the arrow end"
    mesh: str                    # the CAD evidence


FACES: dict[str, Face] = {
    "base_x": Face(0, (1.0, 0.0, 0.0), "elev",
                   "底座板頂面（肩關節旁那塊平板，120×150 mm）", "朝前（遠離操作者）",
                   "follower_01_base.stl: normal +z, z=13 mm, 120x150 mm"),
    "base_y": Face(0, (0.0, 1.0, 0.0), "elev",
                   "底座板頂面，手機長邊改成左右方向", "朝操作者左手邊",
                   "follower_01_base.stl: normal +z, z=13 mm, 120x150 mm"),
    "upper": Face(2, (0.0, 0.0, 1.0), "pitch",
                  "上臂前面或背面（兩面平行，130×44 mm 的長平面）", "朝手肘",
                  "follower_03_middle_verticle.stl: normals -x / +x at x=-10.2 / +10.2 mm, 130x44.5 mm"),
    "forearm": Face(3, (1.0, 0.0, 0.0), "pitch",
                    "前臂頂面或底面（兩面平行，172×44 mm）", "朝手腕",
                    "follower_04_middle_horizontal.stl: normals +z / -z at z=+10.0 / -9.9 mm, 172x44.5 mm"),
    "gripper": Face(5, (1.0, 0.0, 0.0), "pitch",
                    "夾爪寬面（看得到鏤空格子的那一面：腕部舵機側面，或手指根部的實心平面）", "朝指尖",
                    "follower_06_pan_Revised.stl: normals +z / -z, 19x33 mm; finger roots "
                    "follower_07/08: normal +-z at z=+-20 mm (fingers turn about z, so the normal holds "
                    "for any gripper opening)"),
    "gripper_across": Face(5, (0.0, 1.0, 0.0), "elev",
                           "同一個夾爪寬面，手機長邊改成橫跨夾爪", "朝操作者左手邊（量測姿勢下）",
                           "same faces as `gripper`; d = link5 +y"),
}


# --------------------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------------------

def nominal_params() -> dict[str, float]:
    """The constants now in sim/joint_mapping.py, as fit parameters (degrees per unit, degrees)."""
    p = {}
    for j in FIT_JOINTS:
        p[f"s_{j}"] = math.degrees(JM.SIGN[j] * JM.SCALE_RAD_PER_UNIT[j])
        p[f"o_{j}"] = math.degrees(JM.OFFSET_RAD[j])
    p["tilt_x"] = 0.0   # base plate tilt about base +x (deg; + = left side up)
    p["tilt_y"] = 0.0   # base plate tilt about base +y (deg; + = front side DOWN, right-hand rule)
    return p


def joint_rad(state: dict[str, float], p: dict[str, float]) -> list[float]:
    """Five body joint angles (URDF order). Pan stays on joint_mapping's constants: a level cannot see
    it, and it only sets the arm plane's heading, which the `pitch` kind measures relative to."""
    pan = math.radians(math.degrees(JM.SIGN["shoulder_pan"] * JM.SCALE_RAD_PER_UNIT["shoulder_pan"])
                       * state["shoulder_pan"] + math.degrees(JM.OFFSET_RAD["shoulder_pan"]))
    rest = [math.radians(p[f"s_{j}"] * state[j] + p[f"o_{j}"]) for j in FIT_JOINTS]
    return [pan] + rest


def _tilt_R(p: dict[str, float]) -> np.ndarray:
    ax, ay = math.radians(p["tilt_x"]), math.radians(p["tilt_y"])
    rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    return ry @ rx


def predict_deg(face: Face, state: dict[str, float], p: dict[str, float]) -> float:
    """What the phone should read on `face` for this arm state under parameters `p`.

    elev : asin(d_z), arrow end up = positive, -90..90.
    pitch: atan2(d_z, d . h) with h the horizontal heading of the arm's vertical plane -- the full
           signed angle, -180..180, so a link tipped past vertical does not fold back onto one short
           of it (the 2026-09-29 asin fold, see touch_calibrate.fingertip_pitch_deg).
    """
    q = joint_rad(state, p)
    tilt = _tilt_R(p)
    d = tilt @ fk.link_transform(q, face.link)[:3, :3] @ np.asarray(face.d)
    if face.kind == "elev":
        return math.degrees(math.asin(max(-1.0, min(1.0, float(d[2])))))
    x1 = tilt @ fk.link_transform(q, 1)[:3, 0]
    h = np.array([x1[0], x1[1], 0.0])
    h /= np.linalg.norm(h)
    return math.degrees(math.atan2(float(d[2]), float(d @ h)))


def to_value(elev_deg: float, lean: str) -> float:
    """Phone elevation of the arrow end (+ = up, -90..90) and its horizontal lean ('f' toward the
    arm's front, 'b' back toward the base) -> the signed pitch-plane angle `predict_deg` returns."""
    if lean == "b":
        return (180.0 if elev_deg >= 0 else -180.0) - elev_deg
    return elev_deg


def wrap(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


# --------------------------------------------------------------------------------------------------
# Fit (Levenberg-Marquardt, numeric Jacobian; numpy only so it runs on the lab laptop as-is)
# --------------------------------------------------------------------------------------------------

def fit(rows: list[dict], free: list[str], p0: dict[str, float], iters: int = 60):
    p = dict(p0)

    def residuals(pp):
        return np.array([wrap(r["value"] - predict_deg(FACES[r["face"]], r["state"], pp)) for r in rows])

    lam = 1e-3
    res = residuals(p)
    for _ in range(iters):
        J = np.empty((len(rows), len(free)))
        for k, name in enumerate(free):
            q = dict(p)
            q[name] += 1e-4
            J[:, k] = -(residuals(q) - res) / 1e-4
        A = J.T @ J
        g = J.T @ res
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-9), g)
        trial = dict(p)
        for k, name in enumerate(free):
            trial[name] += step[k]
        tres = residuals(trial)
        if tres @ tres < res @ res:
            p, res, lam = trial, tres, lam / 3
            if np.abs(step).max() < 1e-7:
                break
        else:
            lam *= 5
    dof = max(len(rows) - len(free), 1)
    sigma2 = float(res @ res) / dof
    try:
        cov = np.linalg.inv(J.T @ J) * sigma2
        se = {name: float(math.sqrt(max(cov[k, k], 0.0))) for k, name in enumerate(free)}
    except np.linalg.LinAlgError:
        se = {name: float("nan") for name in free}
    return p, res, se


def free_params(rows: list[dict], with_scale: bool) -> list[str]:
    """Only what the rows can see: a joint with no face downstream of it stays at its constant."""
    seen_links = {FACES[r["face"]].link for r in rows}
    kinds = {r["face"] for r in rows}
    free = []
    if 0 in seen_links:
        free += ["tilt_x", "tilt_y"]
    for j, needs in (("shoulder_lift", 2), ("elbow_flex", 3), ("wrist_flex", 5)):
        if max(seen_links) >= needs:
            free.append(f"o_{j}")
            if with_scale:
                free.append(f"s_{j}")
    if "gripper_across" in kinds:
        free.append("o_wrist_roll")
        if with_scale:
            free.append("s_wrist_roll")
    return free


# --------------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    out = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if not r.get("value_deg"):
            continue
        out.append({"pose_id": r["pose_id"], "face": r["face"], "value": float(r["value_deg"]),
                    "approach": r.get("approach", ""), "support": r.get("support", ""),
                    "reading": float(r["reading_deg"]) if r.get("reading_deg") else float("nan"),
                    "reading_rev": float(r["reading_rev_deg"]) if r.get("reading_rev_deg") else float("nan"),
                    "state": {j: float(r[f"state_{j}"]) for j in JOINTS}})
    return out


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------------------------------
# plan
# --------------------------------------------------------------------------------------------------

def _fingertip_above_table_m(state: dict[str, float], p: dict[str, float]) -> float:
    import omx_constants as K  # noqa: PLC0415
    t5 = fk.link5_transform(joint_rad(state, p))
    return float((t5 @ np.array([K.GRIPPER_TIP_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0]))[2]) + ARM_RISER_HEIGHT_M


def _design_rows(state: dict[str, float]) -> list[list[float]]:
    """Linearised sensitivity of the three pitch faces to (offset, scale) of lift/elbow/wrist_flex:
    a pitch face's angle is minus the sum of the +Y joint angles upstream of it plus a constant."""
    r = [state["shoulder_lift"], state["elbow_flex"], state["wrist_flex"]]
    return [[1, r[0], 0, 0, 0, 0], [1, r[0], 1, r[1], 0, 0], [1, r[0], 1, r[1], 1, r[2]]]


def cmd_plan(args) -> int:
    import pyarrow.parquet as pq  # noqa: PLC0415

    files = sorted(glob.glob(os.path.join(args.dataset_root, "data", "chunk-*", "*.parquet")))
    if not files:
        raise SystemExit(f"no data parquet under {args.dataset_root}/data/")
    by_ep: dict[int, list[tuple[int, list[float]]]] = {}
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for e, fi, s in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            by_ep.setdefault(int(e), []).append((int(fi), [float(v) for v in s]))
    for e in by_ep:
        by_ep[e].sort()
    p = nominal_params()
    start_ep = min(by_ep)
    start = by_ep[start_ep][0][1]

    cands = []
    for e, frames in by_ep.items():
        for k in range(args.min_frame, len(frames), args.every):
            st = dict(zip(JOINTS, frames[k][1]))
            if _fingertip_above_table_m(st, p) < args.min_tip_m:
                continue
            cands.append((e, k, st))
    if not cands:
        raise SystemExit("no candidate frame clears --min-tip-m")

    # Greedy D-optimal pick: each added pose maximises log det of the information matrix of the
    # six pitch-chain parameters. That favours spread-out AND decorrelated lift/elbow/wrist readings,
    # which is what separates a scale error from an offset error and one joint from the next.
    # Repeating a pose also raises det, so without a spacing rule the pick stacks near-copies of the
    # same frame (seen 2026-10-08: three picks 3 frames apart). A new pose must differ from every
    # chosen one by --min-spacing-units in at least one pitch joint.
    def spaced(st):
        return all(max(abs(st[j] - cands[c][2][j]) for j in ("shoulder_lift", "elbow_flex", "wrist_flex"))
                   >= args.min_spacing_units for c in chosen)

    chosen, info = [], np.eye(6) * 1e-6
    for _ in range(args.n_poses):
        best, best_val = None, -np.inf
        for i, (_, _, st) in enumerate(cands):
            if i in chosen or not spaced(st):
                continue
            X = np.array(_design_rows(st), float)
            val = np.linalg.slogdet(info + X.T @ X)[1]
            if val > best_val:
                best, best_val = i, val
        if best is None:
            print(f"only {len(chosen)} poses are --min-spacing-units apart; stopping there")
            break
        chosen.append(best)
        X = np.array(_design_rows(cands[best][2]), float)
        info += X.T @ X
    chosen.sort(key=lambda i: (cands[i][0], cands[i][1]))

    poses = []
    for n, i in enumerate(chosen):
        e, k, st = cands[i]
        path = [by_ep[e][m][1] for m in range(0, k + 1, args.path_every)]
        if path[-1] != by_ep[e][k][1]:
            path.append(by_ep[e][k][1])
        poses.append({"id": f"P{n:02d}", "source": f"ep{e} f{by_ep[e][k][0]}", "path": path,
                      "faces": ["upper", "forearm", "gripper"],
                      "predicted": {fc: round(predict_deg(FACES[fc], st, p), 1)
                                    for fc in ("upper", "forearm", "gripper")},
                      "tip_above_table_cm": round(100 * _fingertip_above_table_m(st, p), 1)})
    # Two poses get the extra checks: the highest one (most gravity torque on the pitch chain, and
    # where the overlay fit says the error lives) and the lowest one.
    # The +-bump must keep the fingertip clear of the table too (FK, current constants).
    def bump_clear(n):
        st = dict(zip(JOINTS, poses[n]["path"][-1]))
        return all(_fingertip_above_table_m(dict(st, shoulder_lift=st["shoulder_lift"] + s * args.bump_units,
                                                 elbow_flex=st["elbow_flex"] + s * args.bump_units), p)
                   >= args.min_tip_m for s in (-1, 1))
    ok = [n for n in range(len(poses)) if bump_clear(n)] or list(range(len(poses)))
    hi = max(ok, key=lambda n: poses[n]["tip_above_table_cm"])
    lo = min(ok, key=lambda n: poses[n]["tip_above_table_cm"])
    for n in {hi, lo}:
        poses[n]["hysteresis"] = True
        poses[n]["load_test"] = True

    # wrist_roll: a level only sees roll when the gripper axis is near horizontal (with the gripper
    # pointing down, rolling it just changes heading). Start pose with wrist_flex moved so FK says
    # link5 +x is horizontal -- a move UP and forward at the arm's highest point, away from the table.
    st0 = dict(zip(JOINTS, start))
    best_wf, best_el = st0["wrist_flex"], 1e9
    for wf in np.arange(st0["wrist_flex"] - args.roll_max_wrist_units, st0["wrist_flex"] + args.roll_max_wrist_units + 0.5, 0.5):
        st = dict(st0, wrist_flex=float(wf))
        el = abs(predict_deg(FACES["gripper"], st, p))
        if el < best_el and _fingertip_above_table_m(st, p) >= args.min_tip_m + 0.04:
            best_wf, best_el = float(wf), el
    roll_targets = [round(st0["wrist_roll"] + du, 1) for du in
                    np.linspace(-args.roll_span_units, args.roll_span_units, args.roll_steps)]
    plan = {"what": "S7 link-tilt pose plan (docs/specs/S7_link_tilt_calibration.md)",
            "dataset": os.path.basename(os.path.normpath(args.dataset_root)),
            "start": [round(v, 2) for v in start],
            "constants_at_plan_time": {k: round(v, 4) for k, v in p.items()},
            "poses": poses,
            "roll": {"wrist_flex": round(best_wf, 1), "predicted_gripper_pitch": round(predict_deg(
                FACES["gripper"], dict(st0, wrist_flex=best_wf), p), 1), "targets": roll_targets}}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(plan, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(poses)} poses -> {args.out}")
    print(f"start pose (episode {start_ep} frame 0): {' '.join(f'{v:.2f}' for v in start)}")
    print(f"  uv run python scripts/read_joint_pose.py --goto {' '.join(f'{v:.2f}' for v in start)}")
    for ps in poses:
        st = dict(zip(JOINTS, ps["path"][-1]))
        extra = " +hysteresis +load" if ps.get("hysteresis") else ""
        print(f"  {ps['id']} {ps['source']:>10}  lift {st['shoulder_lift']:6.1f} elbow {st['elbow_flex']:6.1f} "
              f"wrist {st['wrist_flex']:6.1f}  tip {ps['tip_above_table_cm']:5.1f} cm  "
              f"predicted {ps['predicted']}{extra}")
    print(f"  roll: wrist_flex -> {best_wf:.1f} (gripper pitch predicted {plan['roll']['predicted_gripper_pitch']}), "
          f"roll targets {roll_targets}")
    return 0


# --------------------------------------------------------------------------------------------------
# session (the only part that touches hardware)
# --------------------------------------------------------------------------------------------------

class Arm:
    """Same bus pattern as scripts/read_joint_pose.py --goto (proven on this arm): no configure(),
    Goal_Position = Present before torque on, linear interpolation at 50 Hz, torque LEFT ON at exit."""

    def __init__(self, port: str, robot_id: str):
        from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig  # noqa: PLC0415

        self.robot = OmxFollower(OmxFollowerConfig(port=port, id=robot_id, calibration_dir=Path(_REPO) / "calibration"))
        self.bus = self.robot.bus

    def __enter__(self):
        self.bus.connect()
        pos = self.read()
        self.bus.sync_write("Goal_Position", pos)
        self.bus.enable_torque(num_retry=2)
        return self

    def __exit__(self, *exc):
        self.bus.disconnect(False)   # torque stays ON: the arm keeps holding wherever it is
        return False

    def read(self) -> dict[str, float]:
        return self.bus.sync_read("Present_Position", num_retry=2)

    def read_avg(self, n: int = 5) -> dict[str, float]:
        acc = {j: 0.0 for j in JOINTS}
        for _ in range(n):
            for j, v in self.read().items():
                acc[j] += v / n
            time.sleep(0.04)
        return acc

    def move_to(self, target: dict[str, float], seconds: float) -> None:
        start = self.read()
        steps = max(int(seconds * 50), 1)
        for i in range(1, steps + 1):
            t = i / steps
            self.bus.sync_write("Goal_Position", {k: start[k] * (1 - t) + target[k] * t for k in target})
            time.sleep(1 / 50)

    def follow(self, path: list[list[float]], seconds_per_point: float) -> None:
        for pt in path:
            target = dict(zip(JOINTS, pt))
            # The recording closed on a cup (gripper ~47-50); with no cup that goal would squeeze the
            # fingers into each other. They touch at 50.21 [已查證 2026-09-21, joint_mapping.py].
            target["gripper"] = max(target["gripper"], GRIPPER_MIN_GOAL)
            self.move_to(target, seconds_per_point)


def _ask(prompt: str) -> str:
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        print()
        return "q"


def _ask_float(prompt: str):
    while True:
        raw = _ask(prompt)
        if raw.lower() == "q":
            return "q"
        if raw == "":
            return None
        try:
            return float(raw)
        except ValueError:
            print(f"      {raw!r} 不是數字，再輸入一次（空白=略過，q=結束）")


def measure_face(csv_path: Path, pose_id: str, source: str, face_name: str, state: dict[str, float],
                 approach: str, support: str, predicted: float | None):
    """Prompt for one face, both phone orientations; write the row. Returns 'q' to stop."""
    face = FACES[face_name]
    hint = f"（現行常數預測約 {predicted:+.0f}°）" if predicted is not None else ""
    print(f"    ▸ {face_name}: {face.where}")
    print(f"      手機長邊沿著它，箭頭端＝{face.arrow}。讀長邊相對水平的角度，箭頭端較高為正 {hint}")
    a = _ask_float("      讀數（度）: ")
    if a == "q":
        return "q"
    if a is None:
        print("      略過")
        return None
    b = _ask_float("      手機在同一面上轉 180° 再讀一次（仍以箭頭端為準；空白=沒量）: ")
    if b == "q":
        return "q"
    elev = a if b is None else (a + b) / 2.0
    if b is not None and abs(a - b) > REVERSAL_WARN_DEG:
        print(f"      ⚠️  兩次差 {abs(a - b):.1f}° > {REVERSAL_WARN_DEG}°：手機沒貼平，或 App 本身有偏差。"
              "建議重量這一面（空白跳過後再選一次），這筆仍會記下。")
    lean = ""
    if face.kind == "pitch":
        if abs(elev) >= LEAN_ASK_DEG:
            while lean not in ("f", "b"):
                lean = _ask("      箭頭端在水平方向上是朝手臂前方（f）還是後方、往底座那邊（b）？ ").lower()
                if lean == "q":
                    return "q"
        else:
            lean = "f" if predicted is None or abs(predicted) <= 90 else "b"
    value = to_value(elev, lean) if face.kind == "pitch" else elev
    if predicted is not None and abs(wrap(value - predicted)) > 35:
        print(f"      ⚠️  和預測差 {wrap(value - predicted):+.0f}°。常數最多錯 ~25°，差這麼多通常是 App 顯示的是"
              "「與鉛垂的夾角」（要記 90 − 讀數）或箭頭端看反了。")
        if _ask("      確定要記下嗎？(y/N) ").lower() != "y":
            return None
    append_row(csv_path, {"pose_id": pose_id, "source": source, "face": face_name, "kind": face.kind,
                          "reading_deg": a, "reading_rev_deg": "" if b is None else b, "lean": lean,
                          "value_deg": round(value, 2), "approach": approach, "support": support, "note": "",
                          **{f"state_{j}": round(state[j], 3) for j in JOINTS}})
    print(f"      記下 {value:+.1f}°")
    return None


def cmd_session(args) -> int:
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    csv_path = Path(args.csv)
    done = set()
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open(encoding="utf-8")):
            done.add((r["pose_id"], r["face"], r["approach"], r["support"]))
    p = nominal_params()
    start = dict(zip(JOINTS, plan["start"]))
    sp = args.seconds_per_point
    print("=" * 76)
    print("S7 session. 手臂扭力全程開著、自己撐住；每次移動前都會停下來等你按 Enter。")
    print("移動走的是錄影時真實走過的路徑（放慢），所以桌上的杯子要拿走，收納盒可以留在原位。")
    print("讀數時手機用手托著、輕貼在面上，不要讓手機的重量壓在手臂上（負載測試那幾筆例外）。")
    print("=" * 76)
    with Arm(args.port, args.id) as arm:
        now = arm.read()
        far = {j: round(now[j] - start[j], 1) for j in JOINTS if abs(now[j] - start[j]) > args.max_start_jump}
        if far:
            print(f"🔴 手臂不在計畫的起始姿勢（差 {far}）。先執行：")
            print("   uv run python scripts/read_joint_pose.py --goto " + " ".join(f"{v:.2f}" for v in plan["start"]))
            return 1

        def faces_at(pose_id, source, faces, state, approach, support, preds):
            for fc in faces:
                if (pose_id, fc, approach, support) in done:
                    continue
                if measure_face(csv_path, pose_id, source, fc, state, approach, support,
                                preds.get(fc) if preds else predict_deg(FACES[fc], state, p)) == "q":
                    return "q"
            return None

        print("\n--- B0 底座水平（不動手臂）")
        st = arm.read_avg()
        if faces_at("B0", "start", ["base_x", "base_y"], st, "", "", None) == "q":
            return 0

        for ps in plan["poses"]:
            if all((ps["id"], fc, "traj", "held") in done for fc in ps["faces"]) and not ps.get("hysteresis"):
                continue
            print(f"\n--- {ps['id']}  ({ps['source']}，指尖約在桌面上 {ps['tip_above_table_cm']} cm)")
            go = _ask("    Enter = 沿錄影路徑移過去，s = 跳過這個姿勢，q = 結束: ").lower()
            if go == "q":
                return 0
            if go == "s":
                continue
            arm.follow(ps["path"], sp)
            time.sleep(args.settle)
            st = arm.read_avg()
            if faces_at(ps["id"], ps["source"], ps["faces"], st, "traj", "held", ps["predicted"]) == "q":
                return 0
            if ps.get("load_test") and (ps["id"], "forearm", "traj", "resting") not in done:
                print("    負載測試：這次把手機『直接放在』前臂頂面上、手放開，讓手機的重量壓在手臂上。")
                if measure_face(csv_path, ps["id"], ps["source"], "forearm", arm.read_avg(), "traj", "resting",
                                ps["predicted"]["forearm"]) == "q":
                    return 0
            if ps.get("hysteresis"):
                target = dict(zip(JOINTS, ps["path"][-1]))
                for sign, tag in ((+1, "from_plus"), (-1, "from_minus")):
                    if all((ps["id"], fc, tag, "held") in done for fc in ps["faces"]):
                        continue
                    bump = dict(target, shoulder_lift=target["shoulder_lift"] + sign * args.bump_units,
                                elbow_flex=target["elbow_flex"] + sign * args.bump_units)
                    print(f"    背隙：lift、elbow 先往 {'+' if sign > 0 else '-'}{args.bump_units} 單位再回到同一點，重量一次。")
                    if _ask("    Enter = 移動，s = 跳過: ").lower() == "s":
                        continue
                    arm.move_to(bump, 1.5)
                    arm.move_to(target, 1.5)
                    time.sleep(args.settle)
                    if faces_at(ps["id"], ps["source"], ps["faces"], arm.read_avg(), tag, "held", ps["predicted"]) == "q":
                        return 0
            print("    沿原路徑退回起始姿勢……")
            arm.follow(list(reversed(ps["path"])), sp)

        roll = plan["roll"]
        print(f"\n--- R 腕部自轉（wrist_roll）：先把 wrist_flex 轉到 {roll['wrist_flex']}，讓夾爪接近水平朝前")
        if _ask("    Enter = 移動，s = 跳過，q = 結束: ").lower() not in ("s", "q"):
            st0 = arm.read()
            arm.move_to(dict(st0, wrist_flex=roll["wrist_flex"]), 3.0)
            for k, rt in enumerate(roll["targets"]):
                pid = f"R{k:02d}"
                if (pid, "gripper_across", "traj", "held") in done:
                    continue
                arm.move_to(dict(arm.read(), wrist_roll=rt), 2.0)
                time.sleep(args.settle)
                st = arm.read_avg()
                print(f"    {pid}: wrist_roll = {st['wrist_roll']:.1f}")
                if faces_at(pid, "roll sweep", ["gripper_across", "gripper"], st, "traj", "held", None) == "q":
                    return 0
            arm.move_to(dict(arm.read(), wrist_roll=start["wrist_roll"]), 2.0)
            arm.move_to(dict(arm.read(), wrist_flex=start["wrist_flex"]), 3.0)
        print(f"\n完成。手臂停在起始姿勢、扭力開著。下一步：\n  python scripts/measure_link_tilt.py solve --csv {csv_path}")
    return 0


# --------------------------------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------------------------------

def _touch_errors(touch_csv: Path, p: dict[str, float], riser_m: float):
    import omx_constants as K  # noqa: PLC0415

    tip = np.array([K.GRIPPER_TIP_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0])
    eh, ev = [], []
    for r in csv.DictReader(touch_csv.open(encoding="utf-8")):
        st = {j: float(r[f"state_{j}"]) for j in JOINTS}
        x, y, z = (fk.link5_transform(joint_rad(st, p)) @ tip)[:3]
        eh.append(math.hypot(100 * x - float(r["target_x_cm"]), 100 * y - float(r["target_y_cm"])))
        ev.append(100 * (z + riser_m) - float(r["target_z_cm"]))
    return np.array(eh), np.array(ev)


def cmd_solve(args) -> int:
    rows_all = read_rows(Path(args.csv))
    if not rows_all:
        raise SystemExit(f"no rows with value_deg in {args.csv}")
    p0 = nominal_params()
    main = [r for r in rows_all if r["support"] in ("held", "") and r["approach"] in ("traj", "")]
    print(f"{len(rows_all)} rows; {len(main)} used for the fit (held phone, approached along the recorded path)")

    rev = [abs(r["reading"] - r["reading_rev"]) for r in rows_all if not math.isnan(r["reading_rev"])]
    if rev:
        print(f"phone turned 180 deg: |difference| median {np.median(rev):.2f} deg, max {max(rev):.2f} "
              f"(> {REVERSAL_WARN_DEG} means a badly seated phone; half the median ~ the per-reading noise floor)")

    print("\nresidual = measured - predicted, with the constants NOW in sim/joint_mapping.py (deg):")
    for fc in FACES:
        rs = [wrap(r["value"] - predict_deg(FACES[fc], r["state"], p0)) for r in main if r["face"] == fc]
        if rs:
            print(f"  {fc:15s} n={len(rs):2d}  mean {np.mean(rs):+6.2f}  min {min(rs):+6.2f}  max {max(rs):+6.2f}")

    results = {}
    for label, with_scale in (("offsets only (scales = nominal 1.80 deg/unit)", False), ("offsets + scales", True)):
        free = free_params(main, with_scale)
        p, res, se = fit(main, free, p0)
        # leave-one-pose-out: does the model predict a pose it has not seen?
        loo = []
        for pid in sorted({r["pose_id"] for r in main}):
            train = [r for r in main if r["pose_id"] != pid]
            test = [r for r in main if r["pose_id"] == pid]
            if not test or len(train) <= len(free_params(train, with_scale)):
                continue
            pp, _, _ = fit(train, free_params(train, with_scale), p0, iters=30)
            loo += [wrap(r["value"] - predict_deg(FACES[r["face"]], r["state"], pp)) for r in test]
        results[label] = (p, res, se, free, loo)
        print(f"\n=== fit: {label}")
        print(f"  RMS {math.sqrt(np.mean(res ** 2)):.2f} deg over {len(res)} rows; leave-one-pose-out RMS "
              f"{math.sqrt(np.mean(np.square(loo))) if loo else float('nan'):.2f} deg")
        for name in free:
            unit = "deg/unit" if name.startswith("s_") else "deg"
            print(f"  {name:18s} {p[name]:+9.4f} ± {se[name]:.4f} {unit}   (now {p0[name]:+9.4f}, change {p[name] - p0[name]:+.4f})")
        print("  per-face residual after the fit:")
        for fc in FACES:
            rs = [res[i] for i, r in enumerate(main) if r["face"] == fc]
            if rs:
                print(f"    {fc:15s} mean {np.mean(rs):+5.2f}  RMS {math.sqrt(np.mean(np.square(rs))):.2f}  "
                      f"max |{np.abs(rs).max():.2f}|")

    p_sc = results["offsets + scales"][0]
    p_of = results["offsets only (scales = nominal 1.80 deg/unit)"][0]
    print("\nIs it linear? (offsets+scales residual vs each joint reading; a clear slope or bow means")
    print("something a reading->angle line cannot express: link flex, CAD/real mismatch, or gear play)")
    res_sc = results["offsets + scales"][1]
    upstream = {"upper": ("shoulder_lift",), "forearm": ("shoulder_lift", "elbow_flex"),
                "gripper": ("shoulder_lift", "elbow_flex", "wrist_flex")}
    for fc, joints in upstream.items():
        idx = [i for i, r in enumerate(main) if r["face"] == fc]
        if len(idx) < 4:
            continue
        for j in joints:
            x = np.array([main[i]["state"][j] for i in idx])
            y = res_sc[idx]
            if np.ptp(x) < 1e-6:
                continue
            r = float(np.corrcoef(x, y)[0, 1])
            quad = np.polyfit(x, y, 2)[0] * (np.ptp(x) / 2) ** 2
            print(f"  {fc:8s} vs {j:13s}: r = {r:+.2f}   bow over the range {quad:+.2f} deg")

    hyst = [r for r in rows_all if r["approach"] in ("from_plus", "from_minus")]
    if hyst:
        print("\nbacklash (same pose, approached from + vs - after a lift/elbow bump):")
        for pid in sorted({r["pose_id"] for r in hyst}):
            for fc in ("upper", "forearm", "gripper"):
                a = [r["value"] for r in hyst if r["pose_id"] == pid and r["face"] == fc and r["approach"] == "from_plus"]
                b = [r["value"] for r in hyst if r["pose_id"] == pid and r["face"] == fc and r["approach"] == "from_minus"]
                if a and b:
                    print(f"  {pid} {fc:8s}: {a[0] - b[0]:+.2f} deg (+ minus -)")
    load = [r for r in rows_all if r["support"] == "resting"]
    for r in load:
        held = [x for x in main if x["pose_id"] == r["pose_id"] and x["face"] == r["face"]]
        if held:
            print(f"\nload test {r['pose_id']} {r['face']}: phone resting minus held = {r['value'] - held[0]['value']:+.2f} deg "
                  "(the link deflection the encoder does NOT see; >0.5 deg means the phone's weight matters)")

    if args.touch_csv:
        print(f"\ncross-check on the table-touch points ({args.touch_csv}, riser {args.riser_m} m). Those touches were")
        print("taken with torque OFF and the arm hand-supported, S7 with torque ON -- a gap here can be load, not a")
        print("wrong fit (eval_joint_calibration.py docstring). Score grasp/release too: --out, then")
        print("  python3 scripts/eval_joint_calibration.py --candidate-json <that json>")
        for label, pp in (("now", p0), ("offsets only", p_of), ("offsets+scales", p_sc)):
            eh, ev = _touch_errors(Path(args.touch_csv), pp, args.riser_m)
            print(f"  {label:15s} horizontal median {np.median(eh):5.2f} cm   vertical median {np.median(ev):+5.2f} cm   "
                  f"3D RMS {math.sqrt(np.mean(eh ** 2 + ev ** 2)):5.2f} cm")

    tilt = (p_sc.get("tilt_x", 0.0), p_sc.get("tilt_y", 0.0))
    if max(abs(tilt[0]), abs(tilt[1])) > 0.5:
        print(f"\n⚠️  base plate tilt ({tilt[0]:+.2f}, {tilt[1]:+.2f}) deg > 0.5: the sim assumes a level base. Shim "
              "the riser level and re-measure B0 rather than carry a tilt into the constants.")

    if args.out:
        sets = {}
        for label, pp in (("S7 offsets only", p_of), ("S7 offsets+scales", p_sc)):
            scale, offset = dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD)
            for j in FIT_JOINTS:
                scale[j] = math.radians(pp[f"s_{j}"]) / JM.SIGN[j]
                offset[j] = math.radians(pp[f"o_{j}"])
            sets[label] = {"scale": scale, "offset": offset, "tilt_deg": [pp.get("tilt_x", 0.0), pp.get("tilt_y", 0.0)],
                           "source": Path(args.csv).name}
        Path(args.out).write_text(json.dumps(sets, indent=1), encoding="utf-8")
        print(f"\nwrote {args.out} (both fits; shoulder_pan left as now -- S8)")

    print("\n# ---- paste into sim/joint_mapping.py (pick ONE fit; S7 says how) ----")
    for label, pp in (("offsets only", p_of), ("offsets + scales", p_sc)):
        print(f"# [S7 {args.date}] {label}, from {Path(args.csv).name}")
        for j in FIT_JOINTS:
            if f"o_{j}" in results["offsets + scales"][3] or f"o_{j}" in results[
                    "offsets only (scales = nominal 1.80 deg/unit)"][3]:
                print(f'#   "{j}": SCALE {math.radians(pp[f"s_{j}"]) / JM.SIGN[j]:.8f}   OFFSET {math.radians(pp[f"o_{j}"]):.8f}')
    return 0


# --------------------------------------------------------------------------------------------------
# faces / selftest
# --------------------------------------------------------------------------------------------------

def cmd_faces(args) -> int:
    for name, f in FACES.items():
        print(f"{name:15s} link{f.link} d={f.d} {f.kind:5s}  {f.where}；箭頭端＝{f.arrow}")
        print(f"{'':15s} CAD: {f.mesh}")
    if args.mesh_dir:
        print("\nre-derived from the meshes (largest planar faces with a normal in the link's x-z plane):")
        for mesh in ("follower_01_base.stl", "follower_03_middle_verticle.stl",
                     "follower_04_middle_horizontal.stl", "follower_06_pan_Revised.stl"):
            for area, n, ext in planar_faces(Path(args.mesh_dir) / mesh)[:4]:
                print(f"  {mesh:36s} area {area:6.0f} mm2  normal ({n[0]:+.3f},{n[1]:+.3f},{n[2]:+.3f})  extent {ext}")
    return 0


def planar_faces(stl: Path):
    """Largest planar facets of a binary STL, grouped by normal (1 deg) and plane offset (0.5 mm)."""
    b = stl.read_bytes()
    n_tri = int(np.frombuffer(b, "<u4", 1, 80)[0])
    v = np.frombuffer(b, np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), n_tri, 84)["v"].astype(float)
    c = np.cross(v[:, 1] - v[:, 0], v[:, 2] - v[:, 0])
    area = np.linalg.norm(c, axis=1) / 2
    ok = area > 1e-9
    n, area, v = c[ok] / (2 * area[ok, None]), area[ok], v[ok]
    d = (n * v.mean(1)).sum(1)
    groups: dict[tuple, list[int]] = {}
    for i in range(len(n)):
        groups.setdefault(tuple(np.round(n[i] * 60).astype(int)) + (int(round(d[i] / 0.5)),), []).append(i)
    out = []
    for idx in groups.values():
        a = area[idx].sum()
        nn = (n[idx] * area[idx, None]).sum(0)
        nn /= np.linalg.norm(nn)
        if abs(nn[1]) > 0.2:
            continue
        pts = v[idx].reshape(-1, 3)
        out.append((a, nn, tuple(np.round(np.ptp(pts, 0), 1))))
    return sorted(out, key=lambda t: -t[0])


def synthetic_rows(rng, truth: dict[str, float], noise_deg: float) -> list[dict]:
    """Rows a phone would give on an arm whose real constants are `truth`."""
    rows = []
    for k in range(14):
        st = {"shoulder_pan": rng.uniform(-30, 30), "shoulder_lift": rng.uniform(-62, 35),
              "elbow_flex": rng.uniform(-30, 55), "wrist_flex": rng.uniform(-55, 0),
              "wrist_roll": rng.uniform(-5, 5), "gripper": 55.0}
        for fc in ("upper", "forearm", "gripper"):
            rows.append({"pose_id": f"P{k}", "face": fc, "state": st, "approach": "traj", "support": "held",
                         "value": predict_deg(FACES[fc], st, truth) + rng.normal(0, noise_deg)})
    st = {"shoulder_pan": 0.0, "shoulder_lift": -62.0, "elbow_flex": 55.0, "wrist_flex": 0.0, "wrist_roll": 0.0, "gripper": 55.0}
    for fc in ("base_x", "base_y"):
        rows.append({"pose_id": "B0", "face": fc, "state": st, "approach": "", "support": "",
                     "value": predict_deg(FACES[fc], st, truth) + rng.normal(0, noise_deg)})
    for k, rl in enumerate(np.linspace(-30, 30, 5)):
        st = {"shoulder_pan": 0.0, "shoulder_lift": -62.0, "elbow_flex": 55.0, "wrist_flex": -38.0,
              "wrist_roll": float(rl), "gripper": 55.0}
        for fc in ("gripper_across", "gripper"):
            rows.append({"pose_id": f"R{k}", "face": fc, "state": st, "approach": "traj", "support": "held",
                         "value": predict_deg(FACES[fc], st, truth) + rng.normal(0, noise_deg)})
    return rows


def run_selftest(noise_deg: float = 0.3, seed: int = 0):
    rng = np.random.default_rng(seed)
    p0 = nominal_params()
    truth = dict(p0)
    truth.update({"o_shoulder_lift": p0["o_shoulder_lift"] + 4.0, "o_elbow_flex": p0["o_elbow_flex"] - 7.0,
                  "o_wrist_flex": p0["o_wrist_flex"] - 22.0, "o_wrist_roll": p0["o_wrist_roll"] + 3.0,
                  "s_shoulder_lift": p0["s_shoulder_lift"] * 1.03, "s_elbow_flex": p0["s_elbow_flex"] * 0.98,
                  "tilt_x": 0.4, "tilt_y": -0.3})
    rows = synthetic_rows(rng, truth, noise_deg)
    p, res, _ = fit(rows, free_params(rows, True), p0)
    err = {k: p[k] - truth[k] for k in free_params(rows, True)}
    return err, float(math.sqrt(np.mean(res ** 2)))


def cmd_selftest(_args) -> int:
    err, rms = run_selftest()
    for k, e in err.items():
        print(f"  {k:18s} recovered error {e:+.4f}")
    worst_off = max(abs(e) for k, e in err.items() if not k.startswith("s_"))
    worst_sc = max(abs(e) for k, e in err.items() if k.startswith("s_"))
    ok = worst_off < 0.6 and worst_sc < 0.01 and rms < 0.5
    ok &= to_value(-80, "b") == -100 and to_value(70, "b") == 110 and to_value(-30, "f") == -30
    print(f"fit RMS {rms:.2f} deg (noise 0.3); worst offset error {worst_off:.3f} deg, worst scale error {worst_sc:.4f} deg/unit")
    print("✅ solver recovers a known arm" if ok else "🔴 solver is WRONG")
    return 0 if ok else 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415
            ctypes.windll.kernel32.SetConsoleOutputCP(65001)
            ctypes.windll.kernel32.SetConsoleCP(65001)
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    pl = sub.add_parser("plan", help="pick poses from a recorded dataset (no hardware)")
    pl.add_argument("--dataset-root", required=True)
    pl.add_argument("--out", required=True)
    pl.add_argument("--n-poses", type=int, default=10)
    pl.add_argument("--every", type=int, default=3, help="candidate frame stride")
    pl.add_argument("--min-frame", type=int, default=0)
    pl.add_argument("--path-every", type=int, default=3, help="keep every Nth recorded frame of the path")
    pl.add_argument("--min-tip-m", type=float, default=0.05, help="skip frames whose fingertip is lower above the table (FK)")
    pl.add_argument("--min-spacing-units", type=float, default=10.0,
                    help="a new pose must differ from each chosen one by this much in lift, elbow or wrist_flex")
    pl.add_argument("--bump-units", type=float, default=4.0, help="must match session --bump-units")
    pl.add_argument("--roll-span-units", type=float, default=30.0, help="wrist_roll sweep +- (units; 30 = +-54 deg)")
    pl.add_argument("--roll-steps", type=int, default=5)
    pl.add_argument("--roll-max-wrist-units", type=float, default=40.0)
    pl.set_defaults(func=cmd_plan)

    se = sub.add_parser("session", help="move through the plan and record phone readings (needs the arm)")
    se.add_argument("--plan", required=True)
    se.add_argument("--csv", required=True)
    se.add_argument("--port", default="COM8")
    se.add_argument("--id", default="2026-09-18_omx_follower")
    se.add_argument("--seconds-per-point", type=float, default=0.4,
                    help="per kept path point; plan keeps every 3rd frame of 15 fps = 0.2 s, so 0.4 is 2x slower than recorded")
    se.add_argument("--settle", type=float, default=1.0)
    se.add_argument("--bump-units", type=float, default=4.0)
    se.add_argument("--max-start-jump", type=float, default=8.0)
    se.set_defaults(func=cmd_session)

    so = sub.add_parser("solve", help="fit and diagnose (no hardware)")
    so.add_argument("--csv", required=True)
    so.add_argument("--touch-csv", default=None)
    so.add_argument("--riser-m", type=float, default=ARM_RISER_HEIGHT_M)
    so.add_argument("--date", default="YYYY-MM-DD")
    so.add_argument("--out", default=None, help="JSON of both fits, for eval_joint_calibration.py --candidate-json")
    so.set_defaults(func=cmd_solve)

    fa = sub.add_parser("faces", help="list the measurement faces")
    fa.add_argument("--mesh-dir", default=None, help="re-derive from the STL meshes")
    fa.set_defaults(func=cmd_faces)

    st = sub.add_parser("selftest", help="recover a synthetic arm's constants")
    st.set_defaults(func=cmd_selftest)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
