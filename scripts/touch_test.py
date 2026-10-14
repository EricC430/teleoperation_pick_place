#!/usr/bin/env python
"""Camera-free check of a link-marker fit: the arm hovers its closed fingertip over chosen points of the placement mat,
the operator reads off the grid where the tip really is.

Why (2026-10-15): on the 10-14 closed-loop frames every fitted model needs the arm moved ~1 cm forward to line up,
while the camera that does so would break the cup and the background. Every image check goes through a camera solved
from 5 paper markers seen almost edge-on, so this one does not use a camera at all.

How it reads:
  * the same error vector at every point  -> where the model puts the base on the mat is wrong
  * the error growing with reach          -> the arm model itself (offsets, play) is wrong
  * no error                              -> the arm is right; the closed-loop camera is what is off

Motion: the model's inverse (readings -> fingertip, with its sag) is solved for each target, gripper as close to
straight down as reachable, roll and gripper fixed (gripper closed). From HOME the arm goes to a point 6 cm above the
target, then down to --hover-mm above the paper (by the model; the arm sags a few units under position control, which the
readings record). At the hover: d / u lower / raise 3 mm, Enter = read. No hold(): it tripped shoulder_lift Overload on 10-15. The analysis uses the readings
actually present, not the solved ones, so servo tracking does not enter the result.

    uv run python scripts/touch_test.py plan                    # solve and print, no arm
    uv run python scripts/touch_test.py session --csv calibration/2026-10-15_touch_test.csv
    uv run python scripts/touch_test.py analyse --csv calibration/2026-10-15_touch_test.csv

Reading the tip: look straight down at the closed fingertips (both tips meet at one point), read the grid under that
point in cm with one decimal (the mat's x = forward from the pan axis, y = left), e.g. `26.3 -0.4`.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO))
import joint_mapping as jm  # noqa: E402
import omx_constants as K  # noqa: E402
import solve_link_markers as SL  # noqa: E402

JOINTS = jm.LEROBOT_NAMES
TIP = np.array([K.GRIPPER_TIP_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0])
GRIP_CLOSED = 50.5           # fingers touching at 50.21 [已查證 2026-09-21]
# Targets (cm, mat frame): near / mid / far along the axis, left and right; >= 3 cm clear of the 6 cm paper markers
# at (32,20) (38,0) (24,-10) (45,12) (35,-15).
# Reach 18-36 cm; the 10-14 closed-loop grasps sit at 32-37 cm.
TARGETS = [(18.0, 0.0), (25.0, 0.0), (30.0, 8.0), (22.0, 14.0), (26.0, -18.0), (33.0, -6.0)]   # (35,-8): out of reach at 1 cm


class FitModel:
    def __init__(self, path: Path):
        f = json.loads(path.read_text(encoding="utf-8"))
        self.base = np.array(f["world_from_link0"])
        self.chain = [(tuple(o), a) for o, a in f["chain"]]
        self.off = np.array([f["offset_rad"][n] for n in JOINTS[:5]])
        self.sag = np.array([f["sag_rad_per_nm"][n] for n in ("sag_lift", "sag_elbow", "sag_flex")])
        self.g_in_base = bool(f.get("args", {}).get("gravity_in_base", False))
        self.inert = SL.urdf_inertials()
        self.name = path.name

    def link5(self, readings: np.ndarray) -> np.ndarray:
        q = jm.SCALE_RAD_PER_UNIT["shoulder_lift"] * np.asarray(readings[:5], float) + self.off
        g = np.array([0, 0, -SL.G]) if self.g_in_base else self.base[:3, :3].T @ np.array([0, 0, -SL.G])
        tau = SL.gravity_torques(SL.chain_transforms(q, self.chain), self.inert, g)
        q = q.copy()
        q[1:4] += self.sag * tau
        return self.base @ SL.chain_transforms(q, self.chain)[-1]

    def tip(self, readings) -> np.ndarray:
        return (self.link5(readings) @ TIP)[:3]

    def solve(self, target_m: np.ndarray, roll: float, seed: np.ndarray) -> tuple[np.ndarray, float]:
        """Readings (pan, lift, elbow, flex) putting the tip on target_m, pitch as close to straight down as reachable."""
        def res(r):
            rd = np.r_[r, roll, GRIP_CLOSED]
            T5 = self.link5(rd)
            down = np.degrees(np.arccos(np.clip(-T5[2, 0], -1, 1)))   # angle between the finger axis and straight down
            return np.r_[((T5 @ TIP)[:3] - target_m) * 1000.0, 0.05 * down]
        best = None
        for s_ in [seed] + [np.array([seed[0], l_, e_, f_]) for l_ in (30.0, 45.0, 55.0) for e_ in (-20.0, -40.0, -55.0) for f_ in (-20.0, -45.0)]:
            sol = least_squares(res, s_, bounds=([-90, -80, -80, -90], [90, 80, 80, 90]))
            if best is None or sol.cost < best.cost:
                best = sol
        sol = best
        T5 = self.link5(np.r_[sol.x, roll, GRIP_CLOSED])
        return sol.x, float(np.degrees(np.arccos(np.clip(-T5[2, 0], -1, 1))))


def plan(model: FitModel, roll: float, hover_mm: float):
    out = []
    for x, y in TARGETS:
        seed = np.array([np.degrees(np.arctan2(y, x)) / 1.7996, 40.0, -40.0, -30.0])
        hov, tilt = model.solve(np.array([x / 100, y / 100, hover_mm / 1000]), roll, seed)
        abv, _ = model.solve(np.array([x / 100, y / 100, 0.06]), roll, hov)
        err = np.linalg.norm(model.tip(np.r_[hov, roll, GRIP_CLOSED]) - [x / 100, y / 100, hover_mm / 1000]) * 1000
        out.append({"target_cm": (x, y), "hover": hov, "above": abv, "tilt_deg": tilt, "solve_err_mm": err})
    return out


def cmd_plan(args) -> int:
    model = FitModel(REPO / args.fit)
    home = json.loads((REPO / args.plan).read_text(encoding="utf-8"))["start"]
    print(f"model: {model.name}; roll fixed at HOME's {home[4]:.2f}; hover {args.hover_mm} mm")
    for p in plan(model, home[4], args.hover_mm):
        print(f"  target {p['target_cm']}: hover readings {np.round(p['hover'], 1)}  finger axis {p['tilt_deg']:.0f} deg off vertical,"
              f" solve error {p['solve_err_mm']:.1f} mm")
    return 0


def _go(arm, target: dict, min_s: float, units_per_s: float = 15.0) -> None:
    """move_to with a speed cap: HOME -> above-target is up to ~85 units on lift/elbow."""
    jump = max(abs(target[j] - arm.goal[j]) for j in target)
    arm.move_to(target, max(min_s, jump / units_per_s))


def cmd_session(args) -> int:
    import measure_link_tilt as S7  # noqa: PLC0415  (Arm, prompts)

    model = FitModel(REPO / args.fit)
    home_l = json.loads((REPO / args.plan).read_text(encoding="utf-8"))["start"]
    home = dict(zip(JOINTS, home_l))
    roll = home_l[4]
    csv_path = REPO / args.csv
    done = set()
    if csv_path.exists():
        done = {(float(r["target_x_cm"]), float(r["target_y_cm"])) for r in csv.DictReader(csv_path.open(encoding="utf-8"))}
    plans = [p for p in plan(model, roll, args.hover_mm) if tuple(p["target_cm"]) not in done]
    print("=" * 76)
    print(f"touch test: {len(plans)} targets left, model {model.name}. 桌上只留座標紙；夾爪會閉合。")
    print("每個點：手臂先到目標上方 6 cm，再降到紙面上方約 2 cm（依模型）。看著實際指尖用 d / u 降 / 升 3 mm，到離紙約 3-5 mm 讀數。")
    print("停在低處時馬達負荷大：讀完就按 Enter 輸入，不要久停。馬達閃紅燈就 Ctrl+C，跑 scripts/dxl_status.py。")
    print("Enter 後輸入指尖正下方的格子座標（cm，到小數一位，x 往前、y 往左），例如 26.3 -0.4；s = 跳過，q = 結束。")
    print("=" * 76)
    with S7.Arm(args.port, args.id) as arm:
        now = arm.read()
        far = {j: round(now[j] - home[j], 1) for j in JOINTS[:5] if abs(now[j] - home[j]) > 8}
        if far:
            print(f"🔴 手臂不在 HOME（差 {far}）。先執行 uv run python scripts/read_joint_pose.py --goto-home --home-episode 10")
            return 1
        for p in plans:
            x, y = p["target_cm"]
            print(f"\n--- 目標 ({x}, {y}) cm")
            go = S7._ask("    Enter = 移過去，s = 跳過，q = 結束: ").lower()
            if go == "q":
                break
            if go == "s":
                continue
            above = dict(zip(JOINTS, np.r_[p["above"], roll, GRIP_CLOSED]))
            _go(arm, above, 3.0)
            z_mm = args.hover_mm
            hov = dict(zip(JOINTS, np.r_[p["hover"], roll, GRIP_CLOSED]))
            _go(arm, hov, 2.0)
            # No hold(): pushing the goal past the sag kept shoulder_lift near stall and tripped Overload on 10-15
            # (S8 10-12 too). The sag is fine -- the analysis uses the readings actually present.
            while True:
                time.sleep(0.5)
                z_now = model.tip(np.array([v for v in arm.read_avg(5).values()]))[2] * 1000
                k = S7._ask(f"    指令高度 {z_mm:.0f} mm；依目前讀數模型算的指尖高度 {z_now:.0f} mm。"
                            f"看實際指尖：d 降 3 mm / u 升 3 mm / Enter 讀數 / s 跳過: ").lower()
                if k in ("d", "u"):
                    z_mm += -3 if k == "d" else 3
                    sol, _ = model.solve(np.array([x / 100, y / 100, z_mm / 1000]), roll, np.array(p["hover"]))
                    hov = dict(zip(JOINTS, np.r_[sol, roll, GRIP_CLOSED]))
                    _go(arm, hov, 0.8)
                    continue
                break
            if k == "s":
                _go(arm, above, 1.5)
                continue
            st = arm.read_avg(10)
            raw = S7._ask("    指尖正下方的格子座標 x y（cm）: ")
            try:
                mx, my = (float(v) for v in raw.replace(",", " ").split()[:2])
            except ValueError:
                print("    看不懂，這個點不記，回到上方。")
                _go(arm, above, 1.5)
                continue
            rd = np.array([st[j] for j in JOINTS])
            tip = model.tip(rd) * 100
            new = not csv_path.exists()
            with csv_path.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["target_x_cm", "target_y_cm", "measured_x_cm", "measured_y_cm", "model_tip_x_cm", "model_tip_y_cm",
                                "model_tip_z_cm", "fit"] + [f"state_{j}" for j in JOINTS])
                w.writerow([x, y, mx, my, round(tip[0], 2), round(tip[1], 2), round(tip[2], 2), model.name] + [round(st[j], 3) for j in JOINTS])
            print(f"    記下：實際 ({mx:.1f}, {my:.1f})，模型 ({tip[0]:.1f}, {tip[1]:.1f}) -> 差 ({mx - tip[0]:+.1f}, {my - tip[1]:+.1f}) cm")
            _go(arm, above, 1.5)
        print("\n回到 HOME……")
        _go(arm, home, 3.0)
    return cmd_analyse(args)


def cmd_analyse(args) -> int:
    rows = list(csv.DictReader((REPO / args.csv).open(encoding="utf-8")))
    if not rows:
        print("no rows")
        return 1
    model = FitModel(REPO / args.fit)
    print(f"\n{'target':>12} {'measured':>14} {'model tip':>14} {'error (measured - model) cm':>30} {'reach cm':>9}")
    errs = []
    for r in rows:
        rd = np.array([float(r[f"state_{j}"]) for j in JOINTS])
        tip = model.tip(rd) * 100
        m = np.array([float(r["measured_x_cm"]), float(r["measured_y_cm"])])
        e = m - tip[:2]
        reach = np.hypot(*m)
        errs.append((e, reach, m))
        print(f"({float(r['target_x_cm']):5.1f},{float(r['target_y_cm']):5.1f}) ({m[0]:5.1f},{m[1]:5.1f}) ({tip[0]:5.1f},{tip[1]:5.1f})"
              f"        ({e[0]:+5.2f}, {e[1]:+5.2f})  |{np.linalg.norm(e):.2f}|   {reach:6.1f}")
    E = np.array([e for e, _, _ in errs])
    print(f"\nmean error {np.round(E.mean(0), 2)} cm, spread (sd) {np.round(E.std(0), 2)} cm, model {model.name}")
    if len(errs) >= 3:
        u = np.array([m / np.linalg.norm(m) for _, _, m in errs])
        radial = (E * u).sum(1)
        reach = np.array([r for _, r, _ in errs])
        k = np.polyfit(reach, radial, 1)
        print(f"radial error vs reach: slope {k[0] * 100:+.1f} % (error grows {k[0] * 10:+.2f} cm per 10 cm of reach), offset {k[1]:+.2f} cm")
        print("  same vector everywhere -> base position on the mat; growing with reach -> arm model; ~0 -> the camera")
    return 0


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fit", default="calibration/2026-10-14_link_marker_fit.json")
    common.add_argument("--plan", default="configs/link_marker_plan_farwrist.json", help="only its HOME ('start') is used")
    common.add_argument("--hover-mm", type=float, default=20.0)
    p = sub.add_parser("plan", parents=[common]); p.set_defaults(func=cmd_plan)
    s = sub.add_parser("session", parents=[common]); s.add_argument("--csv", required=True)
    s.add_argument("--port", default="COM8"); s.add_argument("--id", default="2026-09-18_omx_follower"); s.set_defaults(func=cmd_session)
    a = sub.add_parser("analyse", parents=[common]); a.add_argument("--csv", required=True); a.set_defaults(func=cmd_analyse)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
