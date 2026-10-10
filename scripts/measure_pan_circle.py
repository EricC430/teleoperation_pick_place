#!/usr/bin/env python
"""S8: where the real pan axis stands on the placement paper, and the pan joint's zero and scale.

Spec: `docs/specs/S8_pan_axis_on_paper.md`.

Circle point analysis: hold every joint but `shoulder_pan`, turn pan in steps, and at each stop
read off the paper the point straight under the fingertip. Those points lie on a circle whose
CENTRE is the pan axis -- whatever lift/elbow/wrist zeros are, they only change the RADIUS. So this
measures the base position against the paper without trusting any other joint, which is what
10/08's paper_to_pan fit could not do (it went through FK, whose zeros were themselves fitted to
paper touches, so a base offset could hide inside them).

Per circle k, point i (pan reading r_i, paper point (x_i, y_i) in cm):

    (x_i, y_i) = c + rho_k * (cos(phi_i + alpha_k), sin(phi_i + alpha_k)),   phi_i = s * r_i + o

c = pan axis on the paper; s, o = pan scale (deg/unit) and zero; rho_k = the circle's radius;
alpha_k = the fingertip's azimuth inside the arm plane (CAD fingertip is 1.65 mm off the midline),
from FK. `o` is the pan zero IN THE PAPER FRAME: it carries any yaw of the base plate on the riser
too, and the two cannot be separated by turning pan -- for the sim that is fine as long as the
base mesh is left unrotated (spec §5).

    # lab day (needs the arm; clear the table within ~30 cm of the arm first)
    uv run python scripts/measure_pan_circle.py session --csv calibration/<date>_pan_circle.csv

    # anywhere
    python scripts/measure_pan_circle.py solve --csv calibration/<date>_pan_circle.csv
    python scripts/measure_pan_circle.py selftest

🔴 Never edits `sim/joint_mapping.py` or `sim/scene_constants.py`; `solve` prints what to paste.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time
from pathlib import Path

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

import numpy as np  # noqa: E402

import measure_link_tilt as S7  # noqa: E402  (shares the bus wrapper, prompts and FK setup)
from measure_link_tilt import JM, JOINTS, fk  # noqa: E402

FIELDS = ("circle", "sweep", "pan_target", "paper_x_cm", "paper_y_cm", "note") + tuple(f"state_{j}" for j in JOINTS)
PAN_AXIS_IN_BASE_M = (-0.01125, 0.0)    # fk._PAN_AXIS_XY: the pan axis sits 1.125 cm behind the base plate centre


def deg_per_unit_nominal() -> float:
    return math.degrees(JM.SIGN["shoulder_pan"] * JM.SCALE_RAD_PER_UNIT["shoulder_pan"])


def tip_polar_in_arm_plane(state: dict[str, float]) -> tuple[float, float]:
    """(radius cm, azimuth deg) of the CAD fingertip about the pan axis with pan at 0, other joints
    on the current constants. The azimuth is the only FK quantity the fit uses; the radius is
    printed as a cross-check of the pitch-chain constants (S7)."""
    import omx_constants as K  # noqa: PLC0415

    q = JM.lerobot_to_urdf_rad(np.array([state[j] for j in JOINTS]))[:5].tolist()
    q[0] = 0.0
    x, y, _ = (fk.link5_transform(q) @ np.array([K.GRIPPER_TIP_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0]))[:3]
    dx, dy = x - PAN_AXIS_IN_BASE_M[0], y - PAN_AXIS_IN_BASE_M[1]
    return 100 * math.hypot(dx, dy), math.degrees(math.atan2(dy, dx))


# --------------------------------------------------------------------------------------------------
# model / fit
# --------------------------------------------------------------------------------------------------

def unpack(theta: np.ndarray, circles: list[str]) -> dict:
    return {"cx": theta[0], "cy": theta[1], "s": theta[2], "o": theta[3],
            "rho": {c: theta[4 + k] for k, c in enumerate(circles)}}


def predict(rows: list[dict], P: dict, alpha: dict[str, float]) -> np.ndarray:
    out = []
    for r in rows:
        a = math.radians(P["s"] * r["pan"] + P["o"] + alpha[r["circle"]])
        out.append((P["cx"] + P["rho"][r["circle"]] * math.cos(a), P["cy"] + P["rho"][r["circle"]] * math.sin(a)))
    return np.array(out)


def kasa(xy: np.ndarray) -> tuple[float, float, float]:
    """Algebraic circle fit, the start point for the geometric fit."""
    A = np.c_[2 * xy, np.ones(len(xy))]
    b = (xy ** 2).sum(1)
    cx, cy, c = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(cx), float(cy), float(math.sqrt(c + cx * cx + cy * cy))


def fit(rows: list[dict], alpha: dict[str, float], fix_scale: bool = False, iters: int = 80):
    circles = sorted({r["circle"] for r in rows})
    obs = np.array([[r["x"], r["y"]] for r in rows])
    cx, cy, _ = kasa(obs)
    rho0 = [float(np.mean([math.hypot(r["x"] - cx, r["y"] - cy) for r in rows if r["circle"] == c])) for c in circles]
    s0 = deg_per_unit_nominal()
    o0 = float(np.degrees(np.angle(np.mean([np.exp(1j * (math.atan2(r["y"] - cy, r["x"] - cx)
                                                         - math.radians(s0 * r["pan"] + alpha[r["circle"]]))) for r in rows]))))
    theta = np.array([cx, cy, s0, o0] + rho0, float)
    free = [k for k in range(len(theta)) if not (fix_scale and k == 2)]

    def res(th):
        return (obs - predict(rows, unpack(th, circles), alpha)).ravel()

    lam, r0 = 1e-3, res(theta)
    for _ in range(iters):
        J = np.empty((len(r0), len(free)))
        for m, k in enumerate(free):
            t = theta.copy()
            t[k] += 1e-5
            J[:, m] = -(res(t) - r0) / 1e-5
        A = J.T @ J
        step = np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), J.T @ r0)
        t = theta.copy()
        t[free] += step
        rt = res(t)
        if rt @ rt < r0 @ r0:
            theta, r0, lam = t, rt, lam / 3
            if np.abs(step).max() < 1e-9:
                break
        else:
            lam *= 5
    return unpack(theta, circles), r0.reshape(-1, 2)


def bootstrap(rows, alpha, n=400, seed=0, fix_scale=False):
    rng = np.random.default_rng(seed)
    out = []
    by_c = {}
    for r in rows:
        by_c.setdefault(r["circle"], []).append(r)
    for _ in range(n):
        sample = []
        for rs in by_c.values():
            sample += [rs[i] for i in rng.integers(0, len(rs), len(rs))]
        if any(len({r["pan"] for r in sample if r["circle"] == c}) < 3 for c in by_c):
            continue
        P, _ = fit(sample, alpha, fix_scale, iters=40)
        out.append((P["cx"], P["cy"], P["s"], P["o"]))
    return np.array(out)


# --------------------------------------------------------------------------------------------------
# CSV
# --------------------------------------------------------------------------------------------------

def read_rows(path: Path) -> list[dict]:
    rows = []
    for r in csv.DictReader(path.open(encoding="utf-8")):
        if not r.get("paper_x_cm"):
            continue
        st = {j: float(r[f"state_{j}"]) for j in JOINTS}
        rows.append({"circle": r["circle"], "sweep": r["sweep"], "target": float(r["pan_target"]),
                     "pan": st["shoulder_pan"], "x": float(r["paper_x_cm"]), "y": float(r["paper_y_cm"]), "state": st})
    return rows


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


# --------------------------------------------------------------------------------------------------
# session
# --------------------------------------------------------------------------------------------------

def hand_pose(arm, joints: list[str]) -> bool:
    """Release `joints` so the operator can place the fingertip; hold everything again on Enter."""
    bus = arm.bus
    pos = arm.read()
    bus.sync_write("Goal_Position", pos)
    print(f"    🔴 即將放掉 {', '.join(joints)}：手臂這幾節會往下掉，先用手托住前臂和夾爪。其他關節保持出力。")
    if S7._ask("    托好了按 Enter（q 結束）: ").lower() == "q":
        return False
    bus.disable_torque(joints, num_retry=2)
    print("    把夾爪擺成『尖端朝正下方、離紙面 3–5 mm、不要碰到紙』，扶著不要動。")
    ok = S7._ask("    擺好、手還扶著時按 Enter，我會在這個姿勢鎖住（q 結束）: ").lower() != "q"
    pos = arm.read()
    bus.sync_write("Goal_Position", pos)
    bus.enable_torque(num_retry=2)
    time.sleep(0.3)
    print("    已鎖住，可以放手。")
    return ok


def ask_xy(prompt: str):
    while True:
        raw = S7._ask(prompt)
        if raw.lower() in ("q", ""):
            return raw.lower() or None
        parts = raw.replace(",", " ").split()
        try:
            if len(parts) == 2:
                return float(parts[0]), float(parts[1])
        except ValueError:
            pass
        print("      格式是『x y』，單位 cm，例如 18.3 -4.1（空白=略過，q=結束）")


def cmd_session(args) -> int:
    csv_path = Path(args.csv)
    done = set()
    if csv_path.exists():
        for r in csv.DictReader(csv_path.open(encoding="utf-8")):
            done.add((r["circle"], r["sweep"], f"{float(r['pan_target']):.2f}"))
    span = args.pan_range_deg / deg_per_unit_nominal()
    targets = list(np.linspace(args.pan_center_units - span, args.pan_center_units + span, args.points))
    print("=" * 76)
    print("S8 session：轉 pan、讀指尖正下方的座標紙座標。開始前：")
    print("  * 手臂周圍 30 cm 內清空（杯子、收納盒移走），座標紙不要動")
    print(f"  * pan 會在 {args.pan_center_units - span:+.1f} 到 {args.pan_center_units + span:+.1f} 單位之間轉"
          f"（約 ±{args.pan_range_deg:.0f}°），每站停下來等你讀座標")
    print("  * 讀座標時眼睛在指尖正上方往下看，避免視差；讀到 0.1 cm")
    print("=" * 76)
    with S7.Arm(args.port, args.id) as arm:
        arm.move_to(dict(arm.read(), gripper=args.gripper_closed), 1.5)
        for k in range(args.circles):
            circle = f"C{k}"
            print(f"\n--- 圓 {circle}：" + ("夾爪伸近一點（半徑約 12–15 cm）" if k == 0 else "夾爪伸遠一點（半徑約 22–25 cm）"))
            go = S7._ask("    Enter = 先把 pan 轉到起點再擺姿勢，s = 跳過這個圓，q = 結束: ").lower()
            if go == "q":
                return 0
            if go == "s":
                continue
            arm.move_to(dict(arm.read(), shoulder_pan=targets[0] - args.preload_units), 2.5)
            if not hand_pose(arm, ["shoulder_lift", "elbow_flex", "wrist_flex"]):
                return 0
            held = arm.read_avg()
            for sweep, seq in (("up", targets), ("down", targets[::-1][::2] if args.reverse else [])):
                if not seq:
                    continue
                if sweep == "down":
                    print("    反向掃描（量 pan 背隙）：每隔一站再讀一次")
                    arm.move_to(dict(arm.read(), shoulder_pan=seq[0] + args.preload_units), 1.5)
                for tgt in seq:
                    if (circle, sweep, f"{tgt:.2f}") in done:
                        continue
                    arm.move_to(dict(arm.read(), shoulder_pan=float(tgt)), 1.5)
                    time.sleep(args.settle)
                    st = arm.read_avg()
                    drift = {j: round(st[j] - held[j], 1) for j in ("shoulder_lift", "elbow_flex", "wrist_flex")
                             if abs(st[j] - held[j]) > 1.0}
                    if drift:
                        print(f"    ⚠️  其他關節跑掉了 {drift}（單位），這一站的點會落在不同半徑上")
                    xy = ask_xy(f"    pan {st['shoulder_pan']:+6.1f}：指尖正下方的座標 x y（cm）: ")
                    if xy == "q":
                        return 0
                    if xy is None:
                        continue
                    append_row(csv_path, {"circle": circle, "sweep": sweep, "pan_target": round(float(tgt), 2),
                                          "paper_x_cm": xy[0], "paper_y_cm": xy[1], "note": "",
                                          **{f"state_{j}": round(st[j], 3) for j in JOINTS}})
            print("    這個圓量完。把夾爪抬離紙面……")
            arm.move_to(dict(arm.read(), wrist_flex=arm.read()["wrist_flex"] - 8.0), 1.5)
        print(f"\n完成，扭力開著。下一步：python scripts/measure_pan_circle.py solve --csv {csv_path}")
    return 0


# --------------------------------------------------------------------------------------------------
# solve
# --------------------------------------------------------------------------------------------------

def cmd_solve(args) -> int:
    rows = read_rows(Path(args.csv))
    if len(rows) < 4:
        raise SystemExit(f"need at least 4 points, {args.csv} has {len(rows)}")
    circles = sorted({r["circle"] for r in rows})
    alpha, rho_fk = {}, {}
    for c in circles:
        st = {j: float(np.mean([r["state"][j] for r in rows if r["circle"] == c])) for j in JOINTS}
        rho_fk[c], alpha[c] = tip_polar_in_arm_plane(st)
    up = [r for r in rows if r["sweep"] == "up"]
    P, res = fit(up, alpha)
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    B = bootstrap(up, alpha, n=args.bootstrap)
    lo, hi = np.percentile(B, 2.5, axis=0), np.percentile(B, 97.5, axis=0)
    s_nom = deg_per_unit_nominal()
    o_now = math.degrees(JM.OFFSET_RAD["shoulder_pan"])
    print(f"{len(up)} points on {len(circles)} circle(s) (up-sweep); fit residual RMS {rms:.2f} cm")
    for c in circles:
        n = sum(1 for r in up if r["circle"] == c)
        span = np.ptp([r["pan"] for r in up if r["circle"] == c]) * s_nom
        print(f"  {c}: {n} points over {span:.0f} deg; radius {P['rho'][c]:.2f} cm (FK with the current constants "
              f"says {rho_fk[c]:.2f} cm -> {P['rho'][c] - rho_fk[c]:+.2f} cm, a pitch-chain error S7 should explain)")
    print(f"\npan axis on the paper:  x = {P['cx']:+.2f} cm [{lo[0]:+.2f}, {hi[0]:+.2f}]   "
          f"y = {P['cy']:+.2f} cm [{lo[1]:+.2f}, {hi[1]:+.2f}]   (95% bootstrap)")
    print(f"pan scale: {P['s']:.4f} deg/unit [{lo[2]:.4f}, {hi[2]:.4f}]   (nominal {s_nom:.4f}; "
          f"{100 * (P['s'] / s_nom - 1):+.2f}%)")
    print(f"pan zero in the paper frame: {P['o']:+.2f} deg [{lo[3]:+.2f}, {hi[3]:+.2f}]   "
          f"(joint_mapping now {o_now:+.2f} -> change {P['o'] - o_now:+.2f})")
    if len(circles) > 1:
        cs = []
        for c in circles:
            Pc, _ = fit([r for r in up if r["circle"] == c], {c: alpha[c]})
            cs.append((c, Pc["cx"], Pc["cy"]))
        print("each circle alone (agreement = the axis is vertical and the readings are consistent):")
        for c, x, y in cs:
            print(f"  {c}: centre ({x:+.2f}, {y:+.2f}) cm")
    down = [r for r in rows if r["sweep"] == "down"]
    if down:
        d_ang = []
        for r in down:
            a_meas = math.degrees(math.atan2(r["y"] - P["cy"], r["x"] - P["cx"])) - alpha[r["circle"]]
            a_pred = P["s"] * r["pan"] + P["o"]
            d_ang.append(S7.wrap(a_meas - a_pred))
        print(f"\nreverse sweep: arm angle minus the up-sweep model, median {np.median(d_ang):+.2f} deg over {len(d_ang)} "
              "points (pan backlash beyond what the encoder reads)")

    print("\nwhat this means for the sim (spec §5):")
    print(f"  * the arm's pan axis is at ({P['cx']:+.2f}, {P['cy']:+.2f}) cm in the paper/world frame, not (0, 0):")
    print("    shift the ROBOT by this in the scene; objects, ArUco extrinsics and placements stay in paper coords.")
    print(f"  * pan zero {P['o']:+.2f} deg (paper frame) replaces OFFSET_RAD['shoulder_pan'] while the base mesh stays unrotated.")
    print(f"\n# ---- paste ({args.date}, from {Path(args.csv).name}) ----")
    print(f'#   sim/joint_mapping.py   "shoulder_pan": SCALE {math.radians(P["s"]) / JM.SIGN["shoulder_pan"]:.8f}   '
          f'OFFSET {math.radians(P["o"]):.8f}')
    print(f"#   sim/scene_constants.py ARM_PAN_AXIS_IN_PAPER_M = ({P['cx'] / 100:.4f}, {P['cy'] / 100:.4f})")
    return 0


# --------------------------------------------------------------------------------------------------
# selftest
# --------------------------------------------------------------------------------------------------

def synthetic(rng, truth: dict, noise_cm: float):
    rows = []
    for c, rho, lift in (("C0", 13.0, 20.0), ("C1", 24.0, 35.0)):
        st0 = {"shoulder_pan": 0.0, "shoulder_lift": lift, "elbow_flex": -10.0, "wrist_flex": -45.0,
               "wrist_roll": 0.0, "gripper": 50.5}
        _, alpha = tip_polar_in_arm_plane(st0)
        for r in np.linspace(-33.3, 33.3, 9):
            a = math.radians(truth["s"] * r + truth["o"] + alpha)
            rows.append({"circle": c, "sweep": "up", "target": r, "pan": float(r) + rng.normal(0, 0.05),
                         "x": truth["cx"] + rho * math.cos(a) + rng.normal(0, noise_cm),
                         "y": truth["cy"] + rho * math.sin(a) + rng.normal(0, noise_cm),
                         "state": dict(st0, shoulder_pan=float(r))})
    return rows


def cmd_selftest(_args) -> int:
    rng = np.random.default_rng(1)
    truth = {"cx": 1.4, "cy": -0.8, "s": deg_per_unit_nominal() * 1.01, "o": -3.5}
    rows = synthetic(rng, truth, 0.1)
    alpha = {}
    for c in ("C0", "C1"):
        st = {j: float(np.mean([r["state"][j] for r in rows if r["circle"] == c])) for j in JOINTS}
        alpha[c] = tip_polar_in_arm_plane(st)[1]
    P, res = fit(rows, alpha)
    err = {k: P[k] - truth[k] for k in ("cx", "cy", "s", "o")}
    for k, e in err.items():
        print(f"  {k:3s} recovered error {e:+.4f}")
    ok = abs(err["cx"]) < 0.15 and abs(err["cy"]) < 0.15 and abs(err["s"]) < 0.01 and abs(err["o"]) < 0.5
    print("✅ circle fit recovers a known pan axis" if ok else "🔴 circle fit is WRONG")
    return 0 if ok else 1


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    se = sub.add_parser("session", help="turn pan and record paper points (needs the arm)")
    se.add_argument("--csv", required=True)
    se.add_argument("--port", default="COM8")
    se.add_argument("--id", default="2026-09-18_omx_follower")
    se.add_argument("--circles", type=int, default=2)
    se.add_argument("--points", type=int, default=9)
    se.add_argument("--pan-range-deg", type=float, default=60.0)
    se.add_argument("--pan-center-units", type=float, default=0.0)
    se.add_argument("--preload-units", type=float, default=3.0,
                    help="approach every stop from the same side: start this far past the first stop")
    se.add_argument("--no-reverse", dest="reverse", action="store_false", help="skip the backlash sweep")
    se.add_argument("--gripper-closed", type=float, default=50.5)
    se.add_argument("--settle", type=float, default=1.0)
    se.set_defaults(func=cmd_session)
    so = sub.add_parser("solve", help="fit the circle(s) (no hardware)")
    so.add_argument("--csv", required=True)
    so.add_argument("--bootstrap", type=int, default=400)
    so.add_argument("--date", default="YYYY-MM-DD")
    so.set_defaults(func=cmd_solve)
    st = sub.add_parser("selftest")
    st.set_defaults(func=cmd_selftest)
    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
