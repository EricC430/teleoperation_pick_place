#!/usr/bin/env python
"""Score candidate joint-mapping constant sets against every PHYSICAL ground truth we have.

No camera anywhere in this: recorded joint readings -> `joint_mapping` -> FK -> gripper points,
compared with positions known from the physical cell. That is deliberate. Until the front-left
camera's pose is ArUco-calibrated (S4 §5-5 T2), judging calibration by eye on a render mixes camera
error into arm error -- and tuning the arm until the render looks right writes camera error into
joint constants.

Gripper geometry (link5 frame, jaws closed): omx_constants.GRIPPER_{CROTCH,PINCH,TIP}_X_M, from the
CAD meshes -- crotch 3.9 cm (a pinched rim stops here), pinch 8.8 cm (jaws meet), tip 9.45 cm (the
lowest point when the gripper points down). Provenance and checks are next to the constants. The
committed "touch LSQ" constants were solved with the old 8.0 cm tip and a 13.05 cm riser.

Three ground truths, three different kinds of arm pose:

  touch    calibration/2026-09-22_touch_calibration.csv. 11 points where the fingertip physically
           touched a known mat coordinate with the gripper held VERTICAL. Torque off, arm
           supported by hand. Truth: tip at (x, y, 0), pitch 90 deg.
  grasp    uvc_60 (9/13), each episode's first sustained gripper close. Torque on, arm carrying
           itself. Episodes 0 and 1 were used to hand-tune the current constants and are not scored.
           xy: [已查證 2026-09-29] the gripper PINCHES THE CUP WALL, one jaw inside, one outside --
             at grasp the gripper reads ~52, i.e. jaws ~10 mm apart by joint_mapping's S6 jaw
             calibration (50.21 = touching) -- the jaws open to 150 mm, but here they are far
             narrower than the 6-7.5 cm cup -- and the wrist frames of ep2/3/11/37 show the rim
             between the jaws.
             So the pinch point sits on the cup's left or right wall, one radius from the
             placement point: the raw residual splits into two groups 8.1 cm apart (rim diameter
             7.5), constant in cm, not in degrees. 'wall-xy' scores against the nearer wall.
             ⚠️ A common tangential offset of ~-2.5 cm remains ('g-tan'). Constant-cm and
             constant-degree (pan-like, ~-5 deg) explain it equally well (sd 1.68 vs 1.71 cm);
             [未確認] which.
           z: [Eric說 2026-09-29] mostly jaws fully inserted -- crotch at the rim, tips at about
             half the cup height; some pinch only the rim. So for every grasp
             pinch z <= rim (9.5 cm) <= crotch z, and crotch-rim ~0 for most ('crotch-rim').
             Frame check: ep3 (front-left) shows the palm at the rim.
  release  uvc_60, the frame of the episode's FINAL gripper opening. Arm raised over the bin.
           Truth: pinch point inside the bin opening.

    python3 scripts/eval_joint_calibration.py

[已查證 2026-09-29] No constant-offset set satisfies touch AND grasp height: fitting touch puts the
crotch ~5 cm above the rim at grasp; fitting the grasps puts the touch points ~10 cm off with the
gripper tipped 32 deg past vertical. The per-joint readings of the grasps lie inside the touch
range (wrist_flex 0% outside), so this is not extrapolation of one joint. Whether it is load droop
(torque on vs off) or an arm-model error cannot be told from this data -- the height gap fits a
constant, ~r and ~r^2 model equally well (held-out 1.16-1.23 cm). That needs the lab test.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "sim"))
sys.path.insert(0, str(_REPO))   # first: the repo-root reach_logger/ package, not scripts/reach_logger.py

import joint_mapping as JM  # noqa: E402
import omx_constants as K  # noqa: E402
import scene_constants as S  # noqa: E402
from reach_logger import fk  # noqa: E402

TOUCH_CSV = _REPO / "calibration/2026-09-22_touch_calibration.csv"
UVC60 = sorted((_REPO / "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/data").glob("chunk-*/file-*.parquet"))
# [Eric說 2026-09-21] uvc_60 walked campA_136sym's t1..t60 in order, BUT [已查證 2026-09-29]
# docs/meeting/2026-09-13.md item 4: "t41 重錄後排在最後一集" -> ep0-39 = t1-t40, ep40-58 = t42-t60,
# ep59 = t41. Reading it as plain i -> t{i+1} put ep40-58 one placement off (23-52 cm "errors").
PLACEMENTS = _REPO / "configs/placements/campA_136sym_20260908_20260908_train.csv"
TUNED_EPISODES = {0, 1}

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NOMINAL = 0.03141593   # 1.80 deg/unit: 4096 ticks / 200 units

X_CROTCH, X_PINCH, X_TIP = K.GRIPPER_CROTCH_X_M, K.GRIPPER_PINCH_X_M, K.GRIPPER_TIP_X_M
TCP_Y = K.GRIPPER_MIDLINE_Y_M
CUP_R_AT_PINCH = 3.4                              # cm; cup radius 2.5 (base) .. 3.75 (rim)


def placement_of(ep: int) -> str:
    return f"train_{41 if ep == 59 else ep + 1 if ep < 40 else ep + 2:03d}"


def candidates() -> dict[str, tuple[dict, dict]]:
    cur_s, cur_o = dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD)
    nom = {k: NOMINAL for k in cur_s}
    roll = cur_o["wrist_roll"]
    touch_o = dict(cur_o)
    touch_o["shoulder_lift"] -= math.radians(2.0)    # undo the 2026-09 hand-tune
    touch_o["wrist_flex"] -= math.radians(24.0)
    return {
        "S6 two-pose (4248342)": (
            {"shoulder_pan": 0.03104445, "shoulder_lift": 0.03143896, "elbow_flex": 0.03248061,
             "wrist_flex": 0.03159506, "wrist_roll": 0.03086570},
            {"shoulder_pan": 0.01592023, "shoulder_lift": -0.36919370, "elbow_flex": 0.44010560,
             "wrist_flex": 1.62968550, "wrist_roll": -0.02788842}),
        # `touch_calibrate solve` as committed: TCP 8.0 cm, riser free -> 13.05 cm.
        "touch LSQ (committed)": (nom, touch_o),
        # T: same 11 touches, tip at the CAD 9.45 cm, riser fixed at the measured 15 cm.
        "T: touch refit, CAD tip": (nom, {"shoulder_pan": -0.03886218, "shoulder_lift": 0.11817732,
                                          "elbow_flex": -0.08640457, "wrist_flex": 1.51261340,
                                          "wrist_roll": roll}),
        # K: fitted on the EVEN grasp episodes only (crotch@rim + wall-pinch xy, soft_l1), touch
        # not used. Scored below on everything, so odd episodes and release are held out for it.
        "K: task-pose fit (even eps)": (nom, {"shoulder_pan": 0.01878756, "shoulder_lift": 0.22986734,
                                              "elbow_flex": -0.02231385, "wrist_flex": 1.95185293,
                                              "wrist_roll": roll}),
        "current joint_mapping.py": (cur_s, cur_o),
        **EXTRA,
    }


# --candidate-json: constant sets written by `measure_link_tilt.py solve --out` (S7) or any JSON of
# the same shape {name: {"scale": {joint: rad/unit}, "offset": {joint: rad}}}, scored beside the rest.
EXTRA: dict[str, tuple[dict, dict]] = {}


def load_extra(paths: list[str]) -> None:
    import json  # noqa: PLC0415

    for path in paths:
        for name, c in json.loads(Path(path).read_text(encoding="utf-8")).items():
            EXTRA[name[:27]] = (c["scale"], c["offset"])


def use(scale: dict, offset: dict) -> None:
    JM.SCALE_RAD_PER_UNIT.clear(); JM.SCALE_RAD_PER_UNIT.update(scale)
    JM.OFFSET_RAD.clear(); JM.OFFSET_RAD.update(offset)


def q5(pos6) -> list[float]:
    return JM.row_to_sim_rad([float(v) for v in pos6])[:5]


def point_cm(q, x_link5: float) -> np.ndarray:
    """A point on the gripper's midline, in cm above the TABLE (arm base on the riser)."""
    p = fk.link5_transform(q) @ np.array([x_link5, TCP_Y, 0.0, 1.0])
    return np.array([p[0] * 100, p[1] * 100, p[2] * 100 + S.ARM_RISER_HEIGHT * 100])


def pitch_deg(q) -> float:
    """90 = straight down; >90 = tipped back past vertical. atan2 along the reach direction:
    an asin(-v_z) version folds 113 deg onto 67 deg -- that fold hid a result once already."""
    v = fk.link5_transform(q)[:3, 0]
    along = v[0] * math.cos(q[0]) + v[1] * math.sin(q[0])
    return math.degrees(math.atan2(-v[2], along))


def load_uvc60() -> dict[int, np.ndarray]:
    import pyarrow.parquet as pq
    eps: dict[int, list] = {}
    for f in UVC60:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        for e, fi, st in zip(t["episode_index"], t["frame_index"], t["observation.state"]):
            eps.setdefault(int(e), []).append((int(fi), st))
    return {e: np.array([s for _, s in sorted(v)]) for e, v in eps.items()}


def grasp_and_release(st: np.ndarray, hold: int = 15) -> tuple[int | None, int | None]:
    """Gripper signal ONLY -- never FK -- so every candidate is scored on the same frames.

    Grasp = first closing that then STAYS closed for `hold` frames (1 s at 15 fps). The earlier
    "most closed frame" (argmin) detector picked a frame mid-carry or at a regrasp in 15 of 60
    episodes -- the gripper stays closed the whole way to the bin, and its minimum can fall anywhere
    in that stretch (ep0: argmin frame 403, 30 cm from the cup; the real grasp is frame 226)."""
    g = st[:, 5]
    thr = np.percentile(g, 95) - 0.6 * (np.percentile(g, 95) - np.percentile(g, 5))
    grasp = next((i for i in range(1, len(g) - hold)
                  if g[i - 1] >= thr > g[i] and np.all(g[i:i + hold] < thr)), None)
    opens = [i for i in range(1, len(g)) if g[i - 1] < thr <= g[i]]
    return grasp, (opens[-1] if opens else None)                # final opening = drop into the bin


def wall_residual(pinch_xy: np.ndarray, cup_xy: np.ndarray) -> tuple[float, float]:
    """(radial, tangential) error of the pinch point against the NEARER side wall of the cup."""
    u = cup_xy / np.linalg.norm(cup_xy); w = np.array([-u[1], u[0]])
    d = pinch_xy - cup_xy
    tan = d @ w
    side = min((1.0, -1.0), key=lambda s: abs(tan - s * CUP_R_AT_PINCH))
    return float(d @ u), float(tan - side * CUP_R_AT_PINCH)


def main() -> int:
    touch = list(csv.DictReader(TOUCH_CSV.open(encoding="utf-8")))
    place = {r["placement_id"]: np.array([float(r["x_cm"]), float(r["y_cm"])])
             for r in csv.DictReader(PLACEMENTS.open(encoding="utf-8"))}
    eps = load_uvc60()
    bin_xy = np.array([S.BIN_CENTER_X, S.BIN_CENTER_Y]) * 100
    bin_r, bin_rim = S.BIN_OPENING_DIA / 2 * 100, (S.ARM_RISER_HEIGHT + S.BIN_HEIGHT) * 100
    rim = S.CUP_HEIGHT * 100
    frames = {e: grasp_and_release(st) for e, st in eps.items()}
    saved = (dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD))

    print(f"touch {len(touch)} pts (tip x={X_TIP*100:.2f}) | grasp ep2-59 (pinch x={X_PINCH*100:.1f}, crotch "
          f"x={X_CROTCH*100:.1f}, cup rim {rim:.1f}) | release -> bin r={bin_r:.0f} rim {bin_rim:.0f} cm\n")
    hdr = (f"{'candidate':<28}| {'TOUCH 3D':>8} {'z':>6} {'pitch':>6} | {'crotch-rim':>10} {'h-ok':>5} "
           f"{'wall-xy':>7} {'g-tan':>6} | {'in-bin':>6} {'d':>6}")
    print(hdr); print("-" * len(hdr))

    results = {}
    for name, (sc, of) in candidates().items():
        use(sc, of)
        t3, tz, tp = [], [], []
        for r in touch:
            q = q5([r[f"state_{j}"] for j in JOINTS])
            p = point_cm(q, X_TIP)
            t3.append(np.linalg.norm(p - [float(r["target_x_cm"]), float(r["target_y_cm"]), 0.0]))
            tz.append(p[2]); tp.append(pitch_deg(q))

        cr, hok, wxy, gtan, rd = [], [], [], [], []
        for e, st in eps.items():
            gi, ri = frames[e]
            if gi is not None and e not in TUNED_EPISODES:
                q = q5(st[gi])
                zc, pp = point_cm(q, X_CROTCH)[2], point_cm(q, X_PINCH)
                cr.append(zc - rim); hok.append(pp[2] <= rim + 1.0 and zc >= rim - 1.0)
                rad, tan = wall_residual(pp[:2], place[placement_of(e)])
                wxy.append(math.hypot(rad, tan)); gtan.append(tan)
            if ri is not None:
                rd.append(np.linalg.norm(point_cm(q5(st[ri]), X_PINCH)[:2] - bin_xy))

        med = lambda a: float(np.median(a))
        results[name] = dict(touch=med(t3), height=abs(med(cr)))
        print(f"{name:<28}| {med(t3):7.1f}c {med(tz):+5.1f}c {med(tp):5.1f}° | {med(cr):+9.1f}c {np.mean(hok):5.0%} "
              f"{med(wxy):6.1f}c {med(gtan):+5.1f}c | {sum(d <= bin_r for d in rd):>3}/{len(rd):<2} {med(rd):5.1f}c")
    use(*saved)

    print("\nhow to read: TOUCH truth is z=0, pitch 90. crotch-rim should be ~0 for most grasps; h-ok = share with")
    print("pinch <= rim <= crotch (1 cm slack). wall-xy = pinch point to the nearer cup wall; g-tan = its signed")
    print("tangential part (the unexplained common offset). in-bin: releases whose pinch point is over the opening.")

    best_t = min(results, key=lambda n: results[n]["touch"])
    best_h = min(results, key=lambda n: results[n]["height"])
    if best_t != best_h:
        a, b = results[best_t], results[best_h]
        print(f"\n🔴 best touch '{best_t}' ({a['touch']:.1f} cm) leaves crotch-rim at {a['height']:+.1f} cm; best grasp height "
              f"'{best_h}' ({b['height']:+.1f} cm) costs touch {b['touch']:.1f} cm.")
        print("   No offset set satisfies both -> touch poses and grasp poses disagree (droop or arm model; see docstring).")
    return 0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidate-json", nargs="*", default=[],
                    help="extra constant sets to score, e.g. the S7 fit (measure_link_tilt.py solve --out)")
    load_extra(ap.parse_args().candidate_json)
    sys.exit(main())
