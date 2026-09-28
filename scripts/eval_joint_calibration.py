#!/usr/bin/env python
"""Score candidate joint-mapping constant sets against every PHYSICAL ground truth we have.

No camera anywhere in this: recorded joint readings -> `joint_mapping` -> FK -> a point on the
gripper, compared with positions known from the physical cell. That is deliberate. Until the
front-left camera's pose is ArUco-calibrated (S4 §5-5 T2), judging calibration by eye on a render
mixes camera error into arm error.

Gripper geometry -- which point on the gripper each truth is about. [已查證 2026-09-29, CAD meshes
follower_06/07/08 in open_manipulator_description, jaws closed = gripper joints at 0], link5 +x:
  crotch 3.9 cm   inner end of the jaw gap (palm face 3.95)   <- "虎口"
  pinch  8.8 cm   where the two jaws meet                     <- what closes on the cup wall
  tip    9.45 cm  outermost fingertip                         <- what touches the mat
sim/omx_constants.py uses 8.0 cm ([柏宇說 2026-09-21], one tape measurement from link5's origin,
which is not a visible point on the arm). Evidence for the CAD value: the touch solve with the riser
left free lands on 13.05 cm with an 8.0 tip but on 14.84 cm with 9.45, and the riser is 15 cm
([Eric說]; swapped once, within ~1 cm). Changing omx_constants is a separate decision -- this
script only needs the right point for each truth.

Three ground truths:

  touch    calibration/2026-09-22_touch_calibration.csv. 11 points where the fingertip (tip) touched
           a known mat coordinate with the gripper held VERTICAL, torque off, arm hand-supported.
           Truth: (x, y, 0) and pitch 90 deg.
  grasp    uvc_60, each episode's first sustained gripper close. Torque on.
           [已查證 2026-09-29, wrist-camera frames of ep2/3/11/37] the gripper PINCHES THE CUP WALL
           (jaws nearly closed: 50.21 = jaws touching, ~52 at grasp), one jaw inside the cup. So the
           pinch point is NOT at the placement (cup centre) but on the cup's left or right wall,
           +-R tangentially. The two groups sit 8.1 cm apart (rim diameter 7.5); which wall varies by
           episode, and the side is taken from the residual's sign.
           Height truth ([Eric說 2026-09-29]): most grasps go in until the crotch sits on the rim, the
           rest pinch the rim with the jaw tips. So for every grasp: pinch z <= rim (9.5) <= crotch z,
           and crotch ~ rim for most.
           Still unexplained: a common tangential offset of ~-2.5 cm after removing the wall. It fits
           "constant cm" and "constant deg (~-5.5)" equally well; unresolved.
  release  uvc_60, the frame of the episode's FINAL gripper opening. Truth: pinch point inside the
           bin opening. Height is only a soft check -- the cup may be lowered into the bin.

    python3 scripts/eval_joint_calibration.py

🔴 [已查證 2026-09-29] No single offset set satisfies touch AND grasp height: fitting touch puts the
crotch ~+5 cm above the rim at grasp; fitting grasp (K) puts touch 10 cm off with a 122 deg pitch;
fitting both (scratch) leaves touch 2.2 cm below the table and the crotch still +2.5 cm. The height
gap grows with reach but offset-type and droop-type models fit it equally well (held-out 1.16-1.23
cm) -- telling them apart needs the lab (same pose torque on vs off; touches at grasp-like poses).
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "sim"))

import joint_mapping as JM  # noqa: E402
import scene_constants as S  # noqa: E402
from reach_logger import fk  # noqa: E402

TOUCH_CSV = _REPO / "calibration/2026-09-22_touch_calibration.csv"
UVC60 = sorted((_REPO / "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/data").glob("chunk-*/file-*.parquet"))
# [Eric說 2026-09-21] uvc_60 walked campA_136sym's t1..t60 in order, BUT [已查證 2026-09-29]
# docs/meeting/2026-09-13.md item 4: "t41 重錄後排在最後一集" -> ep0-39 = t1-t40, ep40-58 = t42-t60,
# ep59 = t41. Reading it as plain i -> t{i+1} put ep40-58 one placement off (23-52 cm "errors").
PLACEMENTS = _REPO / "configs/placements/campA_136sym_20260908_20260908_train.csv"
TUNED_EPISODES = {0, 1}   # used to hand-tune "current"; reported apart, kept out of the score
JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NOMINAL = 0.03141593      # 1.80 deg/unit: 4096 ticks / 200 units; all follower calibrations are 0..4095
RISER_CM = S.ARM_RISER_HEIGHT * 100
X_CROTCH, X_PINCH, X_TIP = 0.039, 0.088, 0.0945   # link5 +x, metres, see docstring
TCP_Y = -0.00165                                  # midline between the jaw pivots (omx_constants)
R_CUP = 3.4               # cup radius around the pinch region (rim 3.75, base 2.5)
CUP_RIM = S.CUP_HEIGHT * 100


def placement_of(ep: int) -> str:
    return f"train_{41 if ep == 59 else ep + 1 if ep < 40 else ep + 2:03d}"


def candidates() -> dict[str, tuple[dict, dict]]:
    cur_s, cur_o = dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD)
    nom = {k: NOMINAL for k in cur_s}
    roll = cur_o["wrist_roll"]
    touch_o = dict(cur_o)
    touch_o["shoulder_lift"] -= math.radians(2.0)    # undo the 2026-09 hand-tune
    touch_o["wrist_flex"] -= math.radians(24.0)

    def o(pan, lift, elbow, wflex):
        return {"shoulder_pan": pan, "shoulder_lift": lift, "elbow_flex": elbow, "wrist_flex": wflex, "wrist_roll": roll}
    return {
        "S6 two-pose (4248342)": (
            {"shoulder_pan": 0.03104445, "shoulder_lift": 0.03143896, "elbow_flex": 0.03248061,
             "wrist_flex": 0.03159506, "wrist_roll": 0.03086570},
            o(0.01592023, -0.36919370, 0.44010560, 1.62968550)),
        # touch_calibrate solve as committed: 8.0 cm tip, riser FREE (prior 14) -> landed on 13.05 cm
        "touch LSQ (tip 8.0, riser->13)": (nom, touch_o),
        "current joint_mapping.py": (cur_s, cur_o),
        # Same 11 touches, CAD tip 9.45 cm, riser fixed 15. [scratch refit 2026-09-29]
        "T: touch, CAD tip, riser 15": (nom, o(-0.03886218, 0.11817732, -0.08640457, 1.51261340)),
        # Fit on the even uvc_60 episodes only (crotch@rim + wall-pinch xy); touch not used.
        # Scored here on everything -- read its grasp columns as ~half in-sample.
        "K: task poses (even eps)": (nom, o(0.01878756, 0.22986734, -0.02231385, 1.95185293)),
    }


def use(scale: dict, offset: dict) -> None:
    JM.SCALE_RAD_PER_UNIT.clear(); JM.SCALE_RAD_PER_UNIT.update(scale)
    JM.OFFSET_RAD.clear(); JM.OFFSET_RAD.update(offset)


def point(pos6, x_l5: float) -> tuple[np.ndarray, float]:
    """(x, y, z above the table) in cm of the gripper point at link5 +x = x_l5, and the pitch in deg
    (90 = straight down, >90 = tipped back past vertical; atan2, so 113 does not fold onto 67)."""
    q = JM.row_to_sim_rad([float(v) for v in pos6])[:5]
    t5 = fk.link5_transform(q)
    p = t5 @ np.array([x_l5, TCP_Y, 0.0, 1.0])
    v = t5[:3, 0]
    along = v[0] * math.cos(q[0]) + v[1] * math.sin(q[0])
    return np.array([p[0] * 100, p[1] * 100, p[2] * 100 + RISER_CM]), math.degrees(math.atan2(-v[2], along))


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
    episodes (ep0: argmin frame 403, 30 cm from the cup; the real grasp is frame 226)."""
    g = st[:, 5]
    thr = np.percentile(g, 95) - 0.6 * (np.percentile(g, 95) - np.percentile(g, 5))
    grasp = next((i for i in range(1, len(g) - hold)
                  if g[i - 1] >= thr > g[i] and np.all(g[i:i + hold] < thr)), None)
    opens = [i for i in range(1, len(g)) if g[i - 1] < thr <= g[i]]
    return grasp, (opens[-1] if opens else None)                # final opening = drop into the bin


def wall_residual(p_xy: np.ndarray, cup_xy) -> tuple[float, float, float]:
    """(radial, tangential-after-wall, raw tangential) in cm. The wall side is the one the pinch
    point is on, judged around the -2.5 cm common offset (see docstring)."""
    t = np.array(cup_xy); u = t / np.linalg.norm(t); w = np.array([-u[1], u[0]])
    d = p_xy - t
    tn = float(d @ w)
    return float(d @ u), tn - math.copysign(R_CUP, tn + 2.5), tn


def main() -> int:
    touch = list(csv.DictReader(TOUCH_CSV.open(encoding="utf-8")))
    place = {r["placement_id"]: (float(r["x_cm"]), float(r["y_cm"]))
             for r in csv.DictReader(PLACEMENTS.open(encoding="utf-8"))}
    eps = load_uvc60()
    frames = {e: grasp_and_release(st) for e, st in eps.items()}
    bin_xy = np.array([S.BIN_CENTER_X, S.BIN_CENTER_Y]) * 100
    bin_r, bin_rim = S.BIN_OPENING_DIA / 2 * 100, (S.ARM_RISER_HEIGHT + S.BIN_HEIGHT) * 100
    saved = (dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD))

    print(f"touch {len(touch)} pts (CAD tip {X_TIP*100:.2f} cm) | uvc_60 {len(eps)} eps, grasp scored on ep2-59 | "
          f"cup rim {CUP_RIM:.1f} cm, R {R_CUP} | bin ({bin_xy[0]:.1f}, {bin_xy[1]:.1f}) r={bin_r:.0f}\n")
    hdr = (f"{'candidate':<32}| {'TOUCH 3D':>8} {'z':>5} {'pitch':>6} | {'crotch-rim':>10} {'ok':>4} {'wall-xy':>7} "
           f"{'common':>6} | {'RELEASE in-bin':>14} {'d':>5}")
    print(hdr); print("-" * len(hdr))

    for name, (sc, of) in candidates().items():
        use(sc, of)
        t3, tz, tp = [], [], []
        for r in touch:
            p, a = point([r[f"state_{j}"] for j in JOINTS], X_TIP)
            t3.append(np.linalg.norm(p - (float(r["target_x_cm"]), float(r["target_y_cm"]), 0.0)))
            tz.append(p[2]); tp.append(a)

        cr, ok, wx, com, rd = [], [], [], [], []
        for e, st in eps.items():
            gi, ri = frames[e]
            if gi is not None and e not in TUNED_EPISODES:
                zc = point(st[gi], X_CROTCH)[0][2]
                pp = point(st[gi], X_PINCH)[0]
                rad, tan, raw = wall_residual(pp[:2], place[placement_of(e)])
                cr.append(zc - CUP_RIM)
                ok.append(pp[2] <= CUP_RIM + 1.0 and zc >= CUP_RIM - 1.0)     # 1 cm tolerance each side
                wx.append(math.hypot(rad, tan)); com.append(tan)
            if ri is not None:
                rd.append(np.linalg.norm(point(st[ri], X_PINCH)[0][:2] - bin_xy))

        med = lambda a: float(np.median(a))
        print(f"{name:<32}| {med(t3):7.1f}c {med(tz):+4.1f}c {med(tp):5.1f}° | {med(cr):+9.1f}c {np.mean(ok):4.0%} "
              f"{med(wx):6.1f}c {med(com):+5.1f}c | {sum(d <= bin_r for d in rd):>6}/{len(rd):<3}     {med(rd):4.1f}c")
    use(*saved)

    print("\nhow to read: TOUCH truth z=0, pitch 90. crotch-rim ~0 for most grasps ([Eric說]); 'ok' = pinch z <= rim <= crotch z")
    print("(1 cm tolerance). wall-xy = pinch point vs the cup wall (+-R), 'common' = the unexplained shared tangential offset.")
    print("No row wins both TOUCH and crotch-rim -- see the docstring before picking one.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
