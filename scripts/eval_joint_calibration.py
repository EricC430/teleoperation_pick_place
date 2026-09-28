#!/usr/bin/env python
"""Score candidate joint-mapping constant sets against every PHYSICAL ground truth we have.

No camera anywhere in this: recorded joint readings -> `joint_mapping` -> FK -> fingertip, compared
with positions known from the physical cell. That is deliberate. Until the front-left camera's pose
is ArUco-calibrated (S4 §5-5 T2), judging calibration by eye on a render mixes camera error into
arm error -- and tuning the arm until the render looks right writes camera error into joint
constants.

Three ground truths, three different kinds of arm pose:

  touch    calibration/2026-09-22_touch_calibration.csv. 11 points where the fingertip physically
           touched a known mat coordinate with the gripper held VERTICAL. Torque off, arm
           supported by hand. Truth: (x, y, 0) and pitch 90 deg.
  grasp    uvc_60, each episode's first sustained gripper close. Torque on, arm carrying itself.
           Truth: the cup's placement (x, y). Height has no exact truth -- it is reported against
           the 9.5 cm rim for reference only. Episodes 0 and 1 were used to hand-tune and to
           eyeball-check the current constants, so they are reported separately and kept out of
           the held-out score.
           🔴 [已查證 2026-09-29] grasp xy does NOT currently measure joint calibration: its residual
           is a ~10 deg azimuth offset that depends on the placement angle (about -12 deg for
           theta > -25, about +3 deg for theta < -30), while touch points at the SAME theta agree
           within +-3 deg. Approach direction does not change it (CW -10.4 / CCW -10.0 deg), so it
           is not pan backlash. Something differed between the 9/13 recording and the 9/22 touch
           session (mat origin/0-deg line vs the arm is the [推論] suspect). Fitting pan to it drags
           pan ~7 deg off what touch says. Read 'g-az' before reading 'GRASP xy'.
  release  uvc_60, the frame of the episode's FINAL gripper opening. Arm raised over the bin.
           Truth: inside the bin opening, above its rim. Only usable if the releases actually
           cluster over the bin -- the script prints that check and says so if they do not.

    python3 scripts/eval_joint_calibration.py

A candidate that improves grasp while making touch worse has not been calibrated. The summary flags
that -- and, while g-az stays ~-10 deg, it also says grasp xy is not usable evidence either way.
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
sys.path.insert(0, str(_REPO / "scripts"))

import joint_mapping as JM  # noqa: E402
import scene_constants as S  # noqa: E402
import touch_calibrate as TC  # noqa: E402
from reach_logger import fk  # noqa: E402

TOUCH_CSV = _REPO / "calibration/2026-09-22_touch_calibration.csv"
UVC60 = sorted((_REPO / "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/data").glob("chunk-*/file-*.parquet"))
# [Eric說 2026-09-21] uvc_60 walked campA_136sym's t1..t60 in order, BUT [已查證 2026-09-29]
# docs/meeting/2026-09-13.md item 4: "t41 重錄後排在最後一集" -> ep0-39 = t1-t40, ep40-58 = t42-t60,
# ep59 = t41. Reading it as plain i -> t{i+1} put ep40-58 one placement off (23-52 cm "errors").
# The per-episode record the S6 notes used (episode_meta/..._paper_cup.csv) was never committed.
PLACEMENTS = _REPO / "configs/placements/campA_136sym_20260908_20260908_train.csv"
TUNED_EPISODES = {0, 1}


def placement_of(ep: int) -> str:
    return f"train_{41 if ep == 59 else ep + 1 if ep < 40 else ep + 2:03d}"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
NOMINAL = 0.03141593   # 1.80 deg/unit: 4096 ticks / 200 units


def candidates() -> dict[str, tuple[dict, dict]]:
    cur_s, cur_o = dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD)
    touch_o = dict(cur_o)
    touch_o["shoulder_lift"] -= math.radians(2.0)    # undo the 2026-09 hand-tune
    touch_o["wrist_flex"] -= math.radians(24.0)
    return {
        "S6 two-pose (4248342)": (
            {"shoulder_pan": 0.03104445, "shoulder_lift": 0.03143896, "elbow_flex": 0.03248061,
             "wrist_flex": 0.03159506, "wrist_roll": 0.03086570},
            {"shoulder_pan": 0.01592023, "shoulder_lift": -0.36919370, "elbow_flex": 0.44010560,
             "wrist_flex": 1.62968550, "wrist_roll": -0.02788842}),
        "touch LSQ (before hand-tune)": ({k: NOMINAL for k in cur_s}, touch_o),
        # Same 11 touch points, same objective, but riser FIXED at the measured 15 cm. `touch_calibrate
        # solve` leaves the riser free with a prior at 14 cm and landed on 13.05 cm -- the committed
        # offsets above were solved for a riser 2 cm lower than the real one. Identical touch residual
        # (1.52 cm) with lift/elbow 7.1/4.4 deg different: vertical-gripper touches at one height
        # barely constrain riser vs lift vs elbow. [已查證 2026-09-29, scratch refit]
        "touch LSQ, riser 15 fixed": ({k: NOMINAL for k in cur_s},
                                      {"shoulder_pan": -0.03881857, "shoulder_lift": 0.21252320,
                                       "elbow_flex": -0.15140055, "wrist_flex": 1.44494233,
                                       "wrist_roll": cur_o["wrist_roll"]}),
        "current joint_mapping.py": (cur_s, cur_o),
    }


def use(scale: dict, offset: dict) -> None:
    JM.SCALE_RAD_PER_UNIT.clear(); JM.SCALE_RAD_PER_UNIT.update(scale)
    JM.OFFSET_RAD.clear(); JM.OFFSET_RAD.update(offset)


def q5(pos6) -> list[float]:
    return JM.row_to_sim_rad([float(v) for v in pos6])[:5]


def pitch_deg(q) -> float:
    """90 = straight down; >90 = tipped back past vertical. Unlike touch_calibrate's asin-based
    version this does not fold 113 deg onto 67 deg -- that fold hid the result once already."""
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


def main() -> int:
    touch = list(csv.DictReader(TOUCH_CSV.open(encoding="utf-8")))
    place = {r["placement_id"]: (float(r["x_cm"]), float(r["y_cm"]))
             for r in csv.DictReader(PLACEMENTS.open(encoding="utf-8"))}
    eps = load_uvc60()
    bin_xy = np.array([S.BIN_CENTER_X, S.BIN_CENTER_Y]) * 100
    bin_r, rim_z = S.BIN_OPENING_DIA / 2 * 100, (S.ARM_RISER_HEIGHT + S.BIN_HEIGHT) * 100
    cup_rim = S.CUP_HEIGHT * 100
    saved = (dict(JM.SCALE_RAD_PER_UNIT), dict(JM.OFFSET_RAD))

    print(f"touch {len(touch)} pts | uvc_60 {len(eps)} episodes, grasp scored on ep2-59 | "
          f"bin centre ({bin_xy[0]:.1f}, {bin_xy[1]:.1f}) cm r={bin_r:.0f} rim {rim_z:.0f} cm | cup rim {cup_rim:.1f} cm\n")
    hdr = (f"{'candidate':<30}| {'TOUCH 3D':>8} {'z':>6} {'pitch':>6} | {'g-az':>6} {'GRASP xy':>8} {'z':>6} "
           f"{'ep0/1 xy':>9} | {'RELEASE in-bin':>14} {'d':>6} {'z-rim':>6}")
    print(hdr); print("-" * len(hdr))

    results = {}
    for name, (sc, of) in candidates().items():
        use(sc, of)
        t3, tz, tp = [], [], []
        for r in touch:
            q = q5([r[f"state_{j}"] for j in JOINTS])
            x, y, z = TC.fingertip_above_table_cm(q)
            t3.append(math.dist((x, y, z), (float(r["target_x_cm"]), float(r["target_y_cm"]), 0.0)))
            tz.append(z); tp.append(pitch_deg(q))

        gxy, gz, gaz, tuned, rd, rz = [], [], [], [], [], []
        for e, st in eps.items():
            gi, ri = grasp_and_release(st)
            if gi is not None:
                x, y, z = TC.fingertip_above_table_cm(q5(st[gi]))
                px, py = place[placement_of(e)]
                d = math.dist((x, y), (px, py))
                if e in TUNED_EPISODES:
                    tuned.append(d)
                else:
                    gxy.append(d); gz.append(z)
                    gaz.append(math.degrees(math.atan2(y, x) - math.atan2(py, px)))
            if ri is not None:
                x, y, z = TC.fingertip_above_table_cm(q5(st[ri]))
                rd.append(math.dist((x, y), bin_xy)); rz.append(z - rim_z)

        med = lambda a: float(np.median(a))
        inbin = sum(d <= bin_r for d in rd)
        results[name] = dict(touch=med(t3), grasp=med(gxy))
        print(f"{name:<30}| {med(t3):7.1f}c {med(tz):+5.1f}c {med(tp):5.1f}° | {med(gaz):+5.1f}° {med(gxy):7.1f}c {med(gz):5.1f}c "
              f"{'/'.join(f'{v:.1f}' for v in tuned):>9} | {inbin:>6}/{len(rd):<3}     {med(rd):5.1f}c {med(rz):+5.1f}c")
    use(*saved)

    print("\nhow to read: TOUCH truth is z=0 and pitch 90 deg. g-az is the grasp azimuth error (see docstring: a")
    print("session-frame offset, not calibration). GRASP xy is distance to the cup; its z has no exact")
    print("truth (rim is %.1f cm). RELEASE 'd' is distance from the bin axis (inside if <= %.0f), z-rim > 0 = above rim." % (cup_rim, bin_r))

    names = list(results)
    base, cur = results[names[1]], results[names[-1]]
    print()
    if cur["grasp"] < base["grasp"] and cur["touch"] > base["touch"]:
        print(f"🔴 '{names[-1]}' vs '{names[1]}': grasp {base['grasp']:.1f}->{cur['grasp']:.1f} cm better, "
              f"touch {base['touch']:.1f}->{cur['touch']:.1f} cm WORSE.")
        print("   But grasp xy carries the ~10 deg session-frame offset (g-az), so a grasp-xy gain is not calibration")
        print("   evidence. Judge candidates on TOUCH and RELEASE until the 9/13 mat frame is resolved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
