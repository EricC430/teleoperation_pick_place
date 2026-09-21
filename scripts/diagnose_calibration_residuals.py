#!/usr/bin/env python
"""Localise WHERE the joint calibration is wrong, instead of eyeballing renders.

Looking at a sim render next to the real video conflates at least three error sources -- the joint
zero offsets, the camera pose (`scene_constants.py` is PLACEHOLDER, S5 gap 4), and the stand-in
cylinder cup. A person can say "that doesn't look right"; the picture cannot say which of the three
is wrong. `[柏宇說 2026-09-21]` after looking at 甲/乙/丁: 「需要更多方法才能知道哪裡出錯」.

This is one such method, and it never renders anything. At each episode's grasp frame the real
fingertip MUST be at that episode's cup -- the cup position is known, it is the seeded placement
that episode was recorded against. So the residual

    fingertip(FK, this calibration) - cup centre

is an error signal with no camera in it at all. 60 episodes span a range of reach distances and
azimuths, and HOW the residual varies across them says which parameter is wrong:

    radial residual, flat vs reach distance   -> an OFFSET is wrong (a constant push in/out)
    radial residual, sloped vs reach distance -> a SCALE is wrong (error grows with extension)
    tangential residual, sloped vs azimuth    -> shoulder_pan's zero or scale
    tangential residual, flat and non-zero    -> a fixed yaw offset, or the cup x/y convention
    vertical residual, flat vs reach distance -> shoulder_lift / elbow OFFSET
    vertical residual, sloped                 -> shoulder_lift / elbow SCALE

That distinction is the point: S6 section 4-a's sweep fits OFFSETS to assumed targets, and it can
always find some offset that minimises the error. It cannot tell you that no offset should have
been fitted because the real fault is a scale. A slope can.

    uv run python scripts/diagnose_calibration_residuals.py
    uv run python scripts/diagnose_calibration_residuals.py --offset-delta-deg shoulder_lift=+18.2,wrist_flex=+7.1

🔴 What this does NOT settle. The grasp frame is a DETECTOR (each episode's most closed gripper
frame), not a label; S5 section 2 records that it picks a non-grasp frame on a few episodes, so
read the median and the slope, not any single episode. And "the fingertip is at the cup centre" is
itself an idealisation -- a real grasp holds the near rim, so a constant radial residual of about
the cup's radius is expected and is NOT evidence of a calibration error. The slopes are the part
that does not depend on that assumption.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "sim"))

import joint_mapping as JM          # noqa: E402
import scene_constants as SC        # noqa: E402
from reach_logger import fk         # noqa: E402

DEFAULT_DATASET = os.path.join(
    _REPO, "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60")
DEFAULT_PLACEMENTS = os.path.join(
    _REPO, "docs/assets/placement_label_map_campA_136sym_20260908.csv")
# omx_constants.TCP_IN_LINK5_M -- the MEASURED fingertip, not link6/link7's pivots.
TCP_IN_LINK5 = np.array([0.08, -0.00165, 0.0, 1.0])
# The placement CSV gives each cup BOTH ways: from the pan axis and in the printed mat's own
# frame. The two differ by this much in x, which is the size of the frame error to look for.
MAT_PAN_DX_CM = 7.13


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", default=DEFAULT_DATASET)
    p.add_argument("--placements", default=DEFAULT_PLACEMENTS)
    p.add_argument("--offset-delta-deg", default="",
                   help="'shoulder_lift=+18.2,wrist_flex=+7.1' -- evaluate a candidate calibration "
                        "without touching joint_mapping.py's constants")
    p.add_argument("--scale-mult", default="",
                   help="'shoulder_lift=1.20' -- multiply SCALE_RAD_PER_UNIT for this run only. "
                        "The vertical slope is the thing to drive to zero with it, and unlike the "
                        "residual's size, zero slope is not an assumption -- correct kinematics "
                        "cannot make the error grow with reach.")
    p.add_argument("--grasp-detector", default="first-close",
                   choices=["first-close", "most-closed"],
                   help="how the grasp frame is picked; 'most-closed' reproduces what S5 section 2 "
                        "and S6 section 4-a used, and is wrong on 15 of 60 episodes")
    p.add_argument("--drop-outliers", type=float, default=15.0, metavar="CM",
                   help="exclude episodes whose horizontal residual exceeds this before fitting "
                        "trends; the detector still fails on some episodes and one 53 cm episode "
                        "moves a slope more than fifty good ones do. 0 disables.")
    p.add_argument("--pan-axis-origin", action="store_true",
                   help="read the placement CSV's x_pan_cm as measured FROM THE PAN AXIS (what the "
                        "column name says) instead of from the arm base, which is what "
                        "replay_render_episode.py currently assumes. The two differ by 1.1 cm.")
    return p.parse_args()


def pick_grasp_frame(gripper, detector, close_frac=0.5):
    """Index of the grasp frame within one episode's gripper channel.

    🔴 `most-closed` -- the whole episode's argmin -- is the detector S5 section 2 and S6 section
    4-a were computed with, and it is WRONG on a quarter of uvc_60: a pick-and-place squeezes twice
    (grasp, then again while releasing over the bin), and the second squeeze is often the tighter
    one, so the argmin lands at the BIN. Episode 0 is the clearest case: argmin puts the fingertip
    27 cm from the cup, while the first closing puts it 3.6 cm away, at frame 222 -- inside the
    f208-236 window where the sim's own rendered gripper is in contact with the cup.

    `first-close` takes the FIRST frame under the same threshold `grasp_attach.calibrate()` uses
    (the trace's own 5th/95th percentiles, `close_frac` of the way down). Non-circular: it never
    looks at where the arm is, so it cannot be tuned into agreeing with a calibration.
    """
    if detector == "most-closed":
        return int(np.argmin(gripper))
    g_open, g_closed = np.percentile(gripper, 95), np.percentile(gripper, 5)
    below = np.where(gripper < g_open - close_frac * (g_open - g_closed))[0]
    return int(below[0]) if len(below) else int(np.argmin(gripper))


def grasp_frames(dataset_root, detector="first-close"):
    """{episode: observation.state at its grasp frame}."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(dataset_root, "data", "chunk-*", "file-*.parquet")))
    if not files:
        sys.exit(f"no parquet under {dataset_root}/data/chunk-*/file-*.parquet")
    eps, states = [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "observation.state"]).to_pydict()
        eps.extend(t["episode_index"])
        states.extend(t["observation.state"])
    eps = np.asarray(eps)
    states = np.asarray(states, dtype=float)
    out = {}
    for e in np.unique(eps):
        block = states[eps == e]
        out[int(e)] = block[pick_grasp_frame(block[:, 5], detector)]   # channel 5 = gripper
    return out


def cup_centres(placements_csv, pan_axis_origin):
    """{short_id 't7' -> (x, y, z) of the cup centre in the arm's base frame}."""
    out = {}
    x0 = fk._PAN_AXIS_XY[0] if pan_axis_origin else 0.0
    with open(placements_csv, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out[row["short_id"]] = np.array([
                x0 + float(row["x_pan_cm"]) / 100.0,
                float(row["y_pan_cm"]) / 100.0,
                SC.TABLE_TOP_Z + SC.CUP_HEIGHT / 2.0 - (SC.TABLE_TOP_Z + SC.ARM_RISER_HEIGHT),
            ])
    return out


def apply_scales(spec):
    applied = {}
    for entry in spec.split(","):
        if not entry.strip():
            continue
        name, val = entry.split("=")
        name = name.strip()
        if name not in JM.SCALE_RAD_PER_UNIT:
            sys.exit(f"--scale-mult: {name!r} is not one of {list(JM.SCALE_RAD_PER_UNIT)}")
        JM.SCALE_RAD_PER_UNIT[name] *= float(val)
        applied[name] = float(val)
    return applied


def apply_deltas(spec):
    applied = {}
    for entry in spec.split(","):
        if not entry.strip():
            continue
        name, val = entry.split("=")
        name = name.strip()
        if name not in JM.OFFSET_RAD:
            sys.exit(f"--offset-delta-deg: {name!r} is not one of {list(JM.OFFSET_RAD)}")
        JM.OFFSET_RAD[name] += math.radians(float(val))
        applied[name] = float(val)
    return applied


def fit(x, y):
    """slope, intercept, pearson r -- plain least squares, no scipy."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    slope, intercept = np.polyfit(x, y, 1)
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r


def main():
    args = parse_args()
    applied = apply_deltas(args.offset_delta_deg) if args.offset_delta_deg else {}
    scaled = apply_scales(args.scale_mult) if args.scale_mult else {}
    if scaled:
        print(f"SCALE_RAD_PER_UNIT multiplied by {scaled} for this run only")
    if applied:
        print(f"calibration under test: committed OFFSET_RAD + {applied} deg")
    else:
        print("calibration under test: the committed OFFSET_RAD (甲)")
    print("cup x from %s\n" % ("the PAN AXIS" if args.pan_axis_origin else
                               "the ARM BASE (what the sim does today)"))

    states = grasp_frames(args.dataset_root, args.grasp_detector)
    cups = cup_centres(args.placements, args.pan_axis_origin)
    pan = np.array([fk._PAN_AXIS_XY[0], fk._PAN_AXIS_XY[1]])

    rows, xy_rows = [], []
    for ep in sorted(states):
        cup = cups.get("t%d" % (ep + 1))                     # uvc_60 walked t1..t60 in order
        if cup is None:
            continue
        q = JM.lerobot_to_urdf_rad(np.asarray(states[ep], dtype=float))[:5]
        tip = (fk.link5_transform(q) @ TCP_IN_LINK5)[:3]

        radial_dir = cup[:2] - pan
        reach = float(np.linalg.norm(radial_dir))            # cup's distance from the pan axis
        radial_dir = radial_dir / reach
        tangent_dir = np.array([-radial_dir[1], radial_dir[0]])

        d = tip - cup
        xy_rows.append((float(cup[0] - pan[0]) * 100, float(cup[1] - pan[1]) * 100,
                        float(d[0]) * 100, float(d[1]) * 100))
        rows.append((ep,
                     reach * 100,
                     math.degrees(math.atan2(cup[1] - pan[1], cup[0] - pan[0])),
                     float(d[:2] @ radial_dir) * 100,
                     float(d[:2] @ tangent_dir) * 100,
                     float(d[2]) * 100))

    if not rows:
        sys.exit("no episode matched a placement -- check --placements")
    n_all = len(rows)
    if args.drop_outliers > 0:
        keep = [i for i, r in enumerate(xy_rows)
                if math.hypot(r[2], r[3]) <= args.drop_outliers]
        dropped = [rows[i][0] for i in range(n_all) if i not in set(keep)]
        rows = [rows[i] for i in keep]
        xy_rows = [xy_rows[i] for i in keep]
        print("detector %r; %d/%d episodes kept, dropped (horizontal residual > %.0f cm): %s\n"
              % (args.grasp_detector, len(rows), n_all, args.drop_outliers,
                 dropped if dropped else "none"))
    arr = np.array([r[1:] for r in rows])
    reach, azim, rad, tan, vert = (arr[:, i] for i in range(5))

    print("%d episodes, reach %.1f-%.1f cm, azimuth %.0f..%.0f deg"
          % (len(rows), reach.min(), reach.max(), azim.min(), azim.max()))
    print("\nresidual = fingertip - cup centre, cm      median    IQR            "
          "vs reach: slope     r")
    for name, series, against in (("radial  (+ = past the cup)", rad, reach),
                                  ("tangential (+ = left)    ", tan, azim),
                                  ("vertical (+ = above)     ", vert, reach)):
        q1, q3 = np.percentile(series, [25, 75])
        slope, _, r = fit(against, series)
        unit = "cm/cm" if against is reach else "cm/deg"
        print("  %s  %+6.1f   %+5.1f..%+5.1f   %+8.3f %s  %+.2f"
              % (name, np.median(series), q1, q3, slope, unit, r))

    print("\nreading it (thresholds are judgement, not measured):")
    srad, _, rrad = fit(reach, rad)
    svert, _, rvert = fit(reach, vert)
    stan, _, rtan = fit(azim, tan)
    for label, slope, r, culprit_flat, culprit_sloped in (
            ("radial", srad, rrad, "an OFFSET (constant in/out)", "a SCALE (grows with reach)"),
            ("vertical", svert, rvert, "shoulder_lift/elbow OFFSET", "shoulder_lift/elbow SCALE"),
            ("tangential", stan, rtan, "a fixed yaw, or the cup x/y convention",
             "shoulder_pan zero or scale")):
        verdict = culprit_sloped if abs(r) > 0.5 else culprit_flat
        print("  %-11s r=%+.2f -> %s" % (label, r, verdict))
    # A tangential error of t cm at reach R cm IS a yaw error of atan(t/R). Converting makes the
    # two candidates separable: a zero error is a CONSTANT yaw, a scale error is a yaw PROPORTIONAL
    # to how far the cup is off centre. Leaving it in cm cannot tell them apart, because reach and
    # azimuth are not independent across the placement set.
    yaw_err = np.degrees(np.arctan2(tan, reach))
    syaw, iyaw, ryaw = fit(azim, yaw_err)
    print("\nshoulder_pan, the same residual as an ANGLE (yaw error = atan(tangential / reach)):")
    print("  median %+.1f deg, IQR %+.1f..%+.1f" % (np.median(yaw_err),
                                                    *np.percentile(yaw_err, [25, 75])))
    print("  vs the cup's azimuth: slope %+.3f deg/deg, intercept %+.1f deg, r %+.2f"
          % (syaw, iyaw, ryaw))
    if abs(ryaw) > 0.5 and abs(syaw) > 0.05:
        print("  -> PROPORTIONAL to azimuth: that is a shoulder_pan SCALE error of %+.1f%%, and no "
              "offset\n     on any other joint can fix it. Implied true scale = %.4f x the "
              "committed one." % (syaw * 100, 1.0 / (1.0 + syaw)))
    elif abs(np.median(yaw_err)) > 2.0:
        print("  -> roughly CONSTANT: a shoulder_pan ZERO error of about %+.1f deg."
              % np.median(yaw_err))
    else:
        print("  -> neither constant nor proportional stands out; shoulder_pan is not the suspect.")

    # ------------------------------------------------------------------ translation vs rotation
    # 🔴 The block above can be FOOLED, and was: a fixed translation between the frame the cup
    # coordinates are expressed in and the arm's actual pan axis also produces a tangential
    # residual that varies with azimuth, because the tangential direction itself rotates with
    # azimuth. Read alone it looks like a pan scale error. Fit both at once and the two separate:
    #     residual_xy  ~=  t  +  delta * J @ (cup - pan)        J = [[0,-1],[1,0]], small delta
    # t is a frame/convention error (someone's origin is in the wrong place, no joint is at fault);
    # delta is a genuine yaw error. Whichever shrinks the residual is the one to chase.
    P = np.array([[r[0] for r in xy_rows], [r[1] for r in xy_rows]]).T      # cup - pan, cm
    R = np.array([[r[2] for r in xy_rows], [r[3] for r in xy_rows]]).T      # tip - cup, cm
    A = np.zeros((2 * len(P), 3))
    A[0::2, 0] = 1.0
    A[1::2, 1] = 1.0
    A[0::2, 2] = -P[:, 1]
    A[1::2, 2] = P[:, 0]
    sol, *_ = np.linalg.lstsq(A, R.reshape(-1), rcond=None)
    tx, ty, delta = sol
    rms = lambda v: float(np.sqrt(np.mean(np.sum(v ** 2, axis=1))))
    both = R - (A @ sol).reshape(-1, 2)
    only_t = R - np.array([tx, ty])
    only_d = R - (A[:, 2:] @ sol[2:]).reshape(-1, 2)
    print("\nsplitting the horizontal residual into a frame shift and a yaw error:")
    print("  raw residual RMS                              %.1f cm" % rms(R))
    print("  after a pure TRANSLATION  (%+.1f, %+.1f) cm     %.1f cm" % (tx, ty, rms(only_t)))
    print("  after a pure YAW          %+.1f deg              %.1f cm"
          % (math.degrees(delta), rms(only_d)))
    print("  after both                                    %.1f cm" % rms(both))
    if rms(only_t) < rms(only_d):
        print("  -> TRANSLATION explains more. The suspect is the cup/base frame convention, NOT a\n"
              "     joint. Note the placement CSV carries x_pan_cm AND x_mat_cm, %0.2f cm apart."
              % MAT_PAN_DX_CM)
    else:
        print("  -> YAW explains more; shoulder_pan stays a suspect.")

    print("\n⚠️  A constant radial residual near the cup's radius (%.1f cm) is EXPECTED -- a grasp "
          "holds the rim,\n    not the centre. The slopes are the part that does not rest on that "
          "assumption." % (SC.CUP_MEAN_DIA / 2 * 100))


if __name__ == "__main__":
    main()
