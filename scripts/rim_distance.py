#!/usr/bin/env python
"""Closed loop: how far is the pinch point from the cup wall at the first sustained gripper close?

Arm model: the touch-pinned link-marker fit (calibration/2026-10-14_link_marker_fit.json: zeros, gravity sag, 6-dof
base), the same one sim/physics_replay.py uses. Held-out touch error ~0.4 cm out to 28 cm reach; NOT checked beyond
28 cm or with wrist_roll away from 0 (all 16 touch points at roll -2.9..-2.5).
Pinch point: link5 @ (GRIPPER_PINCH_X_M, GRIPPER_MIDLINE_Y_M, 0). Cup centre: the c point (eval-close list), radius at
the pinch height eval_joint_calibration.CUP_R_AT_PINCH = 3.4 cm. Grasp frame: fit_paper_to_pan.grasp_frame(st, "first")
(closed for 15 frames); episodes whose gripper reading spans < --min-depth never closed. Method:
docs/handoff/2026-10-14_rim_distance.md.

Per episode:
  rim     = |p - c| - R     signed: > 0 pinch point outside the wall circle, < 0 inside the cup
  d_rad   = (p - c) . u     u = unit vector base -> cup: > 0 past the cup centre, < 0 short of it
  d_tan   = (p - c) . w     w = u turned +90 deg (left, seen from the base)
  jaw     = |jaw axis . u|  link5 y (the axis the jaws close along) on the table: ~0 jaws close across the base->cup
            line (normal grasp, 10-14: roll 0-11), ~0.9 jaws close along it (turned wrist, 10-14: roll 24-44).
            Grasp type is split on jaw >= 0.5 instead of a wrist_roll threshold.
|rim| > --gross cm: the gripper closed somewhere else, not an aim error -- listed, kept out of the medians.

    uv run python scripts/rim_distance.py                      # 10-14 15:12 fldrop30 closed loop
    uv run python scripts/rim_distance.py <rollout dataset name>
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO))
import omx_constants as K  # noqa: E402
import touch_test as TT  # noqa: E402
import fit_paper_to_pan as F  # noqa: E402

CUP_R = F.EJC.CUP_R_AT_PINCH
PINCH = np.array([K.GRIPPER_PINCH_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0])
DATA = Path("D:/hf/lerobot/ericc430")
PLACE = REPO / "configs/placements/campA_136sym_20260908_20260908_eval-close.csv"
DEFAULT = "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208"


def rows(name: str, min_depth: float):
    m = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")
    place = {f"c{int(r['placement_id'].split('_')[1])}": np.array([float(r["x_cm"]), float(r["y_cm"])])
             for r in csv.DictReader(PLACE.open(encoding="utf-8"))}
    F.DATA = DATA
    states = F.load_states(name)
    for md in csv.DictReader((REPO / "episode_meta" / f"{name}.csv").open(encoding="utf-8-sig")):
        e, pid = int(md["episode_index"]), md["placement_id"].strip()
        out = {"ep": e, "pid": pid, "outcome": md["outcome"], "mech": md.get("mechanism", ""),
               "reach": float(np.linalg.norm(place[pid]))}
        if e not in states:
            yield out | {"skip": "no data"}
            continue
        g = states[e][:, 5]
        if np.percentile(g, 95) - np.percentile(g, 5) < min_depth:
            yield out | {"skip": "never closed"}
            continue
        gi = F.grasp_frame(states[e], "first")
        if gi is None:
            yield out | {"skip": "no sustained close"}
            continue
        st = states[e][gi]
        T5 = m.link5(st)
        p = (T5 @ PINCH)[:2] * 100
        c = place[pid]; u = c / np.linalg.norm(c); w = np.array([-u[1], u[0]]); d = p - c
        yield out | {"frame": gi, "roll": float(st[4]), "jaw": float(abs(T5[:2, 1] @ u)),
                     "rim": float(np.linalg.norm(d) - CUP_R), "d_rad": float(d @ u), "d_tan": float(d @ w)}


def summary(label, xs):
    if not xs:
        return f"{label}: n=0"
    a, s = np.abs(np.array(xs)), np.array(xs)
    return (f"{label}: n={len(a)} |rim| median {np.median(a):.2f} cm (IQR {np.percentile(a, 25):.2f}..{np.percentile(a, 75):.2f})"
            f", signed median {np.median(s):+.2f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default=DEFAULT)
    ap.add_argument("--min-depth", type=float, default=5.0, help="gripper reading span (p95 - p5) below this = never closed")
    ap.add_argument("--gross", type=float, default=10.0, help="|rim| above this = closed away from the cup")
    ap.add_argument("--fit-reach-cm", type=float, default=28.0, help="touch-checked reach of the fit")
    a = ap.parse_args()
    R = list(rows(a.dataset, a.min_depth))
    print(f"{a.dataset}\n{'ep':>3} {'pid':>4} {'outcome':>9} {'reach':>5}  {'roll':>5} {'jaw':>4} {'rim':>6} {'d_rad':>6} {'d_tan':>6}  mechanism")
    for r in R:
        if "skip" in r:
            print(f"{r['ep']:>3} {r['pid']:>4} {r['outcome']:>9} {r['reach']:5.1f}   -- {r['skip']}  {r['mech']}")
            continue
        flag = "*" if r["reach"] - CUP_R > a.fit_reach_cm else " "
        print(f"{r['ep']:>3} {r['pid']:>4} {r['outcome']:>9} {r['reach']:5.1f}{flag} {r['roll']:5.1f} {r['jaw']:4.2f} {r['rim']:+6.2f} "
              f"{r['d_rad']:+6.2f} {r['d_tan']:+6.2f}  {r['mech']}")
    ok = [r for r in R if "skip" not in r and abs(r["rim"]) <= a.gross]
    print(f"\n* = cup's near wall beyond the touch-checked {a.fit_reach_cm:.0f} cm (fit not verified there)")
    print(f"skipped: {[(r['ep'], r['skip']) for r in R if 'skip' in r]}")
    print(f"closed away from the cup (|rim| > {a.gross:.0f} cm): {[r['ep'] for r in R if 'skip' not in r and abs(r['rim']) > a.gross]}")
    for near in (True, False):
        g = [r for r in ok if (r["reach"] - CUP_R <= a.fit_reach_cm) == near]
        print(f"[{'near' if near else 'far '}]")
        for turned in (False, True):
            for oc in ("success", "fail"):
                s = [r["rim"] for r in g if (r["jaw"] >= 0.5) == turned and (r["outcome"] == "success") == (oc == "success")]
                print("  " + summary(f"{'turned' if turned else 'normal'} {oc:>7}", s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
