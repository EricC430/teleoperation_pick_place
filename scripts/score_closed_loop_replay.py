"""Sim vs real outcome for the 10-14 closed-loop physics replay, per close of the gripper.
    uv run python scripts/score_closed_loop_replay.py <replay dir> [<sim tag>]

A close = gripper action < 0.2 rad for >= 5 frames. The cup counts as grasped in sim when it is held >= 5 cm above
the table for >= 10 frames during the close (+1 s). Real: the policy re-opens and retries after a miss, so in a
success episode the last close is the grasp and every earlier one missed; in a no_grasp episode every close missed.
(The paper cup is crushed when grasped, so the gripper reading cannot tell a grasp from a miss: ~-0.02 vs -0.03 rad.)

Only the first close is compared as the headline number: after a miss the cup may have been pushed, and the recorded
joints after that react to where the real cup went, not to where the sim cup went.
"""
import json
import sys
from pathlib import Path

import numpy as np

CLOSED_RAD, MIN_CLOSE_FRAMES, LIFT_M, MIN_LIFT_FRAMES = 0.2, 5, 0.05, 10

D = Path(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else "all"


def closes(a: np.ndarray) -> list[tuple[int, int]]:
    ev, k = [], 0
    while k < len(a):
        if a[k] < CLOSED_RAD:
            j = k
            while j < len(a) and a[j] < CLOSED_RAD:
                j += 1
            if j - k >= MIN_CLOSE_FRAMES:
                ev.append((k, j))
            k = j
        else:
            k += 1
    return ev


rows = []
for ed in sorted(p for p in D.glob("cl_ep*") if p.is_dir()):
    E = json.load(open(ed / "episode.json"))
    lg = ed / f"sim_fit_{TAG}" / "physics_log.json"
    if not lg.exists():
        continue
    L = json.load(open(lg))["log"]
    fr = np.array([r["frame"] for r in L]); cup = np.array([r["cup_xyz"] for r in L])
    ev = closes(np.array(json.load(open(ed / "gripper.json"))["gripper_target_rad_from_action"]))
    real_ok = E["outcome"] == "success"
    att = []
    for i, (s, e) in enumerate(ev):
        m = (fr >= s) & (fr < e + E["fps"])
        pre = cup[fr <= s][-1, :2] if (fr <= s).any() else cup[0, :2]
        att.append({"t_s": s / E["fps"], "real": real_ok and i == len(ev) - 1,
                    "sim": bool((cup[m, 2] > LIFT_M).sum() >= MIN_LIFT_FRAMES),
                    "cup_moved_cm": float(np.linalg.norm(pre - cup[0, :2]) * 100)})
    rows.append((E["episode"], E["placement"], real_ok, att))

first = [(e, p, a[0]) for e, p, _, a in rows if a]
agree1 = sum(a["real"] == a["sim"] for _, _, a in first)
print(f"first close: sim agrees with real on {agree1}/{len(first)} episodes "
      f"({len(rows) - len(first)} episodes never closed the gripper)")
for real in (True, False):
    sub = [a for _, _, a in first if a["real"] == real]
    print(f"  real {'grasped' if real else 'missed '} on the first close: {len(sub):2d}, sim lifted the cup in {sum(a['sim'] for a in sub)}")
ep_agree = sum(ok == any(a["sim"] for a in att) for _, _, ok, att in rows)
print(f"whole episode (any close lifts the cup): {ep_agree}/{len(rows)} -- counts a sim grasp on a different close as agreement")
print("episodes with a disagreement on any close (later closes are not comparable once the cup has moved):")
for e, p, ok, att in rows:
    if any(a["real"] != a["sim"] for a in att):
        print(f"  ep{e:02d} {p:>4s} " + " ; ".join(
            f"#{i + 1} {a['t_s']:4.1f}s real {'grasp' if a['real'] else 'miss '} sim {'grasp' if a['sim'] else 'miss '} "
            f"(sim cup moved {a['cup_moved_cm']:.1f} cm before)" for i, a in enumerate(att)))
