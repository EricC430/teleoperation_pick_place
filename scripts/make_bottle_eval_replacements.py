#!/usr/bin/env python
"""Replacement points for plastic-bottle closed-loop eval points that are out of the arm's reach (2026-11-13).

Rule (the one c37 was made with, docs/worklog/2026-10-22.md §2): same theta as the removed point, the largest paper
r <= R_MAX (37.6 cm = the 10/21 teleop limit 38.3 minus its 0.7 cm uncertainty), at least D_MIN (2.0 cm) from every
train t1-t70, every eval-open and every eval-close point still in the bottle list (and from the other new points).
Search r downward in 0.01 cm steps.

Input list: configs/placements/bottle_eval-close_20261022.csv (c1-c36 without c12, plus c37).
Writes configs/placements/bottle_eval-close_<date>.csv (input minus the removed ids, plus the new ones, `replaces`
column filled), episode_meta/templates/plastic_bottle_A1_<tag>.csv (episode order: remaining ids by number, then the
new ones), and a scatter png. Prints each new point and its nearest neighbours.

    python scripts/make_bottle_eval_replacements.py --remove eval-close_006,eval-close_026 --date 20261113 --tag v3
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
PL = REPO / "configs/placements"
CAMP = PL / "campA_136sym_20260908_20260908_"
R_MAX, D_MIN, STEP = 37.6, 2.0, 0.01


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--base", default=str(PL / "bottle_eval-close_20261022.csv"))
    p.add_argument("--remove", required=True, help="comma-separated eval-close ids to replace")
    p.add_argument("--date", required=True)
    p.add_argument("--tag", required=True, help="template suffix, e.g. v3")
    a = p.parse_args()
    base = pd.read_csv(a.base)
    remove = a.remove.split(",")
    missing = [r for r in remove if r not in set(base.placement_id)]
    if missing:
        raise SystemExit(f"not in {a.base}: {missing}")
    keep = base[~base.placement_id.isin(remove)].copy()
    others = pd.concat([pd.read_csv(f"{CAMP}train.csv"), pd.read_csv(f"{CAMP}eval-open.csv"), keep])[["placement_id", "x_cm", "y_cm"]]
    nxt = max(int(s.split("_")[1]) for s in base.placement_id) + 1
    new = []
    for rid in remove:
        th = float(base.set_index("placement_id").loc[rid, "theta_deg"])
        pts = pd.concat([others, pd.DataFrame(new)[["placement_id", "x_cm", "y_cm"]]]) if new else others
        xy = pts[["x_cm", "y_cm"]].values
        found = None
        for r in np.arange(R_MAX, 10, -STEP):
            q = np.array([r * np.cos(np.deg2rad(th)), r * np.sin(np.deg2rad(th))])
            dist = np.hypot(*(xy - q).T)
            if dist.min() >= D_MIN:
                found = (round(r, 2), q, dist); break
        if found is None:
            raise SystemExit(f"no point for {rid}")
        r, q, dist = found
        pid = f"eval-close_{nxt:03d}"; nxt += 1
        near = pts.assign(d=dist).nsmallest(3, "d")
        print(f"{pid} replaces {rid}: r {r:.2f}  theta {th:.2f}  x {q[0]:.2f} y {q[1]:.2f}  nearest: "
              + ", ".join(f"{n.placement_id} {n.d:.2f}" for n in near.itertuples()))
        new.append(dict(placement_id=pid, r_cm=r, theta_deg=round(th, 2), x_cm=round(q[0], 2), y_cm=round(q[1], 2), replaces=rid))
    out = pd.concat([keep, pd.DataFrame(new)], ignore_index=True)
    f = PL / f"bottle_eval-close_{a.date}.csv"
    out.to_csv(f, index=False)
    tpl_src = REPO / "episode_meta/templates/plastic_bottle_A1_v2.csv"
    tpl = pd.read_csv(tpl_src, encoding="utf-8-sig", dtype=str)
    short = lambda s: "c" + str(int(s.split("_")[1]))
    rows = [short(s) for s in out.placement_id]
    rows = sorted(rows[:len(keep)], key=lambda s: int(s[1:])) + rows[len(keep):]
    t = pd.DataFrame([dict(tpl.iloc[0]) for _ in rows])
    t["episode_index"] = range(len(rows)); t["placement_id"] = rows
    tf = REPO / f"episode_meta/templates/plastic_bottle_A1_{a.tag}.csv"
    t.to_csv(tf, index=False, encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(7, 7))
    for lab, d_, c in [("train", pd.read_csv(f"{CAMP}train.csv"), "0.7"), ("eval-open", pd.read_csv(f"{CAMP}eval-open.csv"), "tab:orange"),
                       ("eval-close kept", keep, "tab:blue"), ("removed", base[base.placement_id.isin(remove)], "tab:red"),
                       ("new", pd.DataFrame(new), "tab:green")]:
        ax.scatter(d_.y_cm, d_.x_cm, s=18, c=c, label=lab)
    for n in new:
        ax.annotate(short(n["placement_id"]), (n["y_cm"], n["x_cm"]))
    ax.set_xlabel("y [cm] (left +)"); ax.set_ylabel("x [cm] (forward)"); ax.invert_xaxis(); ax.set_aspect("equal"); ax.legend(fontsize=8)
    fig.savefig(PL / f"bottle_eval-close_{a.date}_scatter.png", dpi=110)
    print(f"list: {f} ({len(out)} points)\ntemplate: {tf} ({len(rows)} episodes)")


if __name__ == "__main__":
    main()
