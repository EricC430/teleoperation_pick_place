"""Sim vs real outcome for the 10-14 closed-loop physics replay: the cup counts as grasped in sim when it is held
>= 5 cm above the table for >= 10 frames (the real arm lifts it ~40 cm to the bin)."""
import json
import sys
from pathlib import Path

import numpy as np

D = Path(sys.argv[1]); TAG = sys.argv[2] if len(sys.argv) > 2 else "all"
rows = []
for ed in sorted(p for p in D.glob("cl_ep*") if p.is_dir()):
    E = json.load(open(ed / "episode.json"))
    lg = ed / f"sim_fit_{TAG}" / "physics_log.json"
    if not lg.exists():
        continue
    z = np.array([r["cup_xyz"][2] for r in json.load(open(lg))["log"]])
    lifted = int((z > 0.05).sum())
    rows.append((E["episode"], E["placement"], E["outcome"], lifted >= 10, float(z.max()), lifted))
agree = sum((o == "success") == s for _, _, o, s, _, _ in rows)
print(f"episodes scored: {len(rows)}; sim agrees with real on {agree}/{len(rows)}")
for o in ("success", "no_grasp"):
    sub = [r for r in rows if r[2] == o]
    if sub:
        print(f"  real {o:9s}: {len(sub):2d} episodes, sim lifted the cup in {sum(r[3] for r in sub)}")
print("disagreements:")
for e, p, o, s, zm, n in rows:
    if (o == "success") != s:
        print(f"  ep{e:02d} {p:>4s} real {o:9s} sim {'lifted' if s else 'not lifted'} (max z {zm * 100:.1f} cm, {n} frames > 5 cm)")
