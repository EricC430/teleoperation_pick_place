"""Tables of docs/worklog/2026-10-31_wrist_cup_offset.md from the per-run JSONs next to this file."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu

H = Path(__file__).resolve().parent
RUNS = ["far_wrist_1008", "20261014_151208", "20261016_142101", "20261016_152018", "20261017_215410"]
ep = {t: json.load(open(H / f"{t}.json"))["episodes"] for t in RUNS}


def arr(t, drop0=False, key=lambda r: True):
    return np.array([[r["rad_med"], r["tan_med"], r["tilt_med"]] + r["dw_med"] for e, r in ep[t].items()
                     if not (drop0 and e == "0") and key(r)])


for t in RUNS:
    a = arr(t, drop0=t == RUNS[0])
    print(f"{t:16s} n={len(a):2d} radial median {np.median(a[:, 0]):+.2f} sd {a[:, 0].std():.2f} | "
          f"tangential median {np.median(a[:, 1]):+.2f} sd {a[:, 1].std():.2f} | tilt median {np.median(a[:, 2]):.0f} deg")
old = arr(RUNS[0], drop0=True); new = np.vstack([arr(t) for t in RUNS[1:]])
print(f"10/08 w/o ep0 n={len(old)} radial median {np.median(old[:, 0]):+.2f}; 10/14-17 n={len(new)} radial median "
      f"{np.median(new[:, 0]):+.2f}; Mann-Whitney p={mannwhitneyu(old[:, 0], new[:, 0]).pvalue:.2g}")
o, n = old[old[:, 2] >= 10], new[new[:, 2] >= 10]
print(f"tilt >= 10 deg only: 10/08 n={len(o)} {np.median(o[:, 0]):+.2f}, 10/14-17 n={len(n)} {np.median(n[:, 0]):+.2f}, "
      f"p={mannwhitneyu(o[:, 0], n[:, 0]).pvalue:.2g}; 10/14-17 tilt < 10: {np.median(new[new[:, 2] < 10, 0]):+.2f}")
W, RT = old[:, 3:5], old[:, 0:2]
print(f"10/08 pattern: constant world shift rms {np.sqrt(((W - W.mean(0)) ** 2).sum(1).mean()):.2f} cm vs constant "
      f"radial/tangential rms {np.sqrt(((RT - RT.mean(0)) ** 2).sum(1).mean()):.2f} cm")
print("ep0:", {k: round(v, 2) for k, v in ep[RUNS[0]]["0"].items() if k in ("rad_med", "tan_med", "tilt_med", "reach_med", "n")})
