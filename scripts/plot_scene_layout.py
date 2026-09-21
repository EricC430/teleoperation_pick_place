#!/usr/bin/env python
"""Top-down plan of the simulated cell, straight from `sim/scene_constants.py`.

Every number in the sim scene is either measured or derived from a measurement, and a wrong one is
far easier to spot on a plan than in a render -- a render confounds geometry with camera pose and
framing, which is exactly what went wrong on 2026-09-21 when the third-person camera's bearing was
misread and the render was hard to diagnose.

    uv run python scripts/plot_scene_layout.py --out outputs/renders/scene_plan.png

Frame: the placement-mat frame -- origin at the pan axis, +X ahead (away from the operator),
+Y to the operator's left. Same frame `docs/assets/placement_label_map_*.csv` uses.
"""
from __future__ import annotations
import argparse, csv, math, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sim"))
import scene_constants as S  # noqa: E402

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--placements", default="docs/assets/placement_label_map_campA_136sym_20260908.csv")
ap.add_argument("--out", default="outputs/renders/scene_plan.png")
a = ap.parse_args()

fig, ax = plt.subplots(figsize=(10, 9))

ax.add_patch(Rectangle((S.TABLE_CENTER_XY[0]-S.TABLE_SIZE[0]/2, S.TABLE_CENTER_XY[1]-S.TABLE_SIZE[1]/2),
                       S.TABLE_SIZE[0], S.TABLE_SIZE[1], fc="#f0ead8", ec="#b9ad84", label="table (derived)"))
ax.add_patch(Rectangle((S.ARM_RISER_CENTER[0]-S.ARM_RISER_SIZE[0]/2, S.ARM_RISER_CENTER[1]-S.ARM_RISER_SIZE[1]/2),
                       S.ARM_RISER_SIZE[0], S.ARM_RISER_SIZE[1], fc="#8fd3e8", ec="#2b7f99", alpha=.85,
                       label=f"riser {S.ARM_RISER_HEIGHT*100:.0f}cm (derived footprint)"))
ax.add_patch(Rectangle((S.ARM_BASE_BACK_X, -S.ARM_BASE_HALF_Y),
                       S.ARM_BASE_FRONT_X-S.ARM_BASE_BACK_X, 2*S.ARM_BASE_HALF_Y,
                       fc="none", ec="#333", lw=2, label="arm base plate (from STL)"))
ax.add_patch(Circle((S.BIN_CENTER_X, S.BIN_CENTER_Y), S.BIN_OPENING_DIA/2, fc="#3f7fc4", ec="#1b4a80",
                    alpha=.9, label=f"bin ⌀{S.BIN_OPENING_DIA*100:.0f}cm h{S.BIN_HEIGHT*100:.0f}cm"))
ax.plot(0, 0, "k+", ms=16, mew=3, label="pan axis (origin)")

xs, ys, tx, ty = [], [], [], []
for r in csv.DictReader(open(a.placements)):
    x, y = float(r["x_pan_cm"])/100, float(r["y_pan_cm"])/100
    (tx if r["short_id"].startswith("t") else xs).append(x)
    (ty if r["short_id"].startswith("t") else ys).append(y)
ax.scatter(xs, ys, s=8, c="#bbb", label="other placements")
ax.scatter(tx, ty, s=14, c="#d1495b", label="t1..t60 (uvc_60)")

cx, cy, cz = S.CAM_FRONT_LEFT_POS
lx, ly, _ = S.CAM_FRONT_LEFT_LOOKAT
ax.plot(cx, cy, "*", ms=20, c="#f0a202", mec="k", label=f"front-left cam (z={cz*100:.0f}cm)")
ax.annotate("", xy=(lx, ly), xytext=(cx, cy), arrowprops=dict(arrowstyle="->", lw=2.5, color="#f0a202"))
bearing = math.degrees(math.atan2(ly-cy, lx-cx))
ax.text(cx, cy+0.03, f"bearing {bearing:.0f}°", color="#a06a00", fontsize=9, ha="center")

# where the cup for episode 0 actually is, and where the camera WOULD have to point to see it
t1 = next((float(r["x_pan_cm"])/100, float(r["y_pan_cm"])/100)
          for r in csv.DictReader(open(a.placements)) if r["short_id"] == "t1")
to_t1 = math.degrees(math.atan2(t1[1]-cy, t1[0]-cx))
ax.annotate("", xy=t1, xytext=(cx, cy), arrowprops=dict(arrowstyle="->", lw=1.6, color="#888", ls="--"))
ax.text(*(t1[0], t1[1]-0.045), f"t1 — needs {to_t1:.0f}°", color="#555", fontsize=9)

ax.set_xlabel("+X  ahead / away from operator  (m)"); ax.set_ylabel("+Y  operator's LEFT  (m)")
ax.set_title("Simulated cell, top-down (placement-mat frame)\nsim/scene_constants.py")
ax.set_aspect("equal"); ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
fig.savefig(a.out, dpi=130, bbox_inches="tight")
print(f"wrote {a.out}")
print(f"camera at ({cx:.3f}, {cy:.3f}, {cz:.3f}), bearing {bearing:.1f}°; t1 would need {to_t1:.1f}°")
