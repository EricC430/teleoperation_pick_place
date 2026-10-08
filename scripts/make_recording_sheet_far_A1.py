#!/usr/bin/env python
"""Field sheet for the far-zone wrist-rotation recording (configs/record_omx_cup_far_wrist_A1.yaml).

Same view as docs/assets/recording_sheet_cup_normal_recovery_A1.png: top-down, arm base at the origin,
up = forward (x_pan), right = the arm's right (-y_pan). Every t point with r >= 35.5 cm gets its first-pass
episode number (second pass = +23) and its (x_pan, y_pan) in cm - the coordinate paper's origin is the pan axis
(its bottom 7.13 cm folded under, butted against the chassis front edge).

    uv run python scripts/make_recording_sheet_far_A1.py   # -> docs/assets/recording_sheet_cup_far_wrist_A1.png
"""
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LABEL_MAP = REPO / "docs/assets/placement_label_map_campA_136sym_20260908.csv"
OUT = REPO / "docs/assets/recording_sheet_cup_far_wrist_A1.png"
BOUNDARY, RIM_LIMIT, WRIST_LIMIT = 35.5, 36.1, 42.1  # cm from the pan axis (docs/meeting/2026-10-08.md §8)
OLD_C, NEW_C, INK, MUTED, TEST = "#2a78d6", "#eb6834", "#21211f", "#a3a29a", "#e8b4b0"  # dataviz palette slots 1/2

pts = {r["short_id"]: r for r in csv.DictReader(LABEL_MAP.open(encoding="utf-8"))}
xy = {k: (-float(r["y_pan_cm"]), float(r["x_pan_cm"])) for k, r in pts.items()}  # (right, forward)
radius = {k: math.hypot(*v) for k, v in xy.items()}
far = sorted((k for k in pts if k[0] == "t" and radius[k] >= BOUNDARY),
             key=lambda k: math.atan2(xy[k][0], xy[k][1]))  # arm's left -> right, the recording order
n = len(far)
# 2026-10-09: ep 0-8 recorded on 10-08; ep 9 = failed attempt at t65's old spot (too far, gripper never closed,
# saved by the stop) -> valid=0. Everything from t65 on shifts by one; pass 2 starts at ep n + 1.
DONE, VOID = 9, 9
pass1 = {k: (i if i < DONE else i + 1) for i, k in enumerate(far)}
pass2 = {k: n + 1 + i for i, k in enumerate(far)}

fig, (ax, tab) = plt.subplots(1, 2, figsize=(17, 9.5), gridspec_kw={"width_ratios": [3.1, 1]})
for k, (x, y) in xy.items():
    if k in far:
        continue
    ax.plot(x, y, "o", ms=4, color=MUTED if k[0] == "t" else TEST, zorder=1)
for r, style, label in ((BOUNDARY, "-", f"r {BOUNDARY}: rotate the wrist RIGHT from here out"),
                        (RIM_LIMIT, ":", f"r {RIM_LIMIT}: rim-grasp limit (29 cm from the front edge)"),
                        (WRIST_LIMIT, "--", f"r {WRIST_LIMIT}: wrist-rotation limit (35 cm)")):
    t = [math.radians(a) for a in range(-75, 76)]
    ax.plot([r * math.sin(a) for a in t], [r * math.cos(a) for a in t], style, color=INK, lw=1, alpha=0.6, label=label)
for i, k in enumerate(far):
    x, y = xy[k]
    new = int(k[1:]) >= 61
    ax.plot(x, y, "D" if new else "o", ms=19, color=NEW_C if new else OLD_C,
            markeredgecolor="white", markeredgewidth=2, zorder=3)
    ax.text(x, y, str(pass1[k]), ha="center", va="center", color="white", fontsize=9, fontweight="bold", zorder=4)
    # name only on the plot (coordinates are in the table); outer ring (r >= 39) outward, the rest inward
    ux, uy = x / radius[k], y / radius[k]
    d = 24 if radius[k] >= 39 else -22
    ax.annotate(k, (x, y), xytext=(d * ux, d * uy), textcoords="offset points", ha="center", va="center",
                fontsize=9, color=INK, zorder=4)
ax.plot([], [], "o", ms=10, color=OLD_C, label="existing t point, re-record with the wrist grasp")
ax.plot([], [], "D", ms=10, color=NEW_C, label="new point t61-t70 (not on the printed mat)")
ax.plot([], [], "o", ms=5, color=MUTED, label="other t points (not today)")
ax.plot([], [], "o", ms=5, color=TEST, label="c / o TEST points: never use")
ax.plot(0, 0, "s", ms=14, color=INK)
ax.text(0, -1.5, "arm base (pan axis)", ha="center", va="top", fontsize=9, color=INK)
ax.set_title(f"Far zone, wrist-rotation grasp: {n} points x 2 = {2 * n} eps. Number = pass-1 episode (pass 2 in the table).\n"
             f"ep 0-{DONE - 1} done 2026-10-08; ep {VOID} = failed t65 attempt (void); resume at ep {DONE + 1} with --dataset.num_episodes={2 * n - DONE}.\n"
             "Teleop-test the edges first: t61/t62 (near the camera), t70 (near the bin).",
             fontsize=11, color=INK)
ax.set_xlabel("<- arm's LEFT        cm        arm's RIGHT ->", color=INK)
ax.set_aspect("equal")
ax.set_xlim(-50, 50)
ax.set_ylim(-3, 49)
ax.grid(alpha=0.25)
ax.legend(loc="lower left", fontsize=8, framealpha=0.95)

# coordinate table: what to measure on the coordinate paper (origin = pan axis)
tab.axis("off")
lines = [f"{'ep':>5} {'point':>6} {'x_pan':>7} {'y_pan':>7}", "-" * 34]
for k in far:
    mark = "*" if int(k[1:]) >= 61 else " "
    done = "done" if pass1[k] < DONE else ""
    lines.append(f"{pass1[k]:>2}/{pass2[k]:<2} {k:>5}{mark} {float(pts[k]['x_pan_cm']):7.1f} {float(pts[k]['y_pan_cm']):7.1f}  {done}")
lines += ["-" * 34, f"ep {VOID}: void (failed t65 attempt)", "ep = pass 1 / pass 2; done = pass 1 recorded", "* = new point (not on the mat)", "cm, origin = pan axis",
          "x forward, y = arm's LEFT (+)"]
tab.text(0, 1, "\n".join(lines), family="monospace", fontsize=10.5, va="top", color=INK, transform=tab.transAxes)
fig.tight_layout()
fig.savefig(OUT, dpi=110)
print(f"{OUT}  ({n} points: {', '.join(far)})")
