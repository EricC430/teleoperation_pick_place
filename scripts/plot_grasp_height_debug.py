#!/usr/bin/env python
"""Side view of the arm at the grasp, drawn from REAL uvc_60 data.

Shows the same recorded frames converted with the old placeholder zeros and with the S6 measured
constants, against the cup that episode was actually reaching for. It also labels the three things
that are easy to confuse: the UPPER ARM (the joint2 -> joint3 segment), that segment's 20.14 deg
lean at URDF zero, and link5 (the wrist_roll body, parent of the gripper).

    uv run python scripts/plot_grasp_height_debug.py

The grasp frame is each episode's MOST CLOSED gripper frame. That detector was picked by scoring
candidates against the recorded cup placements: it is the only one that lands on a closed gripper
AND over the cup (reach error -0.2 cm). Two plausible-looking alternatives are both wrong and give
confident wrong answers -- the first frame with action-state < -1.0 lands on an OPEN gripper, and
the episode's lowest end-effector point lands 21.5 cm away, on a parked pose. See S5 section 2.

Horizontal axis is the PLANAR DISTANCE from the shoulder_pan axis, not raw x, so an episode placed
off to one side is not foreshortened. Writes outputs/grasp_height_debug.png.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow.parquet as pq

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "sim"))
import joint_mapping as JM          # noqa: E402
import scene_constants as SC        # noqa: E402
from reach_logger import fk         # noqa: E402

DATASET = _REPO / ".cache/lerobot/omx_pick_place_pilot_uvc_60/data/chunk-000/file-000.parquet"
META = _REPO / "episode_meta/omx_pick_place_pilot_paper_cup.csv"
PLACEMENTS = _REPO / "configs/placements/campA_136sym_20260908_20260908_train.csv"
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID, SURFACE = "#1a1a1a", "#6b6b6b", "#d8d8d4", "#fcfcfb"
DERIVED = (4095 / 200) * (360 / 4096) * math.pi / 180
PAN_X = fk._PAN_AXIS_XY[0]


def q_of(row, placeholder: bool) -> list[float]:
    return [DERIVED * row[i] for i in range(5)] if placeholder else list(JM.row_to_sim_rad(row)[:5])


def chain_rz(q) -> list[tuple[float, float]]:
    """(planar distance from the pan axis, z) for link0, joint1..5 and the end effector."""
    t = np.eye(4)
    out = [(0.0, 0.0)]
    for (origin, axis), a in zip(fk._CHAIN, q):
        t = t @ fk._translation(origin) @ fk._rotation(axis, a)
        out.append((math.hypot(t[0, 3] - PAN_X, t[1, 3]), t[2, 3]))
    ee = t @ fk._translation(fk._EE_ORIGIN)
    out.append((math.hypot(ee[0, 3] - PAN_X, ee[1, 3]), ee[2, 3]))
    return out


def load() -> tuple[dict[int, list[float]], dict[int, float]]:
    place = {r["placement_id"]: r for r in csv.DictReader(PLACEMENTS.open(encoding="utf-8"))}
    cup_r = {}
    for r in csv.DictReader(META.open(encoding="utf-8", errors="replace")):
        pid = r["placement_id"]
        if pid.startswith("t") and pid[1:].isdigit():          # 't7' -> 'train_007'
            cup_r[int(r["episode_index"])] = float(place["train_%03d" % int(pid[1:])]["r_cm"]) / 100
    t = pq.read_table(DATASET, columns=["episode_index", "observation.state"]).to_pydict()
    ep = np.array(t["episode_index"])
    st = np.array([r for r in t["observation.state"]], float)
    rows = {}
    for e in np.unique(ep):
        block = st[ep == e]
        rows[int(e)] = block[int(np.argmin(block[:, 5]))].tolist()   # most closed gripper
    return rows, cup_r


def main() -> int:
    if not DATASET.exists():
        print(f"dataset not found: {DATASET}")
        return 1
    rows, cup_r = load()
    riser, cup_h = SC.ARM_RISER_HEIGHT, SC.CUP_HEIGHT
    # the episode whose grasp height is the median -- a real frame, and we draw ITS cup
    rows = {e: r for e, r in rows.items() if e in cup_r}
    order = sorted(rows, key=lambda e: fk.ee_position_m(q_of(rows[e], False))[2])
    shown = order[len(order) // 2]
    r_cup = cup_r.get(shown, 0.30)
    tilt = math.degrees(math.atan2(fk._CHAIN[2][0][0], fk._CHAIN[2][0][2]))

    fig, ax = plt.subplots(figsize=(12, 7), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(0, color=INK, lw=2, zorder=2)
    ax.text(-0.115, -0.014, "table", color=MUTED, fontsize=9, va="top")
    ax.add_patch(plt.Rectangle((-0.085, 0), 0.146, riser, fc="#efefec", ec=GRID, zorder=1))
    ax.text(-0.080, 0.05, f"riser\n{riser * 100:.0f} cm", color=MUTED, fontsize=8)
    ax.add_patch(plt.Rectangle((-0.060, riser), 0.12, 0.0575, fc="#e2e2de", ec=GRID, zorder=1))

    ax.add_patch(plt.Rectangle((r_cup - SC.CUP_MEAN_DIA / 2, 0), SC.CUP_MEAN_DIA, cup_h,
                               fc="none", ec=INK, lw=1.6, ls="--", zorder=2))
    ax.plot([r_cup], [cup_h], marker="_", ms=22, color=INK, lw=3, zorder=4)
    ax.annotate(f"cup rim {cup_h * 100:.1f} cm\nat this episode's recorded\nplacement r = "
                f"{r_cup * 100:.1f} cm", xy=(r_cup, cup_h),
                xytext=(r_cup + 0.035, cup_h + 0.085), color=INK, fontsize=9,
                arrowprops=dict(arrowstyle="-", color=INK, lw=1))

    tags = ("J1 pan", "J2 lift", "J3 elbow", "J4 wrist_flex", "J5 = link5", "EE")
    for placeholder, color, label in ((True, ORANGE, "placeholder zeros (before S6)"),
                                      (False, BLUE, "S6 measured constants")):
        pts = [(r, z + riser) for r, z in chain_rz(q_of(rows[shown], placeholder))]
        rs, zs = zip(*pts)
        ax.plot(rs, zs, "-o", color=color, lw=2.2, ms=8, mfc=SURFACE, mew=2, zorder=5,
                label=f"{label}   grasp at {pts[-1][1] * 100:.1f} cm")
        ax.plot([pts[-1][0]], [pts[-1][1]], marker="*", ms=18, color=color, zorder=6)
        ax.annotate(f"{pts[-1][1] * 100:.1f} cm", xy=pts[-1],
                    xytext=(pts[-1][0] + 0.018, pts[-1][1] - 0.014),
                    color=color, fontsize=11, fontweight="bold", va="center")
        if not placeholder:
            for (r, z), tag in zip(pts[1:], tags):
                ax.annotate(tag, xy=(r, z), xytext=(r - 0.009, z + 0.017),
                            color=MUTED, fontsize=8, ha="right")
            ax.annotate("", xy=pts[3], xytext=pts[2],
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=3.2, alpha=0.85))

    ax.set_xlabel("planar distance from the shoulder_pan axis (m)", color=INK, fontsize=10)
    ax.set_ylabel("height above the table (m)", color=INK, fontsize=10)
    ax.set_title("What the S6 calibration fixed, and the 7 cm still open\n"
                 f"episode {shown} at its grasp -- the median of {len(rows)} uvc_60 episodes",
                 color=INK, fontsize=13, loc="left", pad=12)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-0.12, 0.50)
    ax.set_ylim(-0.03, 0.45)
    ax.legend(loc="upper right", frameon=False, fontsize=10, labelcolor=INK)

    fig.subplots_adjust(bottom=0.30)
    fig.text(0.055, 0.155,
             "The three words, on this drawing:\n"
             "   UPPER ARM  the thick dark segment, joint2 to joint3\n"
             f"   its {tilt:.2f} deg lean  at URDF zero that segment is NOT vertical, because "
             f"joint3 sits 4.15 cm forward of joint2 (S6 section 4-a)\n"
             "   link5  the wrist_roll body, the gripper's parent (J5)\n"
             "   EE  end_effector_link, a URDF marker 9.19 cm past link5 -- beyond the fingertips, "
             "so it is the lowest point on the chain, not the grip point",
             color=INK, fontsize=9.5, va="top", linespacing=1.6)

    out = _REPO / "outputs" / "grasp_height_debug.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}  (episode {shown}, cup r = {r_cup * 100:.1f} cm)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
