"""Polar scatter of the reach samples plus a shaded usable wedge.

Spec: docs/specs/S1_reach_logger.md 9(2). Purely a look-before-you-pick-a-margin
aid; nothing downstream reads it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from reach_logger.samples import SampleRow
from reach_logger.summary import ReachSummary

_STYLE = {
    "outer_topdown": ("o", "tab:blue", "outer / top-down"),
    "outer_side": ("s", "tab:gray", "outer / side-only"),
    "inner": ("^", "tab:green", "inner"),
    "azimuth_limit": ("x", "tab:red", "azimuth limit"),
}


def _azimuth(row: SampleRow, offset: float) -> float:
    return row.azimuth_base_deg + offset


def build_figure(rows: Sequence[SampleRow], summary: ReachSummary) -> Figure:
    offset = summary.azimuth_offset_deg or 0.0
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(7, 7))
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)

    # shaded usable wedge, if we have the bounds for it
    if (
        summary.azimuth_min_deg is not None
        and summary.azimuth_max_deg is not None
        and summary.r_outer_topdown_cm is not None
    ):
        r_in = summary.r_inner_cm or 0.0
        thetas = [
            math.radians(a)
            for a in _linspace(summary.azimuth_min_deg, summary.azimuth_max_deg, 60)
        ]
        ax.fill_between(
            thetas,
            r_in,
            summary.r_outer_topdown_cm,
            alpha=0.12,
            color="tab:blue",
            label="usable wedge (pre-margin)",
        )

    for sample_type, (marker, color, label) in _STYLE.items():
        pts = [r for r in rows if r.sample_type == sample_type]
        if not pts:
            continue
        if sample_type == "azimuth_limit":
            th = [math.radians(_azimuth(r, offset)) for r in pts]
            rr = [summary.r_outer_topdown_cm or summary.r_outer_side_cm or 30.0] * len(pts)
        else:
            pts = [r for r in pts if r.radius_cm is not None]
            th = [math.radians(_azimuth(r, offset)) for r in pts]
            rr = [r.radius_cm for r in pts]
        if pts:
            ax.scatter(th, rr, marker=marker, c=color, s=60, label=label, zorder=3)

    ax.set_rlabel_position(90)
    ax.set_ylabel("radius (cm)")
    frame = "mat frame" if summary.azimuth_frame == "mat" else "base frame"
    title = f"reach samples — {frame}"
    if summary.azimuth_frame != "mat":
        title = "UNCALIBRATED — base frame (angles are relative only)"
    ax.set_title(title, pad=20)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    fig.tight_layout()
    return fig


def save_plot(
    rows: Sequence[SampleRow], summary: ReachSummary, out_path: str | Path
) -> Path:
    out_path = Path(out_path)
    fig = build_figure(rows, summary)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _linspace(a: float, b: float, n: int) -> list[float]:
    if n <= 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]
