"""Diagnostic scatter of the three sampled lists.

Spec: docs/specs/S2_placement_sampler.md 5. This is a look-at-it check for
coverage and clustering -- nothing downstream reads it, and it is NOT the
printable polar mat (that is S3, scripts/make_placement_mat.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from placement_sampler.geometry import Sector
from placement_sampler.sampler import Placement

_STYLE = {
    "train": dict(marker="o", edgecolors="tab:blue", facecolors="none", s=42),
    "eval-open": dict(marker="+", c="tab:orange", s=70, linewidths=1.6),
    "eval-close": dict(marker="^", edgecolors="tab:red", facecolors="none", s=48),
}


def _arc(radius: float, t0: float, t1: float, n: int = 120) -> tuple[list[float], list[float]]:
    ts = [math.radians(t0 + (t1 - t0) * i / n) for i in range(n + 1)]
    return [radius * math.cos(t) for t in ts], [radius * math.sin(t) for t in ts]


def build_figure(
    lists: dict[str, Sequence[Placement]],
    sector: Sector,
    *,
    seed: int,
    d_min: float,
) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 6.5))

    xo, yo = _arc(sector.r_outer, sector.theta_min, sector.theta_max)
    xi, yi = _arc(sector.r_inner, sector.theta_min, sector.theta_max)
    ax.plot(xo, yo, "k-", lw=1)
    ax.plot(xi, yi, "k-", lw=1)
    for edge in (sector.theta_min, sector.theta_max):
        e = math.radians(edge)
        ax.plot(
            [sector.r_inner * math.cos(e), sector.r_outer * math.cos(e)],
            [sector.r_inner * math.sin(e), sector.r_outer * math.sin(e)],
            "k-",
            lw=1,
        )

    ring = 5.0
    r = math.ceil(sector.r_inner / ring) * ring
    while r < sector.r_outer:
        gx, gy = _arc(r, sector.theta_min, sector.theta_max)
        ax.plot(gx, gy, color="0.85", lw=0.6, zorder=0)
        r += ring
    for a in range(int(math.ceil(sector.theta_min / 15) * 15), int(sector.theta_max) + 1, 15):
        e = math.radians(a)
        ax.plot(
            [sector.r_inner * math.cos(e), sector.r_outer * math.cos(e)],
            [sector.r_inner * math.sin(e), sector.r_outer * math.sin(e)],
            color="0.9",
            lw=0.6,
            zorder=0,
        )

    for name, style in _STYLE.items():
        pts = list(lists.get(name, []))
        n = len(pts)
        ax.scatter(
            [p.x_cm for p in pts],
            [p.y_cm for p in pts],
            label=f"{name} ({n})",
            zorder=3,
            **style,
        )

    ax.plot(0, 0, "k+", ms=12, mew=2)
    ax.set_aspect("equal")
    ax.set_xlabel("x  (cm)")
    ax.set_ylabel("y  (cm)")
    ax.set_title(
        f"S2 placements — seed {seed}, d_min {d_min:g} cm\n"
        "diagnostic scatter (not the S3 printable mat)"
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, color="0.94", lw=0.5)
    fig.tight_layout()
    return fig


def save_scatter(
    lists: dict[str, Sequence[Placement]],
    sector: Sector,
    out_path: str | Path,
    *,
    seed: int,
    d_min: float,
) -> Path:
    out_path = Path(out_path)
    fig = build_figure(lists, sector, seed=seed, d_min=d_min)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return out_path
