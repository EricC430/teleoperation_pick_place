"""Render the A4-tiled polar mat to PDF.

Spec: docs/specs/S3_placement_mat.md 4.

Physical scale: each page is an A4 figure (sized in inches); the drawing axes
are placed so one data unit (cm) is exactly one cm on paper when printed at
100%. Every page carries a 10 cm calibration ruler so the human can verify
the printer did not rescale (spec 2-2).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from placement_mat.tiling import A4_H_CM, A4_W_CM, Tile, TilePlan, seam_points

_CM_PER_IN = 2.54

_MARKER = {
    "train": "o",
    "eval-open": "P",
    "eval-close": "^",
}


@dataclass(frozen=True)
class TileOpts:
    r_max: float
    grid_mm: float = 5.0
    ring_step: float = 5.0
    ray_step: float = 15.0
    ruler_cm: float = 10.0
    azimuth_offset_deg: float = 0.0


@dataclass(frozen=True)
class RenderReport:
    pages: int
    label_count: int
    distinct_ids: set[str]


def paper_cm_per_data_cm(tile: Tile, *, paper_w: float = A4_W_CM) -> float:
    """cm on paper per cm of data, tracing the same chain build_tile_figure uses.

    figure width = paper_w/2.54 inches; the axes takes a fraction (x-span)/paper_w
    of that; convert back to cm and divide by the x-span it represents. Comes out
    to 1.0 by construction -- this function exists so the invariant is testable if
    the axes-placement rule is ever changed. The real-world guarantee is the 10 cm
    ruler printed on every page (spec 2-2).
    """
    fig_w_in = paper_w / _CM_PER_IN
    ax_w_in = ((tile.x1 - tile.x0) / paper_w) * fig_w_in
    return (ax_w_in * _CM_PER_IN) / (tile.x1 - tile.x0)


def points_on_tile(
    points: Sequence[tuple[str, float, float]], tile: Tile
) -> list[tuple[str, float, float]]:
    return [
        (pid, x, y)
        for pid, x, y in points
        if tile.x0 <= x <= tile.x1 and tile.y0 <= y <= tile.y1
    ]


def _prefix(pid: str) -> str:
    for p in _MARKER:
        if pid.startswith(p):
            return p
    return "train"


def _frange(a: float, b: float, step: float) -> list[float]:
    out, v = [], math.ceil(a / step) * step
    while v <= b + 1e-9:
        out.append(round(v, 6))
        v += step
    return out


def build_tile_figure(
    tile: Tile,
    plan: TilePlan,
    opts: TileOpts,
    *,
    points: Sequence[tuple[str, float, float]] | None = None,
) -> tuple[Figure, int]:
    fig = plt.figure(figsize=(A4_W_CM / _CM_PER_IN, A4_H_CM / _CM_PER_IN))
    ax_w_frac = (tile.x1 - tile.x0) / A4_W_CM
    ax_h_frac = (tile.y1 - tile.y0) / A4_H_CM
    left = (1.0 - ax_w_frac) / 2.0
    bottom = (1.0 - ax_h_frac) / 2.0
    ax = fig.add_axes((left, bottom, ax_w_frac, ax_h_frac))
    ax.set_xlim(tile.x0, tile.x1)
    ax.set_ylim(tile.y0, tile.y1)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)
        spine.set_color("0.6")

    # --- Cartesian grid ---------------------------------------------------
    for gx in _frange(tile.x0, tile.x1, opts.grid_mm / 10.0):
        ax.axvline(gx, color="0.90", lw=0.3, zorder=0)
    for gy in _frange(tile.y0, tile.y1, opts.grid_mm / 10.0):
        ax.axhline(gy, color="0.90", lw=0.3, zorder=0)
    for gx in _frange(tile.x0, tile.x1, 1.0):
        ax.axvline(gx, color="0.72", lw=0.5, zorder=0)
    for gy in _frange(tile.y0, tile.y1, 1.0):
        ax.axhline(gy, color="0.72", lw=0.5, zorder=0)
    for gx in _frange(tile.x0, tile.x1, 5.0):
        ax.axvline(gx, color="0.45", lw=0.8, zorder=0)
        ax.text(gx, tile.y0 + 0.3, f"x={gx:g}", fontsize=5, color="0.4", ha="left", va="bottom")
    for gy in _frange(tile.y0, tile.y1, 5.0):
        ax.axhline(gy, color="0.45", lw=0.8, zorder=0)
        ax.text(tile.x0 + 0.3, gy + 0.15, f"y={gy:g}", fontsize=5, color="0.4", ha="left", va="bottom")

    # --- polar overlay (clipped to r <= r_max) --------------------------
    th = [math.radians(a) for a in range(-180, 181, 2)]
    for rr in _frange(max(opts.ring_step, 0.1), opts.r_max, opts.ring_step):
        xs = [rr * math.cos(t) for t in th]
        ys = [rr * math.sin(t) for t in th]
        ax.plot(xs, ys, color="tab:blue", lw=0.5, alpha=0.5, zorder=1)
        ax.text(rr, 0.2, f"{rr:g}", fontsize=5, color="tab:blue", alpha=0.8)
    for a in range(-180, 181, int(opts.ray_step)):
        e = math.radians(a + opts.azimuth_offset_deg)
        ax.plot([0, opts.r_max * math.cos(e)], [0, opts.r_max * math.sin(e)],
                color="tab:blue", lw=0.4, alpha=0.4, zorder=1)

    # origin furniture, only on the tile that contains (0,0)
    if tile.x0 <= 0 <= tile.x1 and tile.y0 <= 0 <= tile.y1:
        ax.plot(0, 0, "k+", ms=16, mew=2, zorder=5)
        ax.plot([0, opts.r_max], [0, 0], color="k", lw=1.0, zorder=4)
        ax.text(opts.r_max * 0.4, 0.4, "0deg ray -- align to S1 reference", fontsize=6)
        ax.text(0.4, -0.8, "align (0,0) to base rotation axis", fontsize=6)

    # --- 10 cm calibration ruler (every page) --------------------------
    rx = tile.x0 + 1.5
    ry = tile.y0 + 1.0
    ax.plot([rx, rx + opts.ruler_cm], [ry, ry], color="k", lw=1.5, zorder=6)
    for k in range(int(opts.ruler_cm) + 1):
        ax.plot([rx + k, rx + k], [ry, ry + 0.25], color="k", lw=1.0, zorder=6)
    ax.text(rx, ry + 0.4, f"{opts.ruler_cm:.1f} cm  -- measure me; reprint at 100% if wrong",
            fontsize=6, zorder=6)

    # --- registration crosses on the shared seams ---------------------
    for sx, sy in seam_points(plan):
        if tile.x0 <= sx <= tile.x1 and tile.y0 <= sy <= tile.y1:
            ax.plot(sx, sy, "k+", ms=10, mew=1.2, zorder=6)
            ax.text(sx + 0.2, sy + 0.2, f"({sx:g},{sy:g})", fontsize=4.5, color="0.3", zorder=6)

    # --- page label -------------------------------------------------------
    ax.text(
        tile.x1 - 0.3,
        tile.y1 - 0.3,
        f"row {tile.row + 1}/{plan.n_rows}  col {tile.col + 1}/{plan.n_cols}",
        fontsize=7,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", fc="white", ec="0.6", lw=0.5),
        zorder=7,
    )

    # --- placement points ----------------------------------------------
    drawn = 0
    for pid, x, y in points_on_tile(points or [], tile):
        ax.plot(x, y, marker=_MARKER[_prefix(pid)], mfc="none", mec="k", mew=1.0, ms=7, zorder=8)
        ax.annotate(pid, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5, zorder=8)
        drawn += 1

    return fig, drawn


def render_pdf(
    path: str | Path,
    plan: TilePlan,
    opts: TileOpts,
    *,
    points: Sequence[tuple[str, float, float]] | None = None,
) -> RenderReport:
    path = Path(path)
    total = 0
    with PdfPages(path) as pdf:
        for tile in plan.tiles:
            fig, drawn = build_tile_figure(tile, plan, opts, points=points)
            pdf.savefig(fig)
            plt.close(fig)
            total += drawn
    return RenderReport(
        pages=len(plan.tiles),
        label_count=total,
        distinct_ids={pid for pid, _, _ in (points or [])},
    )
