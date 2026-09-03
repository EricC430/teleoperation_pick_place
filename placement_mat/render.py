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

from placement_mat.labels import short_label
from placement_mat.tiling import A4_H_CM, A4_W_CM, Tile, TilePlan, seam_points

_CM_PER_IN = 2.54

_MARKER = {
    "train": "o",
    "eval-open": "P",
    "eval-close": "^",
}

# Print-tuned greyscale + line weights. A laser printer drops a line that is
# either too light (grey below ~0.75) OR too thin (well under ~0.5 pt renders as
# an unreliable hairline). Eric's first print showed only the 5 cm lines, so
# both dials are turned up: darker, solid (no alpha), and every weight kept at
# or above 0.5 pt. --line-scale multiplies all of these for further tuning.
_INK_MM = "0.55"       # fine mm grid
_INK_CM = "0.32"       # 1 cm grid
_INK_5CM = "0.08"      # 5 cm grid (near black)
_INK_POLAR = "0.42"    # polar rings + rays
_INK_LABEL = "0.20"    # axis / ring / ray numbers
_LW_MM = 0.5           # pt; floor for a reliably-printing hairline
_LW_CM = 0.9
_LW_5CM = 1.4
_LW_POLAR = 0.9


@dataclass(frozen=True)
class TileOpts:
    r_max: float
    grid_mm: float = 5.0
    ring_step: float = 5.0
    ray_step: float = 15.0
    ruler_cm: float = 10.0
    azimuth_offset_deg: float = 0.0
    theta_min_deg: float = -90.0
    theta_max_deg: float = 90.0
    line_scale: float = 1.0  # multiplies every grid/overlay line width
    # --- single-page print-shop mat only ---------------------------------
    polar_origin_x: float = 0.0    # x of the pan axis in drawing coords (= -datum_offset)
    near_edge: bool = False        # draw the arm-tips registration line at x=0
    chassis_gap_cm: float = 5.9    # C-opening inner width -> tick spacing on that line
    azimuth_calibrated: bool = True  # False -> print "azimuth UNCALIBRATED - base frame"
    short_labels: bool = False     # annotate points t1/o1/c1 instead of train_001
    chassis_outline: bool = False  # draw the C-base footprint behind the reg line + a notch-cut guide
    chassis_arm_cm: float = 4.5    # inner-edge line -> front tips
    chassis_back_cm: float = 7.5   # back-segment depth, behind the inner-edge line
    chassis_outer_w_cm: float = 12.0  # full plate width


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
    single_page: bool = False,
) -> tuple[Figure, int]:
    w_cm, h_cm = tile.x1 - tile.x0, tile.y1 - tile.y0
    if single_page:
        # one large-format page: the figure IS the mat area, axes fill it,
        # so 1 data-cm == 1 paper-cm at 100% print with no A4 furniture.
        fig = plt.figure(figsize=(w_cm / _CM_PER_IN, h_cm / _CM_PER_IN))
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    else:
        fig = plt.figure(figsize=(A4_W_CM / _CM_PER_IN, A4_H_CM / _CM_PER_IN))
        ax_w_frac = w_cm / A4_W_CM
        ax_h_frac = h_cm / A4_H_CM
        left = (1.0 - ax_w_frac) / 2.0
        bottom = (1.0 - ax_h_frac) / 2.0
        ax = fig.add_axes((left, bottom, ax_w_frac, ax_h_frac))
    ax.set_xlim(tile.x0, tile.x1)
    ax.set_ylim(tile.y0, tile.y1)
    ax.set_xticks([])
    ax.set_yticks([])
    s = opts.line_scale
    for spine in ax.spines.values():
        spine.set_linewidth(0.6 * s)
        spine.set_color(_INK_5CM)

    # --- Cartesian grid -------------------------------------------------
    for gx in _frange(tile.x0, tile.x1, opts.grid_mm / 10.0):
        ax.axvline(gx, color=_INK_MM, lw=_LW_MM * s, zorder=0)
    for gy in _frange(tile.y0, tile.y1, opts.grid_mm / 10.0):
        ax.axhline(gy, color=_INK_MM, lw=_LW_MM * s, zorder=0)
    for gx in _frange(tile.x0, tile.x1, 1.0):
        ax.axvline(gx, color=_INK_CM, lw=_LW_CM * s, zorder=0)
    for gy in _frange(tile.y0, tile.y1, 1.0):
        ax.axhline(gy, color=_INK_CM, lw=_LW_CM * s, zorder=0)
    for gx in _frange(tile.x0, tile.x1, 5.0):
        ax.axvline(gx, color=_INK_5CM, lw=_LW_5CM * s, zorder=0)
        ax.text(gx, tile.y0 + 0.3, f"x={gx:g}", fontsize=5, color=_INK_LABEL, ha="left", va="bottom")
    for gy in _frange(tile.y0, tile.y1, 5.0):
        ax.axhline(gy, color=_INK_5CM, lw=_LW_5CM * s, zorder=0)
        ax.text(tile.x0 + 0.3, gy + 0.15, f"y={gy:g}", fontsize=5, color=_INK_LABEL, ha="left", va="bottom")

    # --- polar overlay: rings + rays, centred on the pan axis (ox, 0) ----
    # ox is 0 for the A4 mat (origin on the sheet); for the single-page mat the
    # pan axis sits off-sheet at x = -datum_offset, so every arc/ray shifts by ox.
    ox = opts.polar_origin_x
    a_lo, a_hi = opts.theta_min_deg, opts.theta_max_deg
    th = [math.radians(a) for a in _frange(a_lo, a_hi, 1.0)]
    for rr in _frange(max(opts.ring_step, 0.1), opts.r_max, opts.ring_step):
        ax.plot([ox + rr * math.cos(t) for t in th], [rr * math.sin(t) for t in th],
                color=_INK_POLAR, lw=_LW_POLAR * s, zorder=1)
        ax.text(ox + rr * math.cos(math.radians(a_hi)), rr * math.sin(math.radians(a_hi)),
                f"{rr:g}", fontsize=5, color=_INK_LABEL)
    ray0 = math.ceil(a_lo / opts.ray_step) * opts.ray_step
    for a in _frange(ray0, a_hi, opts.ray_step):
        e = math.radians(a + opts.azimuth_offset_deg)
        ax.plot([ox, ox + opts.r_max * math.cos(e)], [0, opts.r_max * math.sin(e)],
                color=_INK_POLAR, lw=_LW_POLAR * s, zorder=1)
        lx, ly = ox + opts.r_max * math.cos(e), opts.r_max * math.sin(e)
        if tile.x0 <= lx <= tile.x1 and tile.y0 <= ly <= tile.y1:
            ax.text(lx, ly, f"{a:g}°", fontsize=5, color=_INK_LABEL)

    if single_page:
        if opts.near_edge:
            half = opts.chassis_gap_cm / 2.0
            ax.axvline(0.0, color="k", lw=1.6, zorder=5)
            for ty in (-half, half):
                ax.plot([-0.4, 0.4], [ty, ty], color="k", lw=1.6, zorder=5)
            ax.text(0.3, tile.y1 - 0.6,
                    "align this line to BOTH C-arm front tips  (fixes forward + rotation)",
                    fontsize=6, rotation=90, va="top")
            ax.text(0.3, half + 0.2, "L C-arm tip", fontsize=5, color=_INK_LABEL)
            ax.text(0.3, -half - 0.5, "R C-arm tip", fontsize=5, color=_INK_LABEL)
        if opts.chassis_outline:
            # datum (x=0) is the front-tips line; the C-base sits behind it.
            ho = opts.chassis_outer_w_cm / 2.0
            inner_x = -opts.chassis_arm_cm                       # inner-edge line
            true_back_x = inner_x - opts.chassis_back_cm         # real rear face
            back_x = max(true_back_x, tile.x0)                   # clamp to the sheet edge
            rect = dict(color="0.35", lw=1.1, zorder=4)
            # two arms (the C legs): inner-edge line -> tips, gap..outer on each side
            for lo, hi in ((half, ho), (-ho, -half)):
                ax.plot([inner_x, 0, 0, inner_x, inner_x], [lo, lo, hi, hi, lo], **rect)
            # back segment (full width, behind the inner-edge line; may be clipped to the sheet)
            ax.plot([back_x, inner_x, inner_x, back_x, back_x], [-ho, -ho, ho, ho, -ho], **rect)
            ax.plot([inner_x, inner_x], [-half, half], **rect)  # inner-edge line across the opening
            if true_back_x < tile.x0:
                ax.text(tile.x0 + 0.2, ho - 1.0, "base continues <--", fontsize=5, color=_INK_LABEL)
            if tile.x0 <= ox:
                ax.plot(ox, 0.0, "+", color="k", ms=13, mew=1.8, zorder=6)
                ax.text(ox + 0.3, 0.4, "pan axis", fontsize=5, color=_INK_LABEL)
            # dashed notch-cut guide: trim this from the sheet so it clears the base
            ax.plot([0, tile.x0, tile.x0, 0], [-ho, -ho, ho, ho],
                    color="0.35", lw=1.0, ls="--", zorder=4)
            ax.text(tile.x0 + 0.3, ho - 0.3, "cut this notch (base sits here)", fontsize=5,
                    color=_INK_LABEL, va="top")
        if not opts.azimuth_calibrated:
            ax.text((tile.x0 + tile.x1) / 2, tile.y1 - 0.3,
                    "azimuth UNCALIBRATED - base frame (no S1 reference sample)",
                    fontsize=7, ha="center", va="top", color="0.25")
    # origin furniture, only on the tile that contains (0,0) (A4 mat)
    elif tile.x0 <= 0 <= tile.x1 and tile.y0 <= 0 <= tile.y1:
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

    if not single_page:
        # --- registration crosses on the shared seams -------------------
        for sx, sy in seam_points(plan):
            if tile.x0 <= sx <= tile.x1 and tile.y0 <= sy <= tile.y1:
                ax.plot(sx, sy, "k+", ms=10, mew=1.2, zorder=6)
                ax.text(sx + 0.2, sy + 0.2, f"({sx:g},{sy:g})", fontsize=4.5, color="0.3", zorder=6)

        # --- page label ------------------------------------------------------
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
        label = short_label(pid) if opts.short_labels else pid
        ax.plot(x, y, marker=_MARKER[_prefix(pid)], mfc="none", mec="k", mew=1.0, ms=7, zorder=8)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, 3), fontsize=5, zorder=8)
        drawn += 1

    return fig, drawn


def render_pdf(
    path: str | Path,
    plan: TilePlan,
    opts: TileOpts,
    *,
    points: Sequence[tuple[str, float, float]] | None = None,
    single_page: bool = False,
) -> RenderReport:
    path = Path(path)
    total = 0
    with PdfPages(path) as pdf:
        for tile in plan.tiles:
            fig, drawn = build_tile_figure(tile, plan, opts, points=points, single_page=single_page)
            pdf.savefig(fig)
            plt.close(fig)
            total += drawn
    return RenderReport(
        pages=len(plan.tiles),
        label_count=total,
        distinct_ids={pid for pid, _, _ in (points or [])},
    )
