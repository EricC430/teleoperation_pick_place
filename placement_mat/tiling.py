"""Cover a rectangular mat area with overlapping A4 pages.

Spec: docs/specs/S3_placement_mat.md 4 (multi-page A4 tiling), 5 (acceptance).

Each tile is exactly the printable area of one A4 sheet. Adjacent tiles share
an `overlap_cm` band so the printed pages can be aligned cross-on-cross and
taped. Tiles are NOT clamped to the requested area -- the last row/column runs
a little past it into white space, which is fine for printing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

A4_W_CM = 21.0
A4_H_CM = 29.7


@dataclass(frozen=True)
class Tile:
    row: int
    col: int
    x0: float
    x1: float
    y0: float
    y1: float


@dataclass(frozen=True)
class TilePlan:
    tiles: list[Tile]
    n_rows: int
    n_cols: int
    content_w: float
    content_h: float
    overlap_cm: float
    x_min: float
    y_min: float
    step_x: float
    step_y: float


def plan_tiles(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    paper_w: float = A4_W_CM,
    paper_h: float = A4_H_CM,
    margin_cm: float = 1.0,
    overlap_cm: float = 1.5,
) -> TilePlan:
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("empty mat area")
    content_w = paper_w - 2 * margin_cm
    content_h = paper_h - 2 * margin_cm
    step_x = content_w - overlap_cm
    step_y = content_h - overlap_cm
    if step_x <= 0 or step_y <= 0:
        raise ValueError("overlap_cm must be smaller than the printable area")

    n_cols = max(1, math.ceil((x_max - x_min - overlap_cm) / step_x))
    n_rows = max(1, math.ceil((y_max - y_min - overlap_cm) / step_y))

    tiles = [
        Tile(
            row=r,
            col=c,
            x0=x_min + c * step_x,
            x1=x_min + c * step_x + content_w,
            y0=y_min + r * step_y,
            y1=y_min + r * step_y + content_h,
        )
        for r in range(n_rows)
        for c in range(n_cols)
    ]
    return TilePlan(
        tiles=tiles,
        n_rows=n_rows,
        n_cols=n_cols,
        content_w=content_w,
        content_h=content_h,
        overlap_cm=overlap_cm,
        x_min=x_min,
        y_min=y_min,
        step_x=step_x,
        step_y=step_y,
    )


def seam_points(plan: TilePlan) -> list[tuple[float, float]]:
    """World coords of the interior page seams -- each lands in >=2 tiles.

    Used for the cross-on-cross registration marks: draw one at every seam
    intersection and it appears, at the same coordinate, on both neighbours.
    """
    xs = [plan.x_min + k * plan.step_x + plan.content_w - plan.overlap_cm / 2 for k in range(plan.n_cols - 1)]
    ys = [plan.y_min + m * plan.step_y + plan.content_h - plan.overlap_cm / 2 for m in range(plan.n_rows - 1)]
    # include the outer edges too, so single-row/col plans still get marks
    xs = xs or [(plan.tiles[0].x0 + plan.tiles[0].x1) / 2]
    ys = ys or [(plan.tiles[0].y0 + plan.tiles[0].y1) / 2]
    return [(x, y) for y in ys for x in xs]
