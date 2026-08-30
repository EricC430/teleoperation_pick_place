"""Stratified equal-area sampling of the three frozen placement lists.

Spec: docs/specs/S2_placement_sampler.md 4 (`[Eric決定]` 2026-08-31).

Not pure dart-throwing: at n~30 that leaves luck-of-the-seed empty patches,
and these lists exist to make failure *clustering* visible. Instead each
list's sector is cut into `n` equal-area cells and one uniform-random point
is drawn per cell.

  - Lists are filled largest-first (train 50, eval-close 30, eval-open 10).
  - `d_min` is global: a point must be >= d_min from EVERY accepted point
    across all lists; on a clash it is redrawn inside the SAME cell.
  - No auto-relaxation of d_min. Exhausting the per-cell redraw budget is a
    hard failure.
  - One random.Random(seed), consumed in a fixed order -> byte-identical per
    seed; train is finished before eval-* so eval counts never move a train
    point.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from placement_sampler.geometry import Sector, area_uniform_radius, polar_to_xy

_REDRAW_BUDGET = 200

# (r_inner_sq, r_outer_sq, theta_min_deg, theta_max_deg)
Cell = tuple[float, float, float, float]


@dataclass(frozen=True)
class Placement:
    placement_id: str
    r_cm: float
    theta_deg: float
    x_cm: float
    y_cm: float


class SamplingStuck(RuntimeError):
    """Raised when a cell cannot be filled within the redraw budget."""

    def __init__(self, list_name: str, accepted: int, target: int, d_min: float) -> None:
        self.list_name = list_name
        self.accepted = accepted
        self.target = target
        self.d_min = d_min
        super().__init__(
            f"stuck filling {list_name!r}: accepted {accepted}/{target} points at "
            f"d_min={d_min} cm before a cell ran out of redraws. "
            f"Do NOT relax d_min -- reduce N, shrink d_min deliberately, or enlarge the workspace."
        )


def _ordered(counts: dict[str, int]) -> list[tuple[str, int]]:
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def equal_area_cells(n: int, sector: Sector) -> list[Cell]:
    """`n_rings x n_wedges` equal-area cells covering `sector`, with count >= n.

    Radius is split so each ring has equal area (uniform in r^2); azimuth is
    split into equal angles. Cell count can exceed n; the caller picks n of them.
    """
    if n <= 0:
        return []
    radial = sector.r_outer - sector.r_inner
    arc = math.radians(sector.theta_max - sector.theta_min) * (sector.r_inner + sector.r_outer) / 2.0
    n_rings = max(1, round(math.sqrt(n * radial / arc))) if arc > 0 else 1
    n_wedges = math.ceil(n / n_rings)

    r_in_sq, r_out_sq = sector.r_inner**2, sector.r_outer**2
    cells: list[Cell] = []
    for i in range(n_rings):
        a0 = r_in_sq + (r_out_sq - r_in_sq) * i / n_rings
        a1 = r_in_sq + (r_out_sq - r_in_sq) * (i + 1) / n_rings
        for j in range(n_wedges):
            t0 = sector.theta_min + (sector.theta_max - sector.theta_min) * j / n_wedges
            t1 = sector.theta_min + (sector.theta_max - sector.theta_min) * (j + 1) / n_wedges
            cells.append((a0, a1, t0, t1))
    return cells


def sample_lists(
    sector: Sector,
    *,
    d_min: float,
    counts: dict[str, int],
    seed: int,
) -> dict[str, list[Placement]]:
    if d_min <= 0:
        raise ValueError(f"d_min must be > 0, got {d_min}")
    rng = random.Random(seed)
    d_min_sq = d_min**2
    accepted_xy: list[tuple[float, float]] = []
    out: dict[str, list[Placement]] = {name: [] for name in counts}

    for name, target in _ordered(counts):
        cells = equal_area_cells(target, sector)
        rng.shuffle(cells)
        for a0, a1, t0, t1 in cells[:target]:
            r_lo, r_hi = math.sqrt(a0), math.sqrt(a1)
            for _ in range(_REDRAW_BUDGET):
                r = area_uniform_radius(rng, r_lo, r_hi)
                theta = rng.uniform(t0, t1)
                x, y = polar_to_xy(r, theta)
                if any((x - ax) ** 2 + (y - ay) ** 2 < d_min_sq for ax, ay in accepted_xy):
                    continue
                accepted_xy.append((x, y))
                idx = len(out[name]) + 1
                out[name].append(
                    Placement(
                        placement_id=f"{name}_{idx:03d}",
                        r_cm=r,
                        theta_deg=theta,
                        x_cm=x,
                        y_cm=y,
                    )
                )
                break
            else:
                raise SamplingStuck(name, len(out[name]), target, d_min)
    return out


def global_min_separation_cm(lists: dict[str, list[Placement]]) -> float:
    pts = [(p.x_cm, p.y_cm) for pts in lists.values() for p in pts]
    return min(
        (math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :]),
        default=math.inf,
    )
