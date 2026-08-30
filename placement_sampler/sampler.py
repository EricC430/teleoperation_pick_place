"""Dart-throwing the three frozen placement lists with one global minimum spacing.

Spec: docs/specs/S2_placement_sampler.md 4.

  - Lists are filled largest-first (train 50, eval-close 30, eval-open 10) so the
    big list is not boxed into a corner by the small ones.
  - A candidate is accepted only if it is >= d_min from EVERY already-accepted
    point, across all lists (D024 2026-08-31: the minimum is global).
  - No auto-relaxation of d_min. Exhausting the attempt budget is a hard failure.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from placement_sampler.geometry import Sector, polar_to_xy, sample_sector_point

_ATTEMPTS_PER_POINT = 200


@dataclass(frozen=True)
class Placement:
    placement_id: str
    r_cm: float
    theta_deg: float
    x_cm: float
    y_cm: float


class SamplingStuck(RuntimeError):
    """Raised when a list cannot be completed within the attempt budget."""

    def __init__(self, list_name: str, accepted: int, target: int, d_min: float) -> None:
        self.list_name = list_name
        self.accepted = accepted
        self.target = target
        self.d_min = d_min
        super().__init__(
            f"stuck filling {list_name!r}: accepted {accepted}/{target} points "
            f"at d_min={d_min} cm before the attempt budget ran out. "
            f"Do NOT relax d_min -- reduce N, shrink d_min deliberately, or enlarge the workspace."
        )


def _ordered(counts: dict[str, int]) -> list[tuple[str, int]]:
    # Largest first; ties broken by name for determinism.
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


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
        budget = _ATTEMPTS_PER_POINT * target
        while len(out[name]) < target:
            if budget <= 0:
                raise SamplingStuck(name, len(out[name]), target, d_min)
            budget -= 1
            r, theta = sample_sector_point(rng, sector)
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
    return out


def global_min_separation_cm(lists: dict[str, list[Placement]]) -> float:
    pts = [(p.x_cm, p.y_cm) for pts in lists.values() for p in pts]
    return min(
        (math.dist(a, b) for i, a in enumerate(pts) for b in pts[i + 1 :]),
        default=math.inf,
    )
