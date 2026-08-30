"""Can dart-throwing actually place N points d_min apart in this sector?

Spec: docs/specs/S2_placement_sampler.md 2-3.

Two ceilings, because the perfect-packing one lies:
  N_hex  = A / (0.866 * d^2)   -- flawless hexagonal lattice, unreachable by chance
  N_rsa  = 0.55 * N_hex        -- random sequential adsorption jams around here
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from placement_sampler.geometry import Sector

_HEX = 0.866  # sin(60 deg): area of the rhombus cell per disk in a hex lattice
_RSA_FRACTION = 0.55  # random-sequential-adsorption jamming, as a fraction of N_hex
_WARN_FRACTION = 0.7  # below this fraction of N_rsa: proceed clean; above: warn


class Verdict(Enum):
    OK = "ok"
    OK_WARN = "ok_warn"
    INFEASIBLE_RANDOM = "infeasible_random"
    INFEASIBLE_GEOMETRY = "infeasible_geometry"

    @property
    def is_feasible(self) -> bool:
        return self in (Verdict.OK, Verdict.OK_WARN)


@dataclass(frozen=True)
class Feasibility:
    verdict: Verdict
    n_total: int
    n_hex: float
    n_rsa: float
    area_cm2: float
    d_min: float
    mean_nn_cm: float
    message: str


def expected_mean_nn_cm(n: int, area_cm2: float) -> float:
    """Mean nearest-neighbour distance for n points thrown uniformly at random."""
    if n <= 0:
        return math.inf
    return 0.5 / math.sqrt(n / area_cm2)


def feasibility(sector: Sector, *, d_min: float, n_total: int) -> Feasibility:
    if d_min <= 0:
        raise ValueError(f"d_min must be > 0, got {d_min}")
    area = sector.area_cm2
    n_hex = area / (_HEX * d_min**2)
    n_rsa = _RSA_FRACTION * n_hex
    mean_nn = expected_mean_nn_cm(n_total, area)

    if n_total > n_hex:
        verdict = Verdict.INFEASIBLE_GEOMETRY
        msg = (
            f"{n_total} points cannot fit {d_min} cm apart in {area:.0f} cm^2 "
            f"by any arrangement (hex-packing ceiling {n_hex:.0f})."
        )
    elif n_total > n_rsa:
        verdict = Verdict.INFEASIBLE_RANDOM
        msg = (
            f"{n_total} points is possible in theory (<= {n_hex:.0f}) but not by random "
            f"sampling (realistic ceiling {n_rsa:.0f}). Reduce N or d_min, or enlarge the workspace."
        )
    elif n_total > _WARN_FRACTION * n_rsa:
        verdict = Verdict.OK_WARN
        msg = (
            f"{n_total} points is close to the realistic ceiling {n_rsa:.0f}: sampling may "
            f"be slow or fail late."
        )
    else:
        verdict = Verdict.OK
        msg = f"{n_total} points, realistic ceiling {n_rsa:.0f}: fine."

    return Feasibility(
        verdict=verdict,
        n_total=n_total,
        n_hex=n_hex,
        n_rsa=n_rsa,
        area_cm2=area,
        d_min=d_min,
        mean_nn_cm=mean_nn,
        message=msg,
    )


def feasibility_table(f: Feasibility) -> str:
    """Human-readable block for --dry-run and for the failure exit."""
    return "\n".join(
        [
            f"  usable area      A       = {f.area_cm2:8.1f} cm^2",
            f"  hex ceiling      N_hex   = {f.n_hex:8.1f}",
            f"  realistic ceil.  N_rsa   = {f.n_rsa:8.1f}",
            f"  requested        N_total = {f.n_total:8d}",
            f"  expected mean nearest-neighbour = {f.mean_nn_cm:.2f} cm  (d_min = {f.d_min:.2f} cm)",
            f"  verdict: {f.verdict.value} -- {f.message}",
        ]
    )
