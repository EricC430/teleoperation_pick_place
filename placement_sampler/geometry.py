"""Area-uniform sampling over an annular sector, plus the workspace margin.

Spec: docs/specs/S2_placement_sampler.md 2-1 (the sqrt trick), 3 (--margin).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Sector:
    """An annular sector in the mat frame. Angles in degrees, radii in cm."""

    r_inner: float
    r_outer: float
    theta_min: float
    theta_max: float

    def __post_init__(self) -> None:
        if not 0 <= self.r_inner < self.r_outer:
            raise ValueError(f"need 0 <= r_inner < r_outer, got {self.r_inner}, {self.r_outer}")
        if self.theta_min >= self.theta_max:
            raise ValueError(f"need theta_min < theta_max, got {self.theta_min}, {self.theta_max}")

    @property
    def turn_fraction(self) -> float:
        return (self.theta_max - self.theta_min) / 360.0

    @property
    def area_cm2(self) -> float:
        return self.turn_fraction * math.pi * (self.r_outer**2 - self.r_inner**2)


def polar_to_xy(r_cm: float, theta_deg: float) -> tuple[float, float]:
    rad = math.radians(theta_deg)
    return r_cm * math.cos(rad), r_cm * math.sin(rad)


def area_uniform_radius(rng: random.Random, r_inner: float, r_outer: float) -> float:
    """Radius whose distribution gives uniform *area* density (not uniform r)."""
    return math.sqrt(rng.uniform(r_inner**2, r_outer**2))


def sample_sector_point(rng: random.Random, sector: Sector) -> tuple[float, float]:
    r = area_uniform_radius(rng, sector.r_inner, sector.r_outer)
    theta = rng.uniform(sector.theta_min, sector.theta_max)
    return r, theta


def apply_margin(sector: Sector, margin: float) -> Sector:
    """r_outer -= margin; shrink both azimuth edges by degrees(margin / r_outer_new).

    r_inner is left untouched (D024 2026-08-31): the failure mechanisms cluster at
    the outer boundary and the constrained azimuth, not the inner edge. The angular
    shrink uses the *post-margin* r_outer, the conservative choice.
    """
    if margin < 0:
        raise ValueError(f"margin must be >= 0, got {margin}")
    if margin == 0:
        return sector
    r_outer = sector.r_outer - margin
    if r_outer <= sector.r_inner:
        raise ValueError(
            f"margin {margin} cm leaves no annulus: r_outer {sector.r_outer} - margin "
            f"<= r_inner {sector.r_inner}"
        )
    ang = math.degrees(margin / r_outer)
    theta_min = sector.theta_min + ang
    theta_max = sector.theta_max - ang
    if theta_min >= theta_max:
        raise ValueError(
            f"margin {margin} cm collapses the azimuth sector "
            f"[{sector.theta_min}, {sector.theta_max}] (angular margin {ang:.1f} deg each side)"
        )
    return Sector(r_inner=sector.r_inner, r_outer=r_outer, theta_min=theta_min, theta_max=theta_max)
