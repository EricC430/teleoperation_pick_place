"""S2 acceptance stat: is r^2 uniform over U(r_in^2, r_out^2)? Spec 6."""

from __future__ import annotations

import math
import random

from placement_sampler.stats import ks_uniform_pvalue

LO, HI = 20.0**2, 30.0**2


def test_genuinely_uniform_sample_is_not_rejected():
    rng = random.Random(0)
    vals = [rng.uniform(LO, HI) for _ in range(2000)]
    assert ks_uniform_pvalue(vals, LO, HI) > 0.05


def test_radius_uniform_sample_is_rejected_as_non_uniform_in_r_squared():
    # r ~ U(20,30) (the classic bug) makes r^2 pile up toward LO.
    rng = random.Random(0)
    vals = [rng.uniform(20.0, 30.0) ** 2 for _ in range(2000)]
    assert ks_uniform_pvalue(vals, LO, HI) < 1e-3


def test_area_uniform_sample_passes():
    rng = random.Random(1)
    vals = [math.sqrt(rng.uniform(LO, HI)) ** 2 for _ in range(2000)]
    assert ks_uniform_pvalue(vals, LO, HI) > 0.05


def test_pvalue_is_bounded_between_0_and_1():
    rng = random.Random(2)
    vals = [rng.uniform(LO, HI) for _ in range(50)]
    p = ks_uniform_pvalue(vals, LO, HI)
    assert 0.0 <= p <= 1.0
