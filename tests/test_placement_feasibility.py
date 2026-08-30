"""S2 feasibility gate: two packing ceilings, not one. Spec 2-3."""

from __future__ import annotations

import math

from placement_sampler.feasibility import Verdict, expected_mean_nn_cm, feasibility
from placement_sampler.geometry import Sector

HALF_DISK = Sector(r_inner=20.0, r_outer=30.0, theta_min=0.0, theta_max=180.0)


def test_half_sector_area_matches_hand_calc():
    # 0.5 * pi * (30^2 - 20^2) = 0.5 * pi * 500
    assert math.isclose(HALF_DISK.area_cm2, 0.5 * math.pi * 500.0)


def test_tiny_n_is_ok():
    res = feasibility(HALF_DISK, d_min=2.0, n_total=10)
    assert res.verdict is Verdict.OK
    assert res.n_hex > res.n_rsa > 10


def test_n_above_hex_ceiling_is_geometrically_impossible():
    res = feasibility(HALF_DISK, d_min=2.0, n_total=100_000)
    assert res.verdict is Verdict.INFEASIBLE_GEOMETRY


def test_n_between_rsa_and_hex_is_infeasible_by_random_sampling():
    res = feasibility(HALF_DISK, d_min=2.0, n_total=1)
    n = int((res.n_rsa + res.n_hex) / 2)
    res = feasibility(HALF_DISK, d_min=2.0, n_total=n)
    assert res.verdict is Verdict.INFEASIBLE_RANDOM


def test_n_just_below_rsa_ceiling_warns_but_proceeds():
    probe = feasibility(HALF_DISK, d_min=2.0, n_total=1)
    n = int(0.85 * probe.n_rsa)
    res = feasibility(HALF_DISK, d_min=2.0, n_total=n)
    assert res.verdict is Verdict.OK_WARN


def test_default_campaign_90_points_at_2cm_in_half_sector_is_feasible_but_warned():
    # D024 2026-08-30: at the provisional workspace this is right at the edge --
    # mean nearest-neighbour ~1.5 cm, just under d_min. Feasible, but not comfortably.
    res = feasibility(HALF_DISK, d_min=2.0, n_total=90)
    assert res.verdict is Verdict.OK_WARN
    assert res.verdict.is_feasible
    assert res.mean_nn_cm < 2.0


def test_expected_mean_nn_shrinks_as_points_grow():
    far = expected_mean_nn_cm(n=10, area_cm2=HALF_DISK.area_cm2)
    near = expected_mean_nn_cm(n=90, area_cm2=HALF_DISK.area_cm2)
    assert far > near > 0
