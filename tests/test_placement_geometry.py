"""Geometry for S2: area-uniform annular-sector sampling + margin. Spec 2-1, 3."""

from __future__ import annotations

import math
import random

import pytest

from placement_sampler.geometry import (
    Sector,
    apply_margin,
    area_uniform_radius,
    polar_to_xy,
    sample_sector_point,
)


def test_polar_to_xy_at_90_degrees_is_pure_y():
    x, y = polar_to_xy(10.0, 90.0)
    assert math.isclose(x, 0.0, abs_tol=1e-9)
    assert math.isclose(y, 10.0, abs_tol=1e-9)


def test_polar_to_xy_at_180_degrees_is_negative_x():
    x, y = polar_to_xy(7.0, 180.0)
    assert math.isclose(x, -7.0, abs_tol=1e-9)
    assert math.isclose(y, 0.0, abs_tol=1e-9)


def test_area_uniform_radius_stays_in_bounds():
    rng = random.Random(1)
    for _ in range(1000):
        r = area_uniform_radius(rng, 20.0, 30.0)
        assert 20.0 <= r <= 30.0


def test_area_uniform_radius_is_area_uniform_not_radius_uniform():
    # For area-uniform sampling, half the points fall inside the radius that
    # splits the annulus into two equal areas: r_mid = sqrt((r_in^2+r_out^2)/2).
    # Radius-uniform sampling would put ~57% inside that radius for 20..30.
    rng = random.Random(12345)
    r_in, r_out = 20.0, 30.0
    r_mid = math.sqrt((r_in**2 + r_out**2) / 2)
    n = 20000
    inside = sum(area_uniform_radius(rng, r_in, r_out) < r_mid for _ in range(n))
    assert abs(inside / n - 0.5) < 0.02


def test_sample_sector_point_respects_angular_bounds():
    rng = random.Random(2)
    sector = Sector(r_inner=20.0, r_outer=30.0, theta_min=-35.0, theta_max=88.0)
    for _ in range(1000):
        r, theta = sample_sector_point(rng, sector)
        assert 20.0 <= r <= 30.0
        assert -35.0 <= theta <= 88.0


def test_apply_margin_shrinks_outer_radius_and_both_angles_but_not_inner():
    sector = Sector(r_inner=20.0, r_outer=30.0, theta_min=-35.0, theta_max=88.0)
    out = apply_margin(sector, margin=3.0)
    assert out.r_inner == 20.0  # inner is never margined (D024 2026-08-31)
    assert math.isclose(out.r_outer, 27.0)
    ang = math.degrees(3.0 / 27.0)
    assert math.isclose(out.theta_min, -35.0 + ang)
    assert math.isclose(out.theta_max, 88.0 - ang)


def test_apply_margin_zero_is_identity():
    sector = Sector(r_inner=20.0, r_outer=30.0, theta_min=-35.0, theta_max=88.0)
    out = apply_margin(sector, margin=0.0)
    assert out == sector


def test_apply_margin_rejects_margin_that_collapses_the_sector():
    sector = Sector(r_inner=20.0, r_outer=22.0, theta_min=-1.0, theta_max=1.0)
    with pytest.raises(ValueError):
        apply_margin(sector, margin=5.0)
