"""FK for the omx_f arm, hand-coded from assets/omx_f/omx_f.urdf (D026).

Oracle values are closed-form, computed by hand from the URDF link origins, because
placo (the usual FK oracle) does not install on Windows.
"""

import math

import pytest

from reach_logger import fk


def test_zero_pose_ee_position_matches_sum_of_urdf_origins():
    # All joints at 0 -> every rotation is identity -> EE position is just the
    # sum of the joint origin translations plus the fixed end-effector offset.
    x, y, z = fk.ee_position_m([0.0, 0.0, 0.0, 0.0, 0.0])
    # x: -0.01125 + 0.0415 + 0.162 + 0.0287 + 0.09193
    assert x == pytest.approx(0.31288, abs=1e-9)
    # y: only the fixed end-effector offset contributes
    assert y == pytest.approx(-0.0016, abs=1e-9)
    # z: 0.034 + 0.0635 + 0.11315
    assert z == pytest.approx(0.21065, abs=1e-9)


def test_zero_pose_reach_is_about_32_cm():
    # radius = distance in the base XY plane from the joint1 (shoulder_pan) axis,
    # whose XY intersection is (-0.01125, 0).
    assert fk.reach_cm([0.0, 0.0, 0.0, 0.0, 0.0]) == pytest.approx(32.4134, abs=1e-3)


def test_shoulder_pan_90deg_rotates_azimuth_by_90_and_keeps_reach():
    q = [math.pi / 2, 0.0, 0.0, 0.0, 0.0]
    assert fk.reach_cm(q) == pytest.approx(fk.reach_cm([0.0] * 5), abs=1e-6)
    assert fk.azimuth_base_deg(q) == pytest.approx(
        fk.azimuth_base_deg([0.0] * 5) + 90.0, abs=1e-3
    )


def test_shoulder_lift_90deg_closed_form():
    # Only joint2 (y-axis) rotates. At zero pose the EE sits at (0.32413, -0.0016,
    # 0.11315) relative to the joint2 origin (-0.01125, 0, 0.0975) in the base frame.
    # Ry(90deg): (x, z) -> (z, -x)  ->  (0.11315, -0.0016, -0.32413) relative,
    # so base position is (-0.01125 + 0.11315, -0.0016, 0.0975 - 0.32413).
    x, y, z = fk.ee_position_m([0.0, math.pi / 2, 0.0, 0.0, 0.0])
    assert x == pytest.approx(0.1019, abs=1e-4)
    assert y == pytest.approx(-0.0016, abs=1e-9)
    assert z == pytest.approx(-0.22663, abs=1e-4)
    assert fk.reach_cm([0.0, math.pi / 2, 0.0, 0.0, 0.0]) == pytest.approx(11.316, abs=1e-3)


def test_azimuth_stays_within_180():
    for q1 in (-math.pi, -2.0, 0.0, 2.0, math.pi, 5.0):
        az = fk.azimuth_base_deg([q1, 0.3, -0.4, 0.2, 0.1])
        assert -180.0 < az <= 180.0


def test_wrong_joint_count_raises():
    with pytest.raises(ValueError):
        fk.ee_transform([0.0, 0.0, 0.0])
