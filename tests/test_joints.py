"""Raw Dynamixel ticks -> FK joint-angle vector (radians)."""

import math

import pytest

from reach_logger import joints


def test_ticks_to_rad_quarter_turn():
    assert joints.ticks_to_rad(1024) == pytest.approx(math.pi / 2)


def test_ticks_to_rad_wraps_past_half_turn_to_negative():
    # 3072/4096 of a turn == 270deg == -90deg after wrapping to (-pi, pi].
    assert joints.ticks_to_rad(3072) == pytest.approx(-math.pi / 2)


def test_ticks_to_rad_handles_negative_extended_position_ticks():
    assert joints.ticks_to_rad(-1024) == pytest.approx(-math.pi / 2)


def test_joint_vector_drops_gripper_and_orders_for_fk():
    readings = {
        "shoulder_pan": 1024,
        "shoulder_lift": 0,
        "elbow_flex": 0,
        "wrist_flex": 0,
        "wrist_roll": 0,
        "gripper": 2048,
    }
    vec = joints.joint_vector(readings, joints.identity_calibration())
    assert len(vec) == 5
    assert vec[0] == pytest.approx(math.pi / 2)
    assert vec[1:] == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_joint_vector_applies_sign_and_offset():
    calib = joints.JointCalibration(
        offset_rad=[0.0, math.pi / 2, 0.0, 0.0, 0.0],
        sign=[-1, 1, 1, 1, 1],
    )
    readings = dict.fromkeys(joints.FK_MOTOR_ORDER, 0) | {"gripper": 0, "shoulder_pan": 1024}
    vec = joints.joint_vector(readings, calib)
    # pan: sign -1 applied to +90deg  -> -90deg
    assert vec[0] == pytest.approx(-math.pi / 2)
    # lift: raw 0, minus offset pi/2  -> -pi/2
    assert vec[1] == pytest.approx(-math.pi / 2)


def test_joint_vector_missing_motor_raises():
    with pytest.raises(KeyError):
        joints.joint_vector({"shoulder_pan": 0}, joints.identity_calibration())
