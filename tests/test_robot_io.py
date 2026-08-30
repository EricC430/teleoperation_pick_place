"""Hardware boundary for the reach logger (spec 4).

The real OMX path can only be exercised at the lab; here we pin down the fake,
the dry-run guard, and the motor-set validation that turns an unplugged arm into
a clean message instead of a traceback.
"""

import pytest

from reach_logger import robot_io

ALL_MOTORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _ticks(**overrides):
    d = dict.fromkeys(ALL_MOTORS, 0.0)
    d.update(overrides)
    return d


def test_fake_robot_replays_scripted_ticks():
    fake = robot_io.FakeReachRobot([_ticks(shoulder_pan=100.0), _ticks(shoulder_pan=200.0)])
    with fake as r:
        assert r.read_ticks()["shoulder_pan"] == 100.0
        r.step()
        assert r.read_ticks()["shoulder_pan"] == 200.0


def test_fake_robot_step_count_is_observable():
    fake = robot_io.FakeReachRobot([_ticks()])
    with fake as r:
        r.step()
        r.step()
    assert fake.steps == 2


def test_dry_run_robot_never_reads_hardware():
    with robot_io.DryRunRobot() as r:
        with pytest.raises(robot_io.DryRunError):
            r.read_ticks()
        with pytest.raises(robot_io.DryRunError):
            r.step()


def test_require_motors_accepts_the_full_set():
    assert robot_io.require_motors(_ticks(shoulder_pan=12.0))["shoulder_pan"] == 12.0


def test_require_motors_rejects_a_missing_motor_with_a_clear_message():
    partial = _ticks()
    del partial["wrist_roll"]
    with pytest.raises(robot_io.RobotIOError) as exc:
        robot_io.require_motors(partial)
    assert "wrist_roll" in str(exc.value)


def test_build_robot_dry_run_returns_dry_run_robot():
    r = robot_io.build_robot(config_path="configs/teleoperate_omx.yaml", mode="teleop", dry_run=True)
    assert isinstance(r, robot_io.DryRunRobot)


def test_build_robot_rejects_unknown_mode():
    with pytest.raises(ValueError):
        robot_io.build_robot(config_path="x.yaml", mode="wiggle", dry_run=True)
