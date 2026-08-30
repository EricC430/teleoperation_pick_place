"""5-pose FK validation gate (spec 5): compare FK-predicted EE (x, y) against
what the human measures off the table for a handful of known poses."""

import pytest

from reach_logger import validation


def _check(name, pred, meas):
    return validation.PoseCheck(
        name=name, joint_rad=[0.0] * 5, predicted_xy_cm=pred, measured_xy_cm=meas
    )


def test_passes_when_every_pose_is_within_tolerance():
    checks = [
        _check("home", (32.4, -0.2), (32.0, 0.0)),
        _check("+J1", (0.2, 32.4), (0.5, 32.0)),
        _check("+J2", (11.3, -0.2), (11.0, -0.5)),
    ]
    report = validation.evaluate(checks, tolerance_cm=1.0, date="2026-08-31")
    assert report.result.passed is True
    assert report.result.date == "2026-08-31"
    assert report.result.max_error_cm == pytest.approx(0.5, abs=0.05)


def test_fails_when_one_pose_exceeds_tolerance():
    checks = [
        _check("home", (32.4, 0.0), (32.4, 0.0)),
        _check("+J1", (0.0, 32.4), (0.0, 30.0)),  # 2.4 cm off
    ]
    report = validation.evaluate(checks, tolerance_cm=1.0, date="2026-08-31")
    assert report.result.passed is False
    assert report.result.max_error_cm == pytest.approx(2.4, abs=0.05)


def test_incomplete_when_a_measurement_is_missing():
    checks = [
        _check("home", (32.4, 0.0), (32.4, 0.0)),
        _check("+J1", (0.0, 32.4), None),
    ]
    report = validation.evaluate(checks, tolerance_cm=1.0, date="2026-08-31")
    assert report.result.passed is False
    assert report.incomplete is True


def test_render_table_lists_every_pose_with_error():
    checks = [
        _check("home", (32.4, 0.0), (32.0, 0.2)),
        _check("+J1", (0.0, 32.4), (0.0, 33.9)),
    ]
    table = validation.evaluate(checks, tolerance_cm=1.0, date="2026-08-31").render_table()
    assert "home" in table and "+J1" in table
    assert "error" in table.lower() or "誤差" in table
    # the +J1 error is 1.5 cm -> the table should show it failing
    assert "1.5" in table


def test_needs_at_least_the_home_pose():
    with pytest.raises(ValueError):
        validation.evaluate([], tolerance_cm=1.0, date="2026-08-31")
