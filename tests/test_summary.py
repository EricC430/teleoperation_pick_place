"""Reach-log rows -> terminal summary + reach_summary_<date>.json (spec 9)."""

import pytest

from reach_logger import summary
from reach_logger.samples import SampleRow


def _row(sample_type, *, radius=None, az_base=0.0, az_mat=None, method="fk", note=""):
    return SampleRow(
        ts_iso="2026-08-31T10:00:00+08:00",
        sample_type=sample_type,
        method=method,
        radius_cm=radius,
        ee_x_cm=0.0,
        ee_y_cm=0.0,
        ee_z_cm=0.0,
        azimuth_base_deg=az_base,
        azimuth_mat_deg=az_mat,
        shoulder_pan_pos=0,
        shoulder_lift_pos=0,
        elbow_flex_pos=0,
        wrist_flex_pos=0,
        wrist_roll_pos=0,
        gripper_pos=0,
        note=note,
    )


META = dict(
    source_csv="analysis/reach_log_2026-08-31.csv",
    git_commit="abc1234",
    generated="2026-08-31T18:00:00+08:00",
)


def test_sample_counts():
    rows = [
        _row("outer_topdown", radius=30.0),
        _row("outer_topdown", radius=31.0),
        _row("inner", radius=19.0),
        _row("azimuth_limit", az_base=-35.0),
    ]
    s = summary.build_summary(rows, fk_validation=None, **META)
    assert s.sample_counts == {
        "outer_topdown": 2,
        "outer_side": 0,
        "inner": 1,
        "azimuth_limit": 1,
        "reference": 0,
    }


def test_r_outer_topdown_is_the_worst_direction_with_its_azimuth():
    rows = [
        _row("outer_topdown", radius=33.0, az_base=10.0),
        _row("outer_topdown", radius=28.5, az_base=-20.0),  # most limiting
        _row("outer_topdown", radius=31.0, az_base=40.0),
    ]
    s = summary.build_summary(rows, fk_validation=None, **META)
    assert s.r_outer_topdown_cm == pytest.approx(28.5)
    assert s.r_outer_topdown_worst_azimuth_deg == pytest.approx(-20.0)


def test_r_inner_is_the_largest_inner_sample():
    rows = [
        _row("inner", radius=18.0),
        _row("inner", radius=19.7),
        _row("inner", radius=17.2),
    ]
    s = summary.build_summary(rows, fk_validation=None, **META)
    assert s.r_inner_cm == pytest.approx(19.7)


def test_r_outer_side_is_informational_max_and_may_be_absent():
    s = summary.build_summary(
        [_row("outer_topdown", radius=30.0)], fk_validation=None, **META
    )
    assert s.r_outer_side_cm is None

    rows = [_row("outer_side", radius=42.0), _row("outer_side", radius=43.1)]
    s2 = summary.build_summary(rows, fk_validation=None, **META)
    assert s2.r_outer_side_cm == pytest.approx(43.1)


def test_azimuth_range_from_azimuth_limit_samples():
    rows = [
        _row("azimuth_limit", az_base=-35.2),
        _row("azimuth_limit", az_base=88.6),
        _row("azimuth_limit", az_base=10.0),
    ]
    s = summary.build_summary(rows, fk_validation=None, **META)
    assert s.azimuth_min_deg == pytest.approx(-35.2)
    assert s.azimuth_max_deg == pytest.approx(88.6)


def test_reference_sample_sets_mat_frame_and_offset():
    # reference row: FK azimuth of the pose is +12, human read the mat angle as 0.
    rows = [
        _row("reference", az_base=12.0, az_mat=0.0),
        _row("azimuth_limit", az_base=-20.0),
    ]
    s = summary.build_summary(rows, fk_validation=None, **META)
    assert s.azimuth_frame == "mat"
    assert s.azimuth_offset_deg == pytest.approx(-12.0)
    # azimuth_limit reported in mat frame: -20 + (-12) = -32
    assert s.azimuth_min_deg == pytest.approx(-32.0)


def test_no_reference_means_base_frame():
    s = summary.build_summary(
        [_row("azimuth_limit", az_base=-20.0)], fk_validation=None, **META
    )
    assert s.azimuth_frame == "base"
    assert s.azimuth_offset_deg is None
    assert s.azimuth_min_deg == pytest.approx(-20.0)


def test_as_dict_has_spec_keys_and_margin_not_applied():
    rows = [
        _row("outer_topdown", radius=31.8, az_base=-20.0),
        _row("inner", radius=18.9),
        _row("azimuth_limit", az_base=-35.2),
        _row("azimuth_limit", az_base=88.6),
    ]
    d = summary.build_summary(rows, fk_validation=None, **META).as_dict()
    for key in (
        "generated", "source_csv", "git_commit", "fk_method", "fk_validation",
        "azimuth_frame", "azimuth_offset_deg", "r_outer_topdown_cm",
        "r_outer_topdown_worst_azimuth_deg", "r_outer_side_cm", "r_inner_cm",
        "azimuth_min_deg", "azimuth_max_deg", "sample_counts", "notes",
    ):
        assert key in d
    assert d["r_outer_topdown_cm"] == pytest.approx(31.8)  # margin not subtracted
    assert "margin" in d["notes"].lower()


def test_render_text_warns_margin_is_the_humans_call():
    rows = [
        _row("outer_topdown", radius=31.8, az_base=-20.0),
        _row("inner", radius=18.9),
    ]
    text = summary.build_summary(rows, fk_validation=None, **META).render_text()
    assert "31.8" in text
    assert "margin" in text.lower()
    assert "⚠" in text
