"""Polar reach plot (spec 9(2)) - a visual check before the human picks a margin."""

import matplotlib

matplotlib.use("Agg")

import pytest

from reach_logger import plot, summary
from reach_logger.samples import SampleRow

META = dict(
    source_csv="analysis/reach_log_2026-08-31.csv",
    git_commit="abc1234",
    generated="2026-08-31T18:00:00+08:00",
)


def _row(sample_type, *, radius=None, az_base=0.0, az_mat=None):
    return SampleRow(
        ts_iso="t",
        sample_type=sample_type,
        method="fk",
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
        note="",
    )


def _summary(rows):
    return summary.build_summary(rows, fk_validation=None, **META)


def test_build_figure_has_a_polar_axes():
    rows = [_row("outer_topdown", radius=31.0, az_base=-20.0), _row("inner", radius=19.0)]
    fig = plot.build_figure(rows, _summary(rows))
    assert fig.axes
    assert fig.axes[0].name == "polar"


def test_uncalibrated_title_when_no_reference():
    rows = [_row("azimuth_limit", az_base=-20.0)]
    fig = plot.build_figure(rows, _summary(rows))
    assert "UNCALIBRATED" in _all_text(fig)


def test_no_uncalibrated_marker_when_reference_present():
    rows = [
        _row("reference", az_base=12.0, az_mat=0.0),
        _row("outer_topdown", radius=31.0, az_base=-20.0),
    ]
    fig = plot.build_figure(rows, _summary(rows))
    assert "UNCALIBRATED" not in _all_text(fig)


def test_save_plot_writes_a_png(tmp_path):
    rows = [_row("outer_topdown", radius=31.0, az_base=-20.0), _row("inner", radius=19.0)]
    out = tmp_path / "reach_plot_2026-08-31.png"
    returned = plot.save_plot(rows, _summary(rows), out)
    assert returned == out
    assert out.exists() and out.stat().st_size > 0


def test_empty_rows_do_not_crash():
    fig = plot.build_figure([], _summary([]))
    assert fig.axes


def _all_text(fig) -> str:
    parts = []
    if fig._suptitle is not None:
        parts.append(fig._suptitle.get_text())
    for ax in fig.axes:
        parts.append(ax.get_title())
        for t in ax.texts:
            parts.append(t.get_text())
    return " ".join(parts)
