"""Orchestration bits of the reach logger that don't need hardware or a keyboard."""

import json

import pytest

from reach_logger import joints, session
from reach_logger.samples import SampleWriter


def _ticks(**over):
    d = dict.fromkeys(("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"), 0.0)
    d.update(over)
    return d


CALIB = joints.identity_calibration()
NOW = "2026-08-31T10:00:00+08:00"


def test_key_to_type_covers_the_five_sample_keys():
    assert session.KEY_TO_TYPE == {
        "o": "outer_topdown",
        "s": "outer_side",
        "i": "inner",
        "a": "azimuth_limit",
        "r": "reference",
    }


def test_make_sample_row_fk_fills_radius_and_azimuth_from_fk():
    row = session.make_sample_row(
        "o", _ticks(shoulder_pan=1024.0), CALIB, method="fk", now=NOW
    )
    assert row.sample_type == "outer_topdown"
    assert row.method == "fk"
    # 1024 ticks on joint1 == +90deg pan; reach is unchanged, azimuth ~ +89.7
    assert row.radius_cm == pytest.approx(32.4134, abs=1e-2)
    assert row.azimuth_base_deg == pytest.approx(89.717, abs=0.05)
    assert row.shoulder_pan_pos == 1024.0
    assert row.azimuth_mat_deg is None


def test_make_sample_row_tape_uses_typed_radius():
    row = session.make_sample_row(
        "i", _ticks(), CALIB, method="tape", now=NOW, tape_cm=18.5
    )
    assert row.sample_type == "inner"
    assert row.method == "tape"
    assert row.radius_cm == pytest.approx(18.5)


def test_azimuth_limit_row_has_no_radius():
    row = session.make_sample_row("a", _ticks(shoulder_pan=512.0), CALIB, method="fk", now=NOW)
    assert row.sample_type == "azimuth_limit"
    assert row.radius_cm is None
    assert row.azimuth_base_deg is not None


def test_reference_row_stores_the_typed_mat_angle():
    row = session.make_sample_row(
        "r", _ticks(shoulder_pan=256.0), CALIB, method="fk", now=NOW, mat_deg=0.0
    )
    assert row.sample_type == "reference"
    assert row.azimuth_mat_deg == pytest.approx(0.0)


def test_plan_text_names_the_paths_and_mode():
    text = session.plan_text(
        config="configs/teleoperate_omx.yaml",
        urdf="assets/omx_f/omx_f.urdf",
        out="analysis/reach_log_2026-08-31.csv",
        mode="teleop",
    )
    assert "configs/teleoperate_omx.yaml" in text
    assert "assets/omx_f/omx_f.urdf" in text
    assert "analysis/reach_log_2026-08-31.csv" in text
    assert "teleop" in text


def test_finalize_writes_json_and_plot(tmp_path):
    csv_path = tmp_path / "reach_log_2026-08-31.csv"
    rows = [
        session.make_sample_row("o", _ticks(shoulder_pan=100.0), CALIB, method="fk", now=NOW),
        session.make_sample_row("i", _ticks(), CALIB, method="fk", now=NOW),
    ]
    with SampleWriter(csv_path) as w:
        for r in rows:
            w.append(r)

    result = session.finalize(rows, csv_path=csv_path, fk_validation=None, git_commit="abc1234", now=NOW)

    assert result.summary_json_path.exists()
    assert result.plot_path.exists()
    data = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    assert data["source_csv"].endswith("reach_log_2026-08-31.csv")
    assert data["git_commit"] == "abc1234"
    assert "margin" in data["notes"].lower()
