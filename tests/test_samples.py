"""CSV sample log: fixed schema, flush-on-append, never overwrite."""

import csv

import pytest

from reach_logger import samples


def test_resolve_out_path_unchanged_when_free(tmp_path):
    p = tmp_path / "reach_log_2026-08-31.csv"
    assert samples.resolve_out_path(p) == p


def test_resolve_out_path_bumps_suffix_when_taken(tmp_path):
    p = tmp_path / "reach_log_2026-08-31.csv"
    p.write_text("x")
    p2 = samples.resolve_out_path(p)
    assert p2.name == "reach_log_2026-08-31_2.csv"
    p2.write_text("x")
    assert samples.resolve_out_path(p).name == "reach_log_2026-08-31_3.csv"


def test_writer_starts_with_the_schema_header(tmp_path):
    path = tmp_path / "log.csv"
    with samples.SampleWriter(path):
        pass
    header = next(csv.reader(path.open()))
    assert header == list(samples.COLUMNS)
    assert header[0] == "ts_iso" and header[-1] == "note"


def test_append_is_flushed_to_disk_immediately(tmp_path):
    path = tmp_path / "log.csv"
    with samples.SampleWriter(path) as w:
        w.append(
            samples.SampleRow(
                ts_iso="2026-08-31T10:00:00+08:00",
                sample_type="outer_topdown",
                method="fk",
                radius_cm=31.8,
                ee_x_cm=30.0,
                ee_y_cm=-1.0,
                ee_z_cm=5.0,
                azimuth_base_deg=-32.6,
                azimuth_mat_deg=None,
                shoulder_pan_pos=1024,
                shoulder_lift_pos=0,
                elbow_flex_pos=0,
                wrist_flex_pos=0,
                wrist_roll_pos=0,
                gripper_pos=2048,
                note="cable tight here",
            )
        )
        # Not closed yet: the row must already be on disk (power-loss safety).
        rows = list(csv.DictReader(path.open()))
    assert len(rows) == 1
    assert rows[0]["sample_type"] == "outer_topdown"
    assert rows[0]["radius_cm"] == "31.8"
    assert rows[0]["azimuth_mat_deg"] == ""  # None serialises to empty


def test_rejects_unknown_sample_type():
    with pytest.raises(ValueError):
        samples.SampleRow(
            ts_iso="t",
            sample_type="outer",  # old name, no longer valid
            method="fk",
            radius_cm=1.0,
            ee_x_cm=0,
            ee_y_cm=0,
            ee_z_cm=0,
            azimuth_base_deg=0,
            azimuth_mat_deg=None,
            shoulder_pan_pos=0,
            shoulder_lift_pos=0,
            elbow_flex_pos=0,
            wrist_flex_pos=0,
            wrist_roll_pos=0,
            gripper_pos=0,
            note="",
        )


def test_rejects_unknown_method():
    with pytest.raises(ValueError):
        samples.SampleRow(
            ts_iso="t",
            sample_type="inner",
            method="guess",
            radius_cm=1.0,
            ee_x_cm=0,
            ee_y_cm=0,
            ee_z_cm=0,
            azimuth_base_deg=0,
            azimuth_mat_deg=None,
            shoulder_pan_pos=0,
            shoulder_lift_pos=0,
            elbow_flex_pos=0,
            wrist_flex_pos=0,
            wrist_roll_pos=0,
            gripper_pos=0,
            note="",
        )
