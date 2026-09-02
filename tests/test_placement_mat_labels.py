"""S3 short labels for the single-page print-shop mat: train_007 -> t7 etc. Spec 3."""

from __future__ import annotations

from placement_mat.labels import label_map_rows, short_label


def test_train_id_becomes_t_plus_unpadded_int():
    assert short_label("train_007") == "t7"


def test_eval_open_id_becomes_o():
    assert short_label("eval-open_002") == "o2"


def test_eval_close_id_becomes_c():
    assert short_label("eval-close_010") == "c10"


def test_unknown_prefix_is_returned_unchanged():
    assert short_label("foo_003") == "foo_003"


def test_label_map_shifts_x_by_datum_offset_and_keeps_both_frames():
    rows = label_map_rows([("train_001", 30.0, -5.0)], datum_offset=4.0)
    assert rows == [
        {
            "short_id": "t1",
            "placement_id": "train_001",
            "x_pan_cm": 30.0,
            "y_pan_cm": -5.0,
            "x_mat_cm": 26.0,
            "y_mat_cm": -5.0,
        }
    ]
