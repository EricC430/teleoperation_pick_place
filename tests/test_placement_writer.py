"""S2 output: three dated CSVs + one meta.json, never silently overwritten. Spec 5, 6."""

from __future__ import annotations

import json

import pytest

from placement_sampler.geometry import Sector
from placement_sampler.sampler import sample_lists
from placement_sampler.writer import OutputExists, write_outputs

SECTOR = Sector(r_inner=20.0, r_outer=30.0, theta_min=-35.0, theta_max=88.0)
COUNTS = {"train": 50, "eval-open": 10, "eval-close": 30}


def _lists():
    return sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=20260831)


def _meta():
    return {"seed": 20260831, "d_min": 2.0}


def test_writes_three_dated_csvs_and_one_meta(tmp_path):
    paths = write_outputs(tmp_path, "camp_A", "20260831", _lists(), _meta())
    names = sorted(p.name for p in paths)
    assert names == [
        "camp_A_20260831_eval-close.csv",
        "camp_A_20260831_eval-open.csv",
        "camp_A_20260831_meta.json",
        "camp_A_20260831_train.csv",
    ]


def test_csv_has_the_spec_header_and_two_decimal_places(tmp_path):
    write_outputs(tmp_path, "c", "20260831", _lists(), _meta())
    rows = (tmp_path / "c_20260831_train.csv").read_text(encoding="utf-8").splitlines()
    assert rows[0] == "placement_id,r_cm,theta_deg,x_cm,y_cm"
    assert rows[1].startswith("train_001,")
    fields = rows[1].split(",")
    assert len(fields) == 5
    for value in fields[1:]:
        assert len(value.split(".")[1]) == 2


def test_refuses_to_overwrite_without_force(tmp_path):
    write_outputs(tmp_path, "c", "20260831", _lists(), _meta())
    with pytest.raises(OutputExists):
        write_outputs(tmp_path, "c", "20260831", _lists(), _meta())


def test_force_allows_overwrite(tmp_path):
    write_outputs(tmp_path, "c", "20260831", _lists(), _meta())
    write_outputs(tmp_path, "c", "20260831", _lists(), _meta(), force=True)  # no raise


def test_csv_bytes_are_identical_for_the_same_lists(tmp_path):
    write_outputs(tmp_path / "a", "c", "20260831", _lists(), _meta())
    write_outputs(tmp_path / "b", "c", "20260831", _lists(), _meta())
    for name in ("train", "eval-open", "eval-close"):
        a = (tmp_path / "a" / f"c_20260831_{name}.csv").read_bytes()
        b = (tmp_path / "b" / f"c_20260831_{name}.csv").read_bytes()
        assert a == b


def test_meta_json_records_inputs_and_the_achieved_separation(tmp_path):
    write_outputs(tmp_path, "c", "20260831", _lists(), _meta())
    meta = json.loads((tmp_path / "c_20260831_meta.json").read_text(encoding="utf-8"))
    assert meta["seed"] == 20260831
    assert meta["global_min_separation_cm"] >= 2.0
    assert meta["per_list"]["train"]["n"] == 50
    assert "nearest_neighbour_cm" in meta["per_list"]["train"]
