"""S2 --from-summary: turn S1's reach_summary_<date>.json into a Sector. Spec 3."""

from __future__ import annotations

import json

import pytest

from placement_sampler.reach_summary import sector_from_summary

_GOOD = {
    "generated": "2026-09-05T10:00:00",
    "git_commit": "abc1234",
    "fk_method": "handcoded_urdf",
    "fk_validation": {"passed": True, "date": "2026-09-05", "max_error_cm": 0.4},
    "azimuth_frame": "mat",
    "azimuth_offset_deg": 12.0,
    "r_outer_topdown_cm": 30.0,
    "r_inner_cm": 20.0,
    "azimuth_min_deg": -35.0,
    "azimuth_max_deg": 88.0,
}


def _write(tmp_path, data):
    p = tmp_path / "reach_summary_2026-09-05.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


def test_builds_sector_with_margin_applied_to_raw_outer(tmp_path):
    sector, info = sector_from_summary(_write(tmp_path, _GOOD), margin=3.0)
    assert sector.r_inner == 20.0
    assert sector.r_outer == pytest.approx(27.0)  # 30 raw - 3 margin
    assert info.warnings == []
    assert info.azimuth_frame == "mat"
    assert info.source_git_commit == "abc1234"


def test_null_fk_validation_warns_but_still_returns_a_sector(tmp_path):
    data = dict(_GOOD, fk_validation=None)
    sector, info = sector_from_summary(_write(tmp_path, data), margin=3.0)
    assert sector.r_outer == pytest.approx(27.0)
    assert any("fk_validation" in w for w in info.warnings)


def test_failed_fk_validation_warns(tmp_path):
    data = dict(_GOOD, fk_validation={"passed": False, "date": "x", "max_error_cm": 9.9})
    _sector, info = sector_from_summary(_write(tmp_path, data), margin=3.0)
    assert any("tape" in w.lower() or "fallback" in w.lower() for w in info.warnings)


def test_base_azimuth_frame_is_recorded_not_rejected(tmp_path):
    data = dict(_GOOD, azimuth_frame="base", azimuth_offset_deg=None)
    _sector, info = sector_from_summary(_write(tmp_path, data), margin=3.0)
    assert info.azimuth_frame == "base"
    assert any("base" in w.lower() for w in info.warnings)


def test_missing_required_radius_is_an_error(tmp_path):
    data = dict(_GOOD, r_inner_cm=None)
    with pytest.raises(ValueError):
        sector_from_summary(_write(tmp_path, data), margin=3.0)
