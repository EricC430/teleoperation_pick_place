"""S2 core: dart-throwing three frozen lists with a global d_min. Spec 4."""

from __future__ import annotations

import math

import pytest

from placement_sampler.geometry import Sector
from placement_sampler.sampler import SamplingStuck, equal_area_cells, sample_lists

# A sector with real headroom for 90 points at d_min 2. The provisional
# 123-deg sector is near saturation (see the feasibility check / D024) and is
# covered by test_impossible_request_raises_sampling_stuck instead.
SECTOR = Sector(r_inner=20.0, r_outer=30.0, theta_min=-90.0, theta_max=90.0)
COUNTS = {"train": 50, "eval-open": 10, "eval-close": 30}


def _cell_area(cell):
    a0, a1, t0, t1 = cell
    return math.radians(t1 - t0) / 2.0 * (a1 - a0)


def test_equal_area_cells_are_all_the_same_area_and_cover_the_sector():
    cells = equal_area_cells(50, SECTOR)
    assert len(cells) >= 50
    areas = [_cell_area(c) for c in cells]
    assert max(areas) - min(areas) < 1e-9
    # radius^2 range and angle range are fully covered
    assert min(c[0] for c in cells) == SECTOR.r_inner**2
    assert max(c[1] for c in cells) == SECTOR.r_outer**2
    assert min(c[2] for c in cells) == SECTOR.theta_min
    assert max(c[3] for c in cells) == SECTOR.theta_max


def test_stratified_fills_every_coarse_quadrat():
    # 12 equal-area quadrats (3 radius bands x 4 azimuth bands); with 50 points
    # stratified, none should be empty -- that is the whole point of the change.
    lists = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=20260831)
    band_r = [SECTOR.r_inner**2 + k * (SECTOR.r_outer**2 - SECTOR.r_inner**2) / 3 for k in range(4)]
    band_t = [SECTOR.theta_min + k * (SECTOR.theta_max - SECTOR.theta_min) / 4 for k in range(5)]
    counts = {}
    for p in lists["train"]:
        i = max(0, min(2, next(k for k in range(3) if p.r_cm**2 <= band_r[k + 1] + 1e-6)))
        j = max(0, min(3, next(k for k in range(4) if p.theta_deg <= band_t[k + 1] + 1e-6)))
        counts[(i, j)] = counts.get((i, j), 0) + 1
    assert len(counts) == 12
    assert min(counts.values()) >= 1


def _all_points(lists):
    return [p for pts in lists.values() for p in pts]


def test_returns_exactly_the_requested_counts_per_list():
    lists = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=20260831)
    assert {k: len(v) for k, v in lists.items()} == COUNTS


def test_placement_ids_are_prefixed_and_1_indexed_in_acceptance_order():
    lists = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=1)
    assert [p.placement_id for p in lists["train"][:3]] == [
        "train_001",
        "train_002",
        "train_003",
    ]
    assert lists["eval-close"][-1].placement_id == "eval-close_030"


def test_every_pair_across_all_three_lists_is_at_least_d_min_apart():
    d_min = 2.0
    lists = sample_lists(SECTOR, d_min=d_min, counts=COUNTS, seed=7)
    pts = _all_points(lists)
    worst = min(
        math.dist((a.x_cm, a.y_cm), (b.x_cm, b.y_cm))
        for i, a in enumerate(pts)
        for b in pts[i + 1 :]
    )
    assert worst >= d_min - 1e-9


def test_all_points_lie_inside_the_sector():
    lists = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=3)
    for p in _all_points(lists):
        assert SECTOR.r_inner <= p.r_cm <= SECTOR.r_outer
        assert SECTOR.theta_min <= p.theta_deg <= SECTOR.theta_max


def test_same_seed_reproduces_identical_coordinates():
    a = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=42)
    b = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=42)
    assert [(p.r_cm, p.theta_deg) for p in a["train"]] == [
        (p.r_cm, p.theta_deg) for p in b["train"]
    ]
    assert [(p.r_cm, p.theta_deg) for p in a["eval-close"]] == [
        (p.r_cm, p.theta_deg) for p in b["eval-close"]
    ]


def test_different_seed_gives_different_coordinates():
    a = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=1)
    b = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=2)
    assert [p.r_cm for p in a["train"]] != [p.r_cm for p in b["train"]]


def test_train_is_sampled_before_eval_so_its_points_are_seed_stable_across_eval_sizes():
    # Larger lists are thrown first (spec 4-2): changing an eval count must not
    # move a single train point.
    a = sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=99)
    b = sample_lists(
        SECTOR, d_min=2.0, counts={"train": 50, "eval-open": 10, "eval-close": 5}, seed=99
    )
    assert [(p.r_cm, p.theta_deg) for p in a["train"]] == [
        (p.r_cm, p.theta_deg) for p in b["train"]
    ]


def test_impossible_request_raises_sampling_stuck_naming_the_list_and_index():
    tiny = Sector(r_inner=20.0, r_outer=21.0, theta_min=0.0, theta_max=5.0)
    with pytest.raises(SamplingStuck) as exc:
        sample_lists(tiny, d_min=5.0, counts={"train": 40}, seed=1)
    assert exc.value.list_name == "train"
    assert exc.value.accepted >= 0
