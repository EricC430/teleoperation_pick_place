"""S2 core: dart-throwing three frozen lists with a global d_min. Spec 4."""

from __future__ import annotations

import math

import pytest

from placement_sampler.geometry import Sector
from placement_sampler.sampler import SamplingStuck, sample_lists

SECTOR = Sector(r_inner=20.0, r_outer=30.0, theta_min=-35.0, theta_max=88.0)
COUNTS = {"train": 50, "eval-open": 10, "eval-close": 30}


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
        assert 20.0 <= p.r_cm <= 30.0
        assert -35.0 <= p.theta_deg <= 88.0


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
