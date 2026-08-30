"""S2 --plot: a diagnostic scatter, not the S3 printable mat. Spec 5."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from placement_sampler.geometry import Sector
from placement_sampler.plot import build_figure, save_scatter
from placement_sampler.sampler import sample_lists

SECTOR = Sector(r_inner=20.0, r_outer=30.0, theta_min=-90.0, theta_max=90.0)
COUNTS = {"train": 50, "eval-open": 10, "eval-close": 30}


def _lists():
    return sample_lists(SECTOR, d_min=2.0, counts=COUNTS, seed=1)


def test_save_scatter_writes_a_nonempty_png(tmp_path):
    out = tmp_path / "camp_scatter.png"
    returned = save_scatter(_lists(), SECTOR, out, seed=1, d_min=2.0)
    assert returned == out
    assert out.exists() and out.stat().st_size > 0


def test_all_three_series_are_labelled():
    fig = build_figure(_lists(), SECTOR, seed=1, d_min=2.0)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("train" in x for x in labels)
    assert any("eval-open" in x for x in labels)
    assert any("eval-close" in x for x in labels)


def test_empty_lists_do_not_crash():
    fig = build_figure({"train": [], "eval-open": [], "eval-close": []}, SECTOR, seed=1, d_min=2.0)
    assert fig.axes
