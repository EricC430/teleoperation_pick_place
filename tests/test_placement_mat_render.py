"""S3 PDF render: physical scale, page count, points. Spec 4, 5."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from placement_mat.render import (
    TileOpts,
    paper_cm_per_data_cm,
    points_on_tile,
    render_pdf,
)
from placement_mat.tiling import plan_tiles

OPTS = TileOpts(r_max=40.0, grid_mm=5.0, ring_step=5.0, ray_step=15.0)


def test_one_data_cm_is_one_paper_cm():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    for t in plan.tiles:
        assert abs(paper_cm_per_data_cm(t) - 1.0) < 1e-6


def test_pdf_has_one_page_per_tile(tmp_path):
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    out = tmp_path / "mat.pdf"
    report = render_pdf(out, plan, OPTS)
    assert out.exists() and out.stat().st_size > 0
    assert report.pages == len(plan.tiles)


def test_small_mat_is_a_single_page_pdf(tmp_path):
    plan = plan_tiles(-8.0, 8.0, -8.0, 8.0)
    report = render_pdf(tmp_path / "m.pdf", plan, OPTS)
    assert report.pages == 1


def test_points_on_tile_selects_only_those_inside():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    t = plan.tiles[0]
    pts = [("a", t.x0 + 1, t.y0 + 1), ("b", t.x1 + 100, t.y0 + 1)]
    got = points_on_tile(pts, t)
    assert [p[0] for p in got] == ["a"]


def test_every_placement_id_is_drawn_at_least_once(tmp_path):
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    pts = [("train_001", 10.0, 5.0), ("eval-close_007", -3.0, -20.0), ("eval-open_002", 33.0, 30.0)]
    report = render_pdf(tmp_path / "m.pdf", plan, OPTS, points=pts)
    assert report.label_count >= len(pts)
    assert report.distinct_ids == {p[0] for p in pts}
