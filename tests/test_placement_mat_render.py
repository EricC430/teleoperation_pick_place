"""S3 PDF render: physical scale, page count, points. Spec 4, 5."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from placement_mat.render import (
    TileOpts,
    build_tile_figure,
    paper_cm_per_data_cm,
    points_on_tile,
    render_pdf,
)
from placement_mat.tiling import plan_tiles, single_page_plan

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


def test_line_scale_is_honoured_and_does_not_crash(tmp_path):
    plan = plan_tiles(-8.0, 8.0, -8.0, 8.0)
    heavy = TileOpts(r_max=40.0, line_scale=2.5)
    report = render_pdf(tmp_path / "heavy.pdf", plan, heavy)
    assert report.pages == 1
    assert (tmp_path / "heavy.pdf").stat().st_size > 0


def test_points_on_tile_selects_only_those_inside():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    t = plan.tiles[0]
    pts = [("a", t.x0 + 1, t.y0 + 1), ("b", t.x1 + 100, t.y0 + 1)]
    got = points_on_tile(pts, t)
    assert [p[0] for p in got] == ["a"]


def test_single_page_figure_is_sized_to_the_area_in_cm():
    plan = single_page_plan(-2.0, 40.0, -30.0, 20.0)  # 42 x 50 cm
    fig, _ = build_tile_figure(plan.tiles[0], plan, TileOpts(r_max=40.0), single_page=True)
    w_in, h_in = fig.get_size_inches()
    assert abs(w_in * 2.54 - 42.0) < 1e-6
    assert abs(h_in * 2.54 - 50.0) < 1e-6


def test_single_page_annotates_points_with_short_ids():
    plan = single_page_plan(-2.0, 40.0, -30.0, 20.0)
    opts = TileOpts(r_max=40.0, short_labels=True)
    fig, drawn = build_tile_figure(
        plan.tiles[0], plan, opts, points=[("train_007", 20.0, 3.0)], single_page=True
    )
    assert drawn == 1
    texts = {t.get_text() for a in fig.axes for t in a.texts}
    assert "t7" in texts and "train_007" not in texts


def test_chassis_outline_marks_the_pan_axis_at_minus_datum_offset():
    plan = single_page_plan(-9.0, 32.0, -30.0, 20.0)
    opts = TileOpts(
        r_max=39.0, polar_origin_x=-7.13, near_edge=True, chassis_outline=True,
        chassis_gap_cm=5.9, chassis_arm_cm=4.5, chassis_back_cm=7.5, chassis_outer_w_cm=12.0,
    )
    fig, _ = build_tile_figure(plan.tiles[0], plan, opts, single_page=True)
    ax = fig.axes[0]
    pan_marks = [
        ln for ln in ax.get_lines()
        if ln.get_marker() == "+" and list(ln.get_xdata()) == [-7.13] and list(ln.get_ydata()) == [0.0]
    ]
    assert pan_marks, "expected a '+' pan-axis marker at (-datum_offset, 0)"


def test_chassis_outline_is_off_by_default_and_needs_single_page():
    plan = single_page_plan(-9.0, 32.0, -30.0, 20.0)
    base = TileOpts(r_max=39.0, polar_origin_x=-7.13)
    fig, _ = build_tile_figure(plan.tiles[0], plan, base, single_page=True)
    ax = fig.axes[0]
    assert not [ln for ln in ax.get_lines() if ln.get_marker() == "+" and list(ln.get_xdata()) == [-7.13]]


def test_single_page_render_pdf_is_one_page(tmp_path):
    plan = single_page_plan(-2.0, 40.0, -30.0, 20.0)
    opts = TileOpts(
        r_max=40.0, polar_origin_x=-4.0, near_edge=True, short_labels=True, azimuth_calibrated=False
    )
    pts = [("train_001", 26.0, -5.0), ("eval-close_007", 20.0, -18.0)]
    report = render_pdf(tmp_path / "s.pdf", plan, opts, points=pts, single_page=True)
    assert report.pages == 1 and report.label_count >= 2
    assert (tmp_path / "s.pdf").stat().st_size > 0


def test_every_placement_id_is_drawn_at_least_once(tmp_path):
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0)
    pts = [("train_001", 10.0, 5.0), ("eval-close_007", -3.0, -20.0), ("eval-open_002", 33.0, 30.0)]
    report = render_pdf(tmp_path / "m.pdf", plan, OPTS, points=pts)
    assert report.label_count >= len(pts)
    assert report.distinct_ids == {p[0] for p in pts}
