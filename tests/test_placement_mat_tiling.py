"""S3 A4 tiling: cover the mat area with overlapping A4 pages. Spec 4, 5."""

from __future__ import annotations

from placement_mat.tiling import A4_H_CM, A4_W_CM, plan_tiles, seam_points


def test_small_area_is_a_single_page():
    plan = plan_tiles(-8.0, 8.0, -8.0, 8.0, margin_cm=1.0, overlap_cm=1.5)
    assert plan.n_rows == 1 and plan.n_cols == 1
    assert len(plan.tiles) == 1


def test_big_area_needs_a_grid_of_pages():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0, margin_cm=1.0, overlap_cm=1.5)
    assert plan.n_cols >= 3 and plan.n_rows >= 3
    assert len(plan.tiles) == plan.n_rows * plan.n_cols


def test_adjacent_columns_overlap_by_exactly_the_overlap_width():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0, margin_cm=1.0, overlap_cm=1.5)
    row0 = sorted((t for t in plan.tiles if t.row == 0), key=lambda t: t.col)
    a, b = row0[0], row0[1]
    assert abs((a.x1 - b.x0) - 1.5) < 1e-9


def test_tiles_cover_the_whole_requested_area():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0, margin_cm=1.0, overlap_cm=1.5)
    assert min(t.x0 for t in plan.tiles) <= -8.0 + 1e-9
    assert max(t.x1 for t in plan.tiles) >= 40.0 - 1e-9
    assert min(t.y0 for t in plan.tiles) <= -40.0 + 1e-9
    assert max(t.y1 for t in plan.tiles) >= 40.0 - 1e-9


def test_tile_size_is_the_printable_area_of_one_a4():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0, margin_cm=1.0, overlap_cm=1.5)
    t = plan.tiles[0]
    assert abs((t.x1 - t.x0) - (A4_W_CM - 2.0)) < 1e-9
    assert abs((t.y1 - t.y0) - (A4_H_CM - 2.0)) < 1e-9


def test_seam_points_fall_inside_two_tiles_each():
    plan = plan_tiles(-8.0, 40.0, -40.0, 40.0, margin_cm=1.0, overlap_cm=1.5)
    seams = seam_points(plan)
    assert seams
    for sx, sy in seams:
        covering = [
            t for t in plan.tiles if t.x0 - 1e-9 <= sx <= t.x1 + 1e-9 and t.y0 - 1e-9 <= sy <= t.y1 + 1e-9
        ]
        assert len(covering) >= 2
