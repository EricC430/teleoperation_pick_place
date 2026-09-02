"""S3 CLI: blank mat by default, points + frame check with --placements. Spec 3, 5."""

from __future__ import annotations

import json

import matplotlib
import pytest

matplotlib.use("Agg")

from placement_mat.cli import main


def _csv(path, rows):
    path.write_text(
        "placement_id,r_cm,theta_deg,x_cm,y_cm\n"
        + "".join(f"{pid},0,0,{x},{y}\n" for pid, x, y in rows),
        encoding="utf-8",
    )
    return path


def test_help_exits_zero():
    with pytest.raises(SystemExit) as exc:
        main(["--help"])
    assert exc.value.code == 0


def test_dry_run_writes_nothing_and_reports_page_count(tmp_path, capsys):
    rc = main(["--r-max", "40", "--out", str(tmp_path / "m.pdf"), "--dry-run"])
    assert rc == 0
    assert list(tmp_path.iterdir()) == []
    assert "pages" in capsys.readouterr().out.lower()


def test_blank_mat_writes_pdf_and_operation_card(tmp_path):
    out = tmp_path / "placement_mat_blank.pdf"
    rc = main(["--r-max", "25", "--out", str(out), "--label", "blank"])
    assert rc == 0
    assert out.stat().st_size > 0
    assert (tmp_path / "placement_card_blank.md").exists()


def test_placements_with_base_frame_summary_is_refused(tmp_path):
    pts = _csv(tmp_path / "train.csv", [("train_001", 10.0, 5.0)])
    summ = tmp_path / "s.json"
    summ.write_text(json.dumps({"azimuth_frame": "base", "azimuth_offset_deg": None}), encoding="utf-8")
    rc = main(
        [
            "--r-max", "25", "--out", str(tmp_path / "m.pdf"),
            "--placements", str(pts), "--from-summary", str(summ),
        ]
    )
    assert rc == 2


def test_placements_render_every_id(tmp_path, capsys):
    a = _csv(tmp_path / "train.csv", [("train_001", 10.0, 5.0), ("train_002", -3.0, -8.0)])
    b = _csv(tmp_path / "ec.csv", [("eval-close_001", 12.0, -4.0)])
    rc = main(["--r-max", "25", "--out", str(tmp_path / "m.pdf"), "--placements", str(a), str(b)])
    assert rc == 0
    assert "3" in capsys.readouterr().out  # 3 ids reported drawn


def test_paper_single_needs_datum_offset(tmp_path):
    rc = main(["--paper", "single", "--r-max", "40", "--out", str(tmp_path / "m.pdf")])
    assert rc == 2


def test_paper_single_writes_one_page_pdf_and_label_map(tmp_path):
    a = _csv(tmp_path / "train.csv", [("train_001", 30.0, 5.0), ("eval-open_003", 25.0, -6.0)])
    out = tmp_path / "m.pdf"
    rc = main(
        [
            "--paper", "single", "--datum-offset", "4", "--r-max", "40",
            "--placements", str(a), "--out", str(out), "--label", "campA",
        ]
    )
    assert rc == 0
    assert out.stat().st_size > 0
    mapf = out.parent / "placement_label_map_campA.csv"
    assert mapf.exists()
    body = mapf.read_text(encoding="utf-8")
    assert "t1,train_001" in body and "o3,eval-open_003" in body
    assert "26.0" in body  # x_mat_cm = 30 - 4


def test_paper_single_draws_every_point_even_at_the_sector_edges(tmp_path, capsys):
    # inner radius at a steep angle -> x_mat goes slightly negative (past --x-min);
    # radius == r_max -> x_mat lands at r_max - datum_offset. Neither may be clipped.
    a = _csv(tmp_path / "p.csv", [("train_001", 1.2, -22.0), ("eval-close_002", 39.0, 3.0)])
    rc = main(
        [
            "--paper", "single", "--datum-offset", "4", "--x-min", "-1", "--r-max", "39",
            "--placements", str(a), "--out", str(tmp_path / "m.pdf"), "--label", "e",
        ]
    )
    assert rc == 0
    assert "2 label(s), 2 distinct id(s)" in capsys.readouterr().out


def test_paper_single_allows_placements_without_a_summary(tmp_path):
    a = _csv(tmp_path / "train.csv", [("train_001", 30.0, 5.0)])
    rc = main(
        [
            "--paper", "single", "--datum-offset", "4", "--r-max", "40",
            "--placements", str(a), "--out", str(tmp_path / "m.pdf"),
        ]
    )
    assert rc == 0
