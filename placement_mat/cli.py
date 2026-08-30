"""`uv run scripts/make_placement_mat.py` -- build the A4-tiled polar mat PDF.

Spec: docs/specs/S3_placement_mat.md 3, 4, 5.

Exit codes:
  0  PDF written (or --dry-run)
  2  bad arguments, or --placements given with an un-referenced (base-frame) summary
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from placement_mat.render import RenderReport, TileOpts, paper_cm_per_data_cm, render_pdf
from placement_mat.tiling import plan_tiles

_CARD = """# Placement mat -- operation card ({label})

1. Measure the "10.0 cm" ruler on EVERY page. Not 10.0 cm (+/-0.5 mm)? Set the
   printer to 100% / "actual size" and reprint.
2. Assemble pages by the "+" registration crosses (cross on cross, not paper
   edges). Tape on the back. Page layout: {rows} rows x {cols} cols.
3. Put the mat's (0,0) on the arm base rotation axis; the 0deg ray on the S1
   reference direction / physical alignment mark. Tape the corners.
4. One-time: with a ruler, mark every point from the placement CSV(s) as a small
   cross + id. Photograph the marked mat into docs/assets/.
5. Each episode: look up placement_id -> put the object on that cross ->
   REMOVE THE MAT -> then start recording.
6. Pack-up: never leave the mat in frame. Reuse the same marked mat next time
   (do not reprint, do not re-mark -- both accumulate error).
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_placement_mat.py",
        description="A4-tiled Cartesian+polar placement mat with graduations (S3, D023).",
    )
    p.add_argument("--r-max", type=float, default=40.0, help="cm; largest radius drawn")
    p.add_argument("--x-min", type=float, default=-8.0, help="cm; left edge of the mat")
    p.add_argument("--paper", choices=["a4"], default="a4")
    p.add_argument("--margin-mm", type=float, default=10.0, help="unprintable border per page")
    p.add_argument("--overlap-mm", type=float, default=15.0, help="overlap band between pages")
    p.add_argument("--grid-mm", type=float, default=5.0, help="fine Cartesian grid pitch")
    p.add_argument("--ring-step", type=float, default=5.0, help="cm between radius rings")
    p.add_argument("--ray-step", type=float, default=15.0, help="deg between azimuth rays")
    p.add_argument("--out", type=Path, default=Path("docs/assets/placement_mat_blank.pdf"))
    p.add_argument("--placements", type=Path, nargs="*", default=[], help="S2 CSV(s) to plot")
    p.add_argument("--from-summary", type=Path, help="S1 reach_summary_<date>.json (azimuth_offset)")
    p.add_argument("--label", default="blank")
    p.add_argument("--dry-run", action="store_true")
    return p


def _load_points(paths: list[Path]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for path in paths:
        with path.open(encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                out.append((row["placement_id"], float(row["x_cm"]), float(row["y_cm"])))
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    offset = 0.0
    frame = "mat"
    if args.from_summary is not None:
        data = json.loads(args.from_summary.read_text(encoding="utf-8"))
        offset = data.get("azimuth_offset_deg") or 0.0
        frame = data.get("azimuth_frame", "base")

    if args.placements and frame != "mat":
        print(
            "REFUSED: --placements needs a mat-frame summary (azimuth_frame == 'mat'). "
            "Record an S1 'reference' sample first; a blank mat can still be printed without it."
        )
        return 2

    plan = plan_tiles(
        args.x_min,
        args.r_max,
        -args.r_max,
        args.r_max,
        margin_cm=args.margin_mm / 10.0,
        overlap_cm=args.overlap_mm / 10.0,
    )
    opts = TileOpts(
        r_max=args.r_max,
        grid_mm=args.grid_mm,
        ring_step=args.ring_step,
        ray_step=args.ray_step,
        azimuth_offset_deg=offset,
    )

    print(f"mat area: x [{args.x_min:g}, {args.r_max:g}]  y [{-args.r_max:g}, {args.r_max:g}]  cm")
    print(f"pages: {len(plan.tiles)}  ({plan.n_rows} rows x {plan.n_cols} cols of A4)")
    for t in plan.tiles:
        print(f"  row {t.row + 1} col {t.col + 1}: x [{t.x0:.1f}, {t.x1:.1f}]  y [{t.y0:.1f}, {t.y1:.1f}]")

    if args.dry_run:
        print("--dry-run: not writing.")
        return 0

    points = _load_points(args.placements)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: RenderReport = render_pdf(args.out, plan, opts, points=points or None)

    card = args.out.parent / f"placement_card_{args.label}.md"
    card.write_text(_CARD.format(label=args.label, rows=plan.n_rows, cols=plan.n_cols), encoding="utf-8")

    scale_ok = all(abs(paper_cm_per_data_cm(t) - 1.0) < 1e-6 for t in plan.tiles)
    print(f"wrote {args.out}  ({report.pages} pages)")
    print(f"wrote {card}")
    print(f"self-check: 1 data-cm == 1 paper-cm on every page: {scale_ok}")
    if points:
        print(f"placement points drawn: {report.label_count} label(s), {len(report.distinct_ids)} distinct id(s)")
    return 0
