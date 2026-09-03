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

from placement_mat.labels import label_map_rows
from placement_mat.render import RenderReport, TileOpts, paper_cm_per_data_cm, render_pdf
from placement_mat.tiling import plan_tiles, single_page_plan

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

_CARD_SINGLE = """# Placement mat (single sheet) -- operation card ({label})

1. Take the PDF to a print shop; print ONE sheet at 100% / actual size.
   Measure the "10.0 cm" ruler on the print. Wrong? Reprint at 100%.
2. Move the arm fully out of the way (it sweeps the near strip of the mat).
3. Lay the sheet on the table. Align the bold near-edge line to the FRONT TIPS
   of BOTH C-arms at once ({gap} cm apart) -- touching both fixes forward
   position and rotation. The sheet's blank margin may sit behind that line,
   into the C-slot; the PRINTED LINE is the datum, not the paper edge.
4. Tape the far corners. The pan axis is off the sheet, behind the near edge.
5. One-time: with a ruler, mark every point (t1, o1, c1 ...) as a small cross +
   its short id. Cross-check a few against placement_label_map_{label}.csv.
   Photograph the marked sheet into docs/assets/.
6. Each episode: short id -> object on that cross -> REMOVE THE SHEET ->
   then start recording. Never leave it in frame. Reuse the same marked sheet.
"""


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="make_placement_mat.py",
        description="A4-tiled Cartesian+polar placement mat with graduations (S3, D023).",
    )
    p.add_argument("--r-max", type=float, default=40.0, help="cm; largest radius drawn")
    p.add_argument("--x-min", type=float, default=-3.0, help="cm; left edge of the mat")
    p.add_argument("--theta-min", type=float, default=-90.0, help="deg; sector drawn (default 180deg)")
    p.add_argument("--theta-max", type=float, default=90.0, help="deg; sector drawn (default 180deg)")
    p.add_argument("--paper", choices=["a4", "single"], default="a4",
                   help="a4 = tiled pages; single = one large-format sheet for a print shop")
    p.add_argument("--datum-offset", type=float, default=None,
                   help="cm; pan-axis centre -> C-arm front-tips line. Required for --paper single. "
                        "S2 points are shifted x_mat = x_cm - datum_offset onto the mat frame.")
    p.add_argument("--chassis-gap", type=float, default=5.9,
                   help="cm; C-opening inner width -> tick spacing on the registration line (single)")
    p.add_argument("--chassis-outline", action=argparse.BooleanOptionalAction, default=True,
                   help="single: draw the C-base footprint behind the reg line + a notch-cut guide")
    p.add_argument("--chassis-arm", type=float, default=4.5, help="cm; inner-edge line -> front tips")
    p.add_argument("--chassis-back", type=float, default=7.5, help="cm; back-segment depth")
    p.add_argument("--chassis-outer-w", type=float, default=12.0, help="cm; full plate width")
    p.add_argument("--margin-mm", type=float, default=10.0, help="unprintable border per page")
    p.add_argument("--overlap-mm", type=float, default=15.0, help="overlap band between pages")
    p.add_argument("--grid-mm", type=float, default=5.0, help="fine Cartesian grid pitch")
    p.add_argument("--ring-step", type=float, default=5.0, help="cm between radius rings")
    p.add_argument("--ray-step", type=float, default=15.0, help="deg between azimuth rays")
    p.add_argument("--line-scale", type=float, default=1.0, help="multiply every grid/overlay line width")
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

    single = args.paper == "single"

    if single and args.datum_offset is None:
        print("REFUSED: --paper single needs --datum-offset (pan-axis centre -> C-arm tips line, cm).")
        return 2

    if args.placements and not single and frame != "mat":
        print(
            "REFUSED: --placements needs a mat-frame summary (azimuth_frame == 'mat'). "
            "Record an S1 'reference' sample first; a blank mat can still be printed without it."
        )
        return 2

    import math

    edge = [math.radians(a) for a in (args.theta_min, args.theta_max)]
    y_lo = min(0.0, *(args.r_max * math.sin(e) for e in edge))
    y_hi = max(0.0, *(args.r_max * math.sin(e) for e in edge))
    if args.theta_min < 90 < args.theta_max or args.theta_min < -90 < args.theta_max:
        y_hi, y_lo = args.r_max, -args.r_max  # sector crosses +/-90deg -> full height

    dx = args.datum_offset or 0.0
    points = _load_points(args.placements)
    x_lo = args.x_min
    x_hi = args.r_max - dx if single else args.r_max  # single: mat frame, x_mat = x_pan - dx

    if single:
        # the sheet must contain every seeded point after the datum shift, incl.
        # inner-radius points at steep angles (x_mat can go slightly negative) and
        # points at r == r_max (x_mat == r_max - dx). Grow the area to fit them.
        if points:
            pad = 1.0
            xs = [x - dx for _, x, _ in points]
            ys = [y for _, _, y in points]
            x_lo = min(x_lo, min(xs) - pad)
            x_hi = max(x_hi, max(xs) + pad)
            y_lo = min(y_lo, min(ys) - pad)
            y_hi = max(y_hi, max(ys) + pad)
        if args.chassis_outline:
            # show the arms, the inner-edge line and the pan-axis mark on the sheet
            x_lo = min(x_lo, -dx - 1.5, -args.chassis_arm - 1.5)
            y_lo = min(y_lo, -args.chassis_outer_w / 2.0 - 1.0)
            y_hi = max(y_hi, args.chassis_outer_w / 2.0 + 1.0)
        plan = single_page_plan(x_lo, x_hi, y_lo, y_hi)
    else:
        plan = plan_tiles(
            x_lo, x_hi, y_lo, y_hi,
            margin_cm=args.margin_mm / 10.0,
            overlap_cm=args.overlap_mm / 10.0,
        )
    opts = TileOpts(
        r_max=args.r_max,
        grid_mm=args.grid_mm,
        ring_step=args.ring_step,
        ray_step=args.ray_step,
        azimuth_offset_deg=offset,
        theta_min_deg=args.theta_min,
        theta_max_deg=args.theta_max,
        line_scale=args.line_scale,
        polar_origin_x=-dx if single else 0.0,
        near_edge=single,
        chassis_gap_cm=args.chassis_gap,
        azimuth_calibrated=(args.from_summary is not None and frame == "mat"),
        short_labels=single,
        chassis_outline=single and args.chassis_outline,
        chassis_arm_cm=args.chassis_arm,
        chassis_back_cm=args.chassis_back,
        chassis_outer_w_cm=args.chassis_outer_w,
    )

    print(
        f"mat area: x [{x_lo:g}, {x_hi:g}]  y [{y_lo:g}, {y_hi:g}]  cm  "
        f"(sector {args.theta_min:g}..{args.theta_max:g} deg)"
    )
    if single:
        print(f"single page: {x_hi - x_lo:g} x {y_hi - y_lo:g} cm  "
              f"(datum offset {dx:g} cm, pan axis at x_mat={-dx:g})")
        if args.chassis_outline:
            print(f"chassis outline: drawn (arm {args.chassis_arm:g} / back {args.chassis_back:g} "
                  f"/ outer-w {args.chassis_outer_w:g} cm) + notch-cut guide")
    else:
        print(f"pages: {len(plan.tiles)}  ({plan.n_rows} rows x {plan.n_cols} cols of A4)")
        for t in plan.tiles:
            print(f"  row {t.row + 1} col {t.col + 1}: x [{t.x0:.1f}, {t.x1:.1f}]  y [{t.y0:.1f}, {t.y1:.1f}]")

    if args.dry_run:
        print("--dry-run: not writing.")
        return 0

    draw_points = [(pid, x - dx, y) for pid, x, y in points] if single else points
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report: RenderReport = render_pdf(
        args.out, plan, opts, points=draw_points or None, single_page=single
    )

    if single:
        card = args.out.parent / f"placement_card_{args.label}_single.md"
        card.write_text(_CARD_SINGLE.format(label=args.label, gap=args.chassis_gap), encoding="utf-8")
        if points:
            mapf = args.out.parent / f"placement_label_map_{args.label}.csv"
            rows = label_map_rows(points, datum_offset=dx)
            with mapf.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f"wrote {mapf}  ({len(rows)} rows)")
    else:
        card = args.out.parent / f"placement_card_{args.label}.md"
        card.write_text(
            _CARD.format(label=args.label, rows=plan.n_rows, cols=plan.n_cols), encoding="utf-8"
        )

    print(f"wrote {args.out}  ({report.pages} page{'s' if report.pages != 1 else ''})")
    print(f"wrote {card}")
    if single:
        print("self-check: axes fill the page -> 1 data-cm == 1 paper-cm at 100% print (verify the ruler)")
    else:
        scale_ok = all(abs(paper_cm_per_data_cm(t) - 1.0) < 1e-6 for t in plan.tiles)
        print(f"self-check: 1 data-cm == 1 paper-cm on every page: {scale_ok}")
    if points:
        print(f"placement points drawn: {report.label_count} label(s), {len(report.distinct_ids)} distinct id(s)")
    return 0
