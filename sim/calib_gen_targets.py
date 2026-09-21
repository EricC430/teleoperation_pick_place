#!/usr/bin/env python
"""Generate printable calibration targets for S4 §5-5 T1/T2 (docs/specs/S5_sim_replay_augmentation.md §2 gap 4).

Two targets, one script:

  aruco         Individual markers to tape onto the placement mat at measured points.
                Consumed by calib_extrinsics_aruco.py to solve camera extrinsics (T2).
  checkerboard  A chessboard target. Consumed by calib_intrinsics_checkerboard.py to calibrate
                the Innomaker U20CAM-720P wrist camera (T1) -- unlike the D455, a UVC webcam has
                no factory intrinsics an API can read, so this stands in for that.

🔴 Printer "100% / actual size" is not trustworthy -- same lesson already paid for on the
placement mat (docs/meeting/2026-09-03.md §4-3: "交代店家 100% 原尺寸...取件後用尺量一格 5 cm 驗
證"). After printing, MEASURE the printed target with a ruler or calipers and feed the MEASURED
value (not the one you asked for) into calib_extrinsics_aruco.py / calib_intrinsics_checkerboard.py
via --marker-side-m / --square-mm. The DPI tag on the saved PNG is a best-effort hint to the
print driver, not a guarantee.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parents[1]
MM_PER_INCH = 25.4


def _save_png(arr: np.ndarray, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, dpi=(dpi, dpi))


def gen_aruco(args: argparse.Namespace) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, args.dict))
    side_px = round(args.side_mm / MM_PER_INCH * args.dpi)
    border_px = side_px // 4  # quiet zone; ArUco detection wants >= ~1 module width of white margin
    canvas = side_px + 2 * border_px

    out_dir = Path(args.out_dir)
    for marker_id in range(args.count):
        marker = cv2.aruco.generateImageMarker(dictionary, marker_id, side_px)
        page = np.full((canvas, canvas), 255, dtype=np.uint8)
        page[border_px : border_px + side_px, border_px : border_px + side_px] = marker
        _save_png(page, out_dir / f"aruco_{args.dict}_id{marker_id:02d}.png", args.dpi)

    print(f"wrote {args.count} markers to {out_dir}/ (dict={args.dict}, requested side {args.side_mm} mm @ {args.dpi} dpi)")
    print(
        f"print at 100% scale, then measure the BLACK SQUARE (not the white border) of one marker "
        f"with a ruler -- pass the MEASURED value to calib_extrinsics_aruco.py --marker-side-m"
    )
    if args.screen_ppi:
        px = args.side_mm / MM_PER_INCH * args.screen_ppi
        print()
        print(f"--- displaying on a {args.screen_ppi:g} ppi screen instead of paper ---")
        print(f"  a {args.side_mm:g} mm black square needs to be {px:.1f} screen pixels")
        print(f"  one screen pixel = {MM_PER_INCH / args.screen_ppi:.4f} mm")
        print("  🔴 a viewer that 'fits to screen' RESAMPLES the image and silently changes that size.")
        print("     Display at 1:1, then MEASURE the black square on the glass with a ruler anyway and")
        print("     pass the measured value -- the ruler step does not go away, it just moves.")
        print("  🔴 --points-csv's height_above_table_m must carry the PHONE'S THICKNESS (~8-9 mm),")
        print("     not 0: the screen surface, not the table, is where the marker plane sits.")
        print("  🔴 glare: a glossy screen mirrors the ceiling lights back at a camera looking down at")
        print("     an angle, which is exactly T2's geometry. Paper is matte and does not. Kill the")
        print("     overhead light or light from the side, screen to full brightness, auto-brightness")
        print("     and auto-lock OFF, and check a captured frame for blown highlights before trusting")
        print("     a session's worth of shots.")


def gen_checkerboard(args: argparse.Namespace) -> None:
    # cols/rows are INNER corners (cv2.findChessboardCorners convention) -> cols+1 / rows+1 squares.
    square_px = round(args.square_mm / MM_PER_INCH * args.dpi)
    n_cols_sq, n_rows_sq = args.cols + 1, args.rows + 1
    margin_px = square_px  # one extra square of white border on every side, for corner detection margin

    h = n_rows_sq * square_px + 2 * margin_px
    w = n_cols_sq * square_px + 2 * margin_px
    board = np.full((h, w), 255, dtype=np.uint8)
    for r in range(n_rows_sq):
        for c in range(n_cols_sq):
            if (r + c) % 2 == 0:
                y0 = margin_px + r * square_px
                x0 = margin_px + c * square_px
                board[y0 : y0 + square_px, x0 : x0 + square_px] = 0

    out_path = Path(args.out)
    _save_png(board, out_path, args.dpi)
    print(f"wrote {n_cols_sq}x{n_rows_sq}-square checkerboard ({args.cols}x{args.rows} inner corners) to {out_path}")
    print(f"requested square size {args.square_mm} mm @ {args.dpi} dpi")
    print(
        "print at 100% scale, then measure one square's edge with a ruler -- pass the MEASURED value "
        "to calib_intrinsics_checkerboard.py --square-mm (not the requested one)"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="target", required=True)

    p_aruco = sub.add_parser("aruco", help="generate individual ArUco markers for T2 extrinsics")
    p_aruco.add_argument("--count", type=int, default=8, help="number of markers (ids 0..count-1)")
    p_aruco.add_argument("--dict", default="DICT_4X4_50", help="cv2.aruco dictionary name")
    p_aruco.add_argument("--side-mm", type=float, default=50.0, help="requested black-square side length")
    p_aruco.add_argument("--dpi", type=int, default=300)
    p_aruco.add_argument("--out-dir", default=str(_REPO / "calibration" / "aruco_markers"))
    p_aruco.add_argument(
        "--screen-ppi",
        type=float,
        default=None,
        help="display on a phone/tablet screen instead of paper: print the exact pixel size a "
        "marker needs at this screen's ppi, plus the gotchas paper does not have. "
        "calib_extrinsics_aruco.py's mode (b) is built for this.",
    )
    p_aruco.set_defaults(func=gen_aruco)

    p_cb = sub.add_parser("checkerboard", help="generate a checkerboard for T1 UVC intrinsics")
    p_cb.add_argument("--cols", type=int, default=9, help="inner corners, horizontal")
    p_cb.add_argument("--rows", type=int, default=6, help="inner corners, vertical")
    p_cb.add_argument("--square-mm", type=float, default=25.0, help="requested square edge length")
    p_cb.add_argument("--dpi", type=int, default=300)
    p_cb.add_argument("--out", default=str(_REPO / "calibration" / "checkerboard_9x6.png"))
    p_cb.set_defaults(func=gen_checkerboard)

    args = ap.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
