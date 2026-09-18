#!/usr/bin/env python
"""T1 (S4 §5-5) for the Innomaker U20CAM-720P wrist camera (opencv_uvc, D022 2026-09-13).

Unlike the D455, a generic UVC webcam has no factory intrinsics an API can read -- this does a
standard checkerboard calibration (cv2.calibrateCamera), which solves intrinsics AND distortion
in one pass from several photos of a known board at different angles.

Print the target first: `uv run python sim/calib_gen_targets.py checkerboard`, then MEASURE the
printed square with a ruler and pass that measured value via --square-mm (not the requested one
-- printer scaling is not trustworthy, see that script's docstring).

Opens the camera at the SAME index/backend/resolution/fourcc as the given lerobot camera YAML's
`wrist` block, so the calibration matches what actually gets recorded. 🔴 index is NOT a stable
id on Windows (configs/teleoperate_omx.yaml's own warning) -- run `uv run lerobot-find-cameras
opencv` first if unsure which index is the wrist camera today.

🔴 The interactive capture loop (cv2.VideoCapture / imshow) has NOT been run against the real
camera in the environment this was written in -- no camera attached here. The
calibrateCamera/undistortPoints math below WAS self-tested against synthetic data (see the
conversation this was written in). Run this on the lab machine and report back anything that
breaks. `[AI推論，互動迴圈未實測]`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]

_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY, "v4l2": cv2.CAP_V4L2}


def load_camera_block(config_path: Path, camera: str) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cams = (raw.get("robot") or {}).get("cameras") or {}
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not in {config_path} (has: {sorted(cams)})")
    block = cams[camera]
    if block.get("type") != "opencv_uvc":
        raise SystemExit(f"camera '{camera}' is type '{block.get('type')}', not opencv_uvc -- use calib_intrinsics_realsense.py instead")
    return block


def object_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * (square_mm / 1000.0)
    return objp


def capture_loop(cap, cols: int, rows: int, square_mm: float, n_captures: int):
    objp = object_points(cols, rows, square_mm)
    objpoints, imgpoints = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    print(f"live preview: SPACE to capture when the board is highlighted green, ESC to stop early (need >= 8), q to abort")
    while len(objpoints) < n_captures:
        ok, frame = cap.read()
        if not ok:
            raise SystemExit("camera read failed -- check --index / --backend")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, (cols, rows), None)
        vis = frame.copy()
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(vis, (cols, rows), corners, found)
        cv2.putText(vis, f"captured {len(objpoints)}/{n_captures}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if found else (0, 0, 255), 2)
        cv2.imshow("calib_intrinsics_checkerboard", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(" ") and found:
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"  captured {len(objpoints)}/{n_captures}")
        elif key == 27:  # ESC
            if len(objpoints) < 8:
                print(f"only {len(objpoints)} captures, calibration needs >= 8 for a stable solve -- keep going")
                continue
            break
        elif key == ord("q"):
            raise SystemExit("aborted")
    cv2.destroyAllWindows()
    return objpoints, imgpoints


def distortion_displacement_px(K: np.ndarray, dist: np.ndarray, width: int, height: int, n_grid: int = 9) -> np.ndarray:
    """Per-sample-point |distorted_pixel - ideal_pinhole_pixel| using the SOLVED distortion model."""
    us, vs = np.meshgrid(np.linspace(0, width - 1, n_grid), np.linspace(0, height - 1, n_grid))
    pts = np.stack([us.ravel(), vs.ravel()], axis=1).astype(np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    return np.hypot(undistorted[:, 0] - pts[:, 0, 0], undistorted[:, 1] - pts[:, 0, 1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "teleoperate_omx.yaml")
    ap.add_argument("--camera", default="wrist")
    ap.add_argument("--index", type=int, default=None, help="override the config's index_or_path")
    ap.add_argument("--backend", default=None, choices=list(_BACKENDS), help="override the config's backend")
    ap.add_argument("--cols", type=int, default=9, help="inner corners, horizontal -- must match calib_gen_targets.py")
    ap.add_argument("--rows", type=int, default=6, help="inner corners, vertical")
    ap.add_argument("--square-mm", type=float, required=True, help="MEASURED printed square edge length, not the requested one")
    ap.add_argument("--captures", type=int, default=15)
    ap.add_argument("--max-accept-px", type=float, default=5.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    block = load_camera_block(args.config, args.camera)
    index = args.index if args.index is not None else block["index_or_path"]
    backend = _BACKENDS[args.backend if args.backend is not None else block.get("backend", "any").lower()]
    width, height, fourcc = block["width"], block["height"], block.get("fourcc")
    print(f"camera '{args.camera}' from {args.config}: index={index} backend={args.backend or block.get('backend')} {width}x{height} fourcc={fourcc}")

    cap = cv2.VideoCapture(index, backend)
    if fourcc:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {index} with backend {args.backend or block.get('backend')}")

    try:
        objpoints, imgpoints = capture_loop(cap, args.cols, args.rows, args.square_mm, args.captures)
    finally:
        cap.release()

    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (width, height), None, None)
    print(f"cv2.calibrateCamera RMS reprojection error: {rms:.3f} px")

    per_image_err = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
        per_image_err.append(float(np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1).mean()))
    per_image_err = np.array(per_image_err)
    print(f"per-image mean reprojection error: p50={np.percentile(per_image_err, 50):.3f} max={per_image_err.max():.3f} px")

    disp = distortion_displacement_px(K, dist, width, height)
    p50, p95, pmax = float(np.percentile(disp, 50)), float(np.percentile(disp, 95)), float(disp.max())
    verdict = "OK -- negligible" if pmax <= args.max_accept_px else "WARN -- undistort real T2/T3 images before comparing to the sim's ideal-pinhole render"
    print(f"distortion-induced pixel displacement across the frame: p50={p50:.2f} p95={p95:.2f} max={pmax:.2f} px -> {verdict}")

    result = {
        "camera": args.camera,
        "index": index,
        "width": width,
        "height": height,
        "fx": float(K[0, 0]),
        "fy": float(K[1, 1]),
        "cx": float(K[0, 2]),
        "cy": float(K[1, 2]),
        "distortion_coeffs": dist.ravel().tolist(),
        "calibration_rms_px": float(rms),
        "per_image_err_p50_px": float(np.percentile(per_image_err, 50)),
        "per_image_err_max_px": float(per_image_err.max()),
        "distortion_px_p50": p50,
        "distortion_px_p95": p95,
        "distortion_px_max": pmax,
        "n_captures": len(objpoints),
        "square_mm": args.square_mm,
        "measured_at": dt.date.today().isoformat(),
        "source_config": str(args.config),
    }
    out = args.out or (_REPO / "calibration" / f"{dt.date.today().isoformat()}_camera_intrinsics_{args.camera}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
