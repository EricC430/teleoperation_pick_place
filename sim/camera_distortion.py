#!/usr/bin/env python
"""Lens distortion for the sim <-> real camera gap (S4 §5-5 T2/T3, S5 gap 4).

Isaac Sim renders IDEAL pinhole images. The real cameras are not ideal:

  D455 (RealSense)  coeffs = (k1, k2, p1, p2, k3), reported as `inverse_brown_conrady`. Behaves as
                    librealsense's FORWARD Brown-Conrady (ideal -> distorted), with the SDK inverting
                    it by fixed-point iteration when it deprojects. "rs_bc" below. It is close to,
                    but not the same formula as, OpenCV's: librealsense applies the tangential terms
                    to the radially-scaled point and uses the unscaled r^2 (~0.05 px apart here).
  Innomaker (UVC)   cv2.calibrateCamera output: ordinary forward OpenCV model. "opencv" below.

`[已查證 2026-10-07]` against the SDK-derived numbers from the lab run of calib_intrinsics_realsense.py
(distortion displacement over a 9x9 grid, p50/p95/max = 3.68/6.21/6.61 px): the iteration below
reproduces all three EXACTLY. The other candidates miss: reading the coefficients as a direct
"undistort polynomial" (the opposite direction) gives 3.68/6.15/6.59, tangential terms swapped gives
3.27/6.70/6.80, radial-only 3.89/5.18/5.22, plain OpenCV-forward 3.65/6.23/6.63. Magnitude statistics
alone cannot separate the two directions (both move a corner point ~6 px, one inward and one
outward); exactness of all three numbers is what did. The decisive per-point check is
calib_intrinsics_realsense.py's SDK cross-check (`formula_vs_sdk_max_px`): a wrong direction shows up
as ~10 px there, a right one as ~0.0x px. calib_extrinsics_aruco.py refuses to run on a RealSense
JSON that lacks that check or fails it.

Two uses, one conversion underneath:
  undistort_pixels(pts, model)   real detections -> ideal pixels (T2 solvePnP, T3 residual)
  apply_distortion(img, maps)    ideal sim render -> what the real camera would show (S5 frames)
                                 -- render with set_intrinsic_matrices(K from the SAME json), at the
                                 json's width x height, then remap.

    uv run python sim/camera_distortion.py --intrinsics-json calibration/<d>_camera_intrinsics_front-left.json \\
        --image render.png --out render_distorted.png
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraModel:
    K: np.ndarray  # 3x3
    width: int
    height: int
    kind: str  # "none" | "opencv" | "rs_bc"
    coeffs: tuple  # (k1, k2, p1, p2, k3)


def kind_from_name(name: str | None) -> str:
    n = (name or "opencv").lower()
    if "modified_brown_conrady" in n:
        raise NotImplementedError(f"distortion model '{name}' is not implemented")
    if "brown_conrady" in n and "opencv" not in n:  # RealSense brown_conrady / inverse_brown_conrady
        return "rs_bc"
    if n.endswith("none"):
        return "none"
    if "opencv" in n:
        return "opencv"
    raise NotImplementedError(f"unknown distortion model '{name}'")


def make_model(fx, fy, cx, cy, width, height, model_name, coeffs) -> CameraModel:
    c = list(coeffs or [])
    c = tuple(float(v) for v in (c + [0.0] * 5)[:5])
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)
    return CameraModel(K, int(width), int(height), kind_from_name(model_name), c)


def load_model(path: Path) -> CameraModel:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return make_model(d["fx"], d["fy"], d["cx"], d["cy"], d["width"], d["height"], d.get("distortion_model"), d.get("distortion_coeffs"))


def _opencv_forward(x, y, c):
    k1, k2, p1, p2, k3 = c
    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    return x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x), y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y)


def _rs_forward(x, y, c):
    k1, k2, p1, p2, k3 = c
    r2 = x * x + y * y
    f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xq, yq = x * f, y * f
    return xq + 2.0 * p1 * xq * yq + p2 * (r2 + 2.0 * xq * xq), yq + 2.0 * p2 * xq * yq + p1 * (r2 + 2.0 * yq * yq)


def _rs_inverse(xo, yo, c, iters: int):
    """librealsense's deprojection loop (it runs 10 iterations; more just converges tighter)."""
    k1, k2, p1, p2, k3 = c
    x, y = xo.copy(), yo.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
        xq, yq = x / icdist, y / icdist
        dx = 2.0 * p1 * xq * yq + p2 * (r2 + 2.0 * xq * xq)
        dy = 2.0 * p2 * xq * yq + p1 * (r2 + 2.0 * yq * yq)
        x, y = (xo - dx) * icdist, (yo - dy) * icdist
    return x, y


def _norm(pts: np.ndarray, K: np.ndarray):
    return (pts[:, 0] - K[0, 2]) / K[0, 0], (pts[:, 1] - K[1, 2]) / K[1, 1]


def _pix(x, y, K: np.ndarray) -> np.ndarray:
    return np.stack([x * K[0, 0] + K[0, 2], y * K[1, 1] + K[1, 2]], axis=1)


def undistort_pixels(pts, m: CameraModel, iters: int = 30) -> np.ndarray:
    """(N, 2) pixels as the REAL camera saw them -> (N, 2) pixels of the same point in an ideal
    pinhole camera with the same K."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if m.kind == "none" or not any(m.coeffs):
        return pts.copy()
    if m.kind == "opencv":
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 60, 1e-12)
        out = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), m.K, np.array(m.coeffs), None, m.K, crit)
        return out.reshape(-1, 2)
    xo, yo = _norm(pts, m.K)
    x, y = _rs_inverse(xo, yo, m.coeffs, iters)
    return _pix(x, y, m.K)


def distort_pixels(ideal_pts, m: CameraModel) -> np.ndarray:
    """Inverse of undistort_pixels: ideal-pinhole pixels -> pixels the real camera would record."""
    ideal = np.asarray(ideal_pts, dtype=np.float64).reshape(-1, 2)
    if m.kind == "none" or not any(m.coeffs):
        return ideal.copy()
    x, y = _norm(ideal, m.K)
    dx, dy = (_opencv_forward if m.kind == "opencv" else _rs_forward)(x, y, m.coeffs)
    return _pix(dx, dy, m.K)


def padded_render_spec(m: CameraModel, margin: int):
    """(width, height, K) to render the IDEAL image at when `margin` extra pixels per side are needed:
    same focal lengths, principal point shifted by `margin`. Feed K to set_intrinsic_matrices."""
    K = m.K.copy()
    K[0, 2] += margin
    K[1, 2] += margin
    return m.width + 2 * margin, m.height + 2 * margin, K


def build_distort_maps(m: CameraModel, margin: int = 0):
    """cv2.remap maps that turn an ideal render into the real camera's image. Returns
    (map_x, map_y, coverage). With margin=0 the ideal render is width x height; coverage < 1 then
    means some output pixels sample outside it (the D455's corners: ~97%, 6.6 px of distortion) --
    render at padded_render_spec(m, margin) instead (margin >= the max distortion displacement,
    calib_intrinsics_*'s `distortion_px_max`, rounded up) and coverage reaches 1."""
    w, h = m.width, m.height
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src = undistort_pixels(np.stack([us.ravel(), vs.ravel()], axis=1), m).reshape(h, w, 2) + margin
    map_x, map_y = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)
    inside = (map_x >= 0) & (map_x <= w + 2 * margin - 1) & (map_y >= 0) & (map_y <= h + 2 * margin - 1)
    return map_x, map_y, float(inside.mean())


def apply_distortion(img: np.ndarray, maps) -> np.ndarray:
    map_x, map_y = maps[0], maps[1]
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apply a camera's lens distortion to an ideal (sim) render.")
    ap.add_argument("--intrinsics-json", type=Path, required=True)
    ap.add_argument("--image", type=Path, required=True, help="ideal render, at the json's width x height")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    m = load_model(args.intrinsics_json)
    img = cv2.imread(str(args.image))
    if img is None:
        raise SystemExit(f"could not read {args.image}")
    if (img.shape[1], img.shape[0]) != (m.width, m.height):
        raise SystemExit(f"image is {img.shape[1]}x{img.shape[0]}, intrinsics are for {m.width}x{m.height}")
    map_x, map_y, coverage = build_distort_maps(m)
    cv2.imwrite(str(args.out), apply_distortion(img, (map_x, map_y)))
    print(f"model={m.kind} coeffs={m.coeffs}; {coverage * 100:.2f}% of output pixels sample inside the render -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
