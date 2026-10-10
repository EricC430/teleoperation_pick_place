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
  render -> real image           an Isaac render -> what the real camera would show (S5 frames):
                                 render_spec() + isaac_K() for the sim camera, then
                                 build_render_to_real_maps() + apply_distortion().

Why the sim render is NOT simply "the real K": `[已查證 2026-10-08]` in the isaac-lab container,
isaaclab/utils/sensors.py convert_camera_intrinsics_to_usd drops aperture offsets ("c_x and c_y will
be half of width and height") and averages fx/fy, and camera.py _update_intrinsic_matrices sets
f_y = f_x ("rendering does not use aperture offsets or vertical aperture"). So Isaac can only render a
square-pixel image with the optical axis at the centre. render_spec() sizes such an image wide enough
to contain every ray the real camera sees; the remap then applies the real principal point, the real
fx/fy and the lens distortion in one step. The D455's principal point is (7, 4) px off-centre and the
wrist camera's (-32, -3) px -- ignoring it would shift the whole image by that much.

Pure numpy apart from cv2.remap / imread / imwrite: the container's OpenCV is 5.0 and has no
cv2.undistortPointsIter (checked 2026-10-08), so the OpenCV-model inverse is done here.

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


def _opencv_inverse(xd, yd, c, iters: int):
    """OpenCV's own undistortPoints fixed-point iteration, then Newton steps: with the wrist lens
    (k1 = -0.53, ~130 px of distortion at the corners) the fixed point alone converges slowly."""
    k1, k2, p1, p2, k3 = c
    x, y = xd.copy(), yd.copy()
    for _ in range(iters):
        r2 = x * x + y * y
        icdist = 1.0 / (1.0 + ((k3 * r2 + k2) * r2 + k1) * r2)
        dx = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        dy = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x, y = (xd - dx) * icdist, (yd - dy) * icdist
    eps = 1e-7
    for _ in range(4):
        fx0, fy0 = _opencv_forward(x, y, c)
        ax, ay = _opencv_forward(x + eps, y, c)
        bx, by = _opencv_forward(x, y + eps, c)
        j11, j21, j12, j22 = (ax - fx0) / eps, (ay - fy0) / eps, (bx - fx0) / eps, (by - fy0) / eps
        rx, ry = fx0 - xd, fy0 - yd
        det = j11 * j22 - j12 * j21
        x, y = x - (j22 * rx - j12 * ry) / det, y - (j11 * ry - j21 * rx) / det
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
    xo, yo = _norm(pts, m.K)
    x, y = (_opencv_inverse if m.kind == "opencv" else _rs_inverse)(xo, yo, m.coeffs, iters)
    return _pix(x, y, m.K)


def distort_pixels(ideal_pts, m: CameraModel) -> np.ndarray:
    """Inverse of undistort_pixels: ideal-pinhole pixels -> pixels the real camera would record."""
    ideal = np.asarray(ideal_pts, dtype=np.float64).reshape(-1, 2)
    if m.kind == "none" or not any(m.coeffs):
        return ideal.copy()
    x, y = _norm(ideal, m.K)
    dx, dy = (_opencv_forward if m.kind == "opencv" else _rs_forward)(x, y, m.coeffs)
    return _pix(dx, dy, m.K)


def _real_pixel_rays(m: CameraModel):
    """Normalised ideal-pinhole rays (x, y) of every pixel of the real image, shaped (height, width)."""
    us, vs = np.meshgrid(np.arange(m.width, dtype=np.float64), np.arange(m.height, dtype=np.float64))
    ideal = undistort_pixels(np.stack([us.ravel(), vs.ravel()], axis=1), m)
    x = (ideal[:, 0] - m.K[0, 2]) / m.K[0, 0]
    y = (ideal[:, 1] - m.K[1, 2]) / m.K[1, 1]
    return x.reshape(us.shape), y.reshape(us.shape)


def render_spec(m: CameraModel, f: float | None = None, pad_px: int = 2):
    """(width, height, f) of the ideal image Isaac should render for this camera: square pixels,
    optical axis at the centre, wide enough to contain every ray the real image samples. f defaults
    to the mean of the real fx/fy, so the centre of the render has the real camera's resolution."""
    f = float(f if f is not None else (m.K[0, 0] + m.K[1, 1]) / 2.0)
    x, y = _real_pixel_rays(m)
    half_w = f * float(np.abs(x).max()) + pad_px
    half_h = f * float(np.abs(y).max()) + pad_px
    return 2 * int(np.ceil(half_w)), 2 * int(np.ceil(half_h)), f


def isaac_K(width: int, height: int, f: float) -> np.ndarray:
    """The matrix to hand to Isaac Lab's Camera.set_intrinsic_matrices for a render_spec() render.
    Isaac's own convention puts the axis at (W/2, H/2); anything else is dropped with a warning."""
    return np.array([[f, 0.0, width / 2.0], [0.0, f, height / 2.0], [0.0, 0.0, 1.0]])


def build_render_to_real_maps(m: CameraModel, render_w: int, render_h: int, f: float):
    """cv2.remap maps: Isaac render (render_w x render_h, square pixels, centred axis, focal f) ->
    the real camera's image (m.width x m.height, its own principal point, fx/fy and distortion).
    Returns (map_x, map_y, coverage); coverage should be 1.0 for a render_spec() render.

    In remap's pixel-centre coordinates a symmetric frustum's axis is at ((W-1)/2, (H-1)/2); Isaac
    reports W/2, H/2 in its own continuous convention. The two differ by half a pixel, which is the
    remaining uncertainty here."""
    x, y = _real_pixel_rays(m)
    map_x = (f * x + (render_w - 1) / 2.0).astype(np.float32)
    map_y = (f * y + (render_h - 1) / 2.0).astype(np.float32)
    inside = (map_x >= 0) & (map_x <= render_w - 1) & (map_y >= 0) & (map_y <= render_h - 1)
    return map_x, map_y, float(inside.mean())


def build_distort_maps(m: CameraModel):
    """cv2.remap maps from an ideal image WITH THE REAL K (same size) to the real camera's image.
    Host-side utility; for Isaac renders use build_render_to_real_maps (Isaac cannot render the
    real K). Returns (map_x, map_y, coverage); corners sample outside the input when coverage < 1."""
    w, h = m.width, m.height
    us, vs = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    src = undistort_pixels(np.stack([us.ravel(), vs.ravel()], axis=1), m).reshape(h, w, 2)
    map_x, map_y = src[..., 0].astype(np.float32), src[..., 1].astype(np.float32)
    inside = (map_x >= 0) & (map_x <= w - 1) & (map_y >= 0) & (map_y <= h - 1)
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
