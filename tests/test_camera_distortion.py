import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sim"))
import camera_distortion as cd  # noqa: E402

# D455 intrinsics read on the lab machine 2026-10-06 (calib_intrinsics_realsense.py), 848x480.
D455 = cd.make_model(
    427.151, 425.969, 431.073, 243.667, 848, 480, "distortion.inverse_brown_conrady",
    [-0.054625071585178375, 0.06350171566009521, -0.0012396894162520766, -0.0006715220515616238, -0.020685186609625816],
)
UVC = cd.make_model(600.0, 600.0, 320.0, 240.0, 640, 480, "opencv_brown_conrady", [0.05, -0.02, 0.001, 0.0005, 0.0])


def _random_pixels(m, n=500, seed=0):
    rng = np.random.default_rng(seed)
    return np.stack([rng.uniform(0, m.width - 1, n), rng.uniform(0, m.height - 1, n)], axis=1)


def test_rs_bc_reproduces_the_sdk_displacement_stats():
    # The SDK reported p50/p95/max = 3.68/6.21/6.61 px for this camera over a 9x9 grid. Only the
    # iterative librealsense formulation hits all three; the direct-polynomial reading does not.
    us, vs = np.meshgrid(np.linspace(0, 847, 9), np.linspace(0, 479, 9))
    pts = np.stack([us.ravel(), vs.ravel()], axis=1)
    disp = np.linalg.norm(cd.undistort_pixels(pts, D455) - pts, axis=1)
    assert abs(np.percentile(disp, 50) - 3.68) < 0.01
    assert abs(np.percentile(disp, 95) - 6.21) < 0.01
    assert abs(disp.max() - 6.61) < 0.01


def test_round_trip_rs_bc():
    pts = _random_pixels(D455)
    assert np.abs(cd.undistort_pixels(cd.distort_pixels(pts, D455), D455) - pts).max() < 1e-4


def test_round_trip_opencv():
    pts = _random_pixels(UVC)
    assert np.abs(cd.undistort_pixels(cd.distort_pixels(pts, UVC), UVC) - pts).max() < 1e-4


def test_kind_from_name():
    assert cd.kind_from_name("distortion.inverse_brown_conrady") == "rs_bc"
    assert cd.kind_from_name("distortion.brown_conrady") == "rs_bc"
    assert cd.kind_from_name("distortion.none") == "none"
    assert cd.kind_from_name(None) == "opencv"
    assert cd.kind_from_name("opencv_brown_conrady") == "opencv"


def test_remap_agrees_with_pointwise_distortion():
    # White squares at known ideal pixel centres; after the remap each must sit where distort_pixels says.
    for m in (D455, UVC):
        centres = np.array([[x, y] for y in (60, 240, 420) for x in (80, m.width / 2, m.width - 80)], dtype=np.float64)
        ideal = np.zeros((m.height, m.width), np.uint8)
        for cx, cy in centres:
            ideal[int(cy) - 5 : int(cy) + 6, int(cx) - 5 : int(cx) + 6] = 255
        maps = cd.build_distort_maps(m)
        distorted = cd.apply_distortion(ideal, maps)
        _, _, _, found = cv2.connectedComponentsWithStats((distorted > 127).astype(np.uint8))
        found = found[1:]
        expected = cd.distort_pixels(centres + 0.0, m)
        for e in expected:
            assert np.linalg.norm(found - e, axis=1).min() < 0.6


def test_margin_removes_black_corners_and_keeps_geometry():
    assert cd.build_distort_maps(D455)[2] < 1.0  # D455 corners sample outside an unpadded render
    margin = 8
    assert cd.build_distort_maps(D455, margin)[2] == 1.0
    w, h, K = cd.padded_render_spec(D455, margin)
    centres = np.array([[100.0, 80.0], [424.0, 240.0], [750.0, 400.0]])
    ideal = np.zeros((h, w), np.uint8)
    for cx, cy in centres:  # ideal pixel (x, y) of the real frame lives at (x + margin, y + margin) in the padded render
        ideal[int(cy + margin) - 5 : int(cy + margin) + 6, int(cx + margin) - 5 : int(cx + margin) + 6] = 255
    distorted = cd.apply_distortion(ideal, cd.build_distort_maps(D455, margin))
    assert distorted.shape == (D455.height, D455.width)
    _, _, _, found = cv2.connectedComponentsWithStats((distorted > 127).astype(np.uint8))
    for e in cd.distort_pixels(centres, D455):
        assert np.linalg.norm(found[1:] - e, axis=1).min() < 0.6
