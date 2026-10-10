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


# Innomaker wrist lens as calibrated 2026-10-07 (calib_intrinsics_checkerboard.py, 23 shots): ~130 px
# of barrel distortion at the corners and a principal point 32 px off-centre -- the hard case.
WRIST = cd.make_model(
    694.6581748746704, 696.6725868716137, 288.19156625247837, 236.72215105630147, 640, 480, "opencv_brown_conrady",
    [-0.5255522197350703, 0.4725669886410813, 0.003127505905239609, 0.0015699038457483063, -0.37031118447591643],
)


def test_numpy_opencv_inverse_matches_opencv():
    # The container's OpenCV 5 has no undistortPointsIter, so camera_distortion does the inverse in
    # numpy. Check it against OpenCV's own where OpenCV has it.
    if not hasattr(cv2, "undistortPointsIter"):
        return
    us, vs = np.meshgrid(np.arange(0, 640, 7.0), np.arange(0, 480, 7.0))
    pts = np.stack([us.ravel(), vs.ravel()], axis=1)
    crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 200, 1e-14)
    ref = cv2.undistortPointsIter(pts.reshape(-1, 1, 2), WRIST.K, np.array(WRIST.coeffs), None, WRIST.K, crit).reshape(-1, 2)
    assert np.abs(cd.undistort_pixels(pts, WRIST) - ref).max() < 1e-3


def test_round_trip_wrist():
    pts = _random_pixels(WRIST)
    assert np.abs(cd.distort_pixels(cd.undistort_pixels(pts, WRIST), WRIST) - pts).max() < 1e-6


def test_render_to_real_maps_land_rays_where_the_real_camera_would():
    # A centred, square-pixel render (all Isaac can produce) with dots on known rays; after the
    # remap every dot must sit where the REAL camera (its own cx/cy, fx/fy, distortion) puts that ray.
    rays = np.array([[-0.25, -0.15], [0.0, 0.0], [0.3, 0.2], [-0.35, 0.25], [0.1, -0.2]])
    for m in (D455, WRIST):
        w, h, f = cd.render_spec(m)
        map_x, map_y, coverage = cd.build_render_to_real_maps(m, w, h, f)
        assert coverage == 1.0
        assert map_x.shape == (m.height, m.width)
        render = np.zeros((h, w), np.uint8)
        drawn = []
        for x, y in rays:  # draw on integer pixels, then use the ray through the pixel actually drawn
            iu, iv = int(round(f * x + (w - 1) / 2.0)), int(round(f * y + (h - 1) / 2.0))
            render[iv - 4 : iv + 5, iu - 4 : iu + 5] = 255
            drawn.append(((iu - (w - 1) / 2.0) / f, (iv - (h - 1) / 2.0) / f))
        drawn = np.array(drawn)
        real = cd.apply_distortion(render, (map_x, map_y)).astype(np.float64)
        n, labels = cv2.connectedComponents((real > 0).astype(np.uint8))
        vs, us = np.mgrid[0 : m.height, 0 : m.width]
        found = np.array([[(us * real)[labels == k].sum() / real[labels == k].sum(),
                           (vs * real)[labels == k].sum() / real[labels == k].sum()] for k in range(1, n)])
        ideal_real_px = np.stack([m.K[0, 0] * drawn[:, 0] + m.K[0, 2], m.K[1, 1] * drawn[:, 1] + m.K[1, 2]], axis=1)
        for e in cd.distort_pixels(ideal_real_px, m):
            assert 6 <= e[0] < m.width - 6 and 6 <= e[1] < m.height - 6  # every test ray is well inside
            assert np.linalg.norm(found - e, axis=1).min() < 0.3
