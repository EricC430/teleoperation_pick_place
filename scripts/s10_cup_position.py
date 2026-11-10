#!/usr/bin/env python3
"""S10: where is the cup on the table at frame 0 of a real episode? CAD-outline fit on the calibrated front-left camera.

Same idea as scripts/fit_camera_from_cad.py (2026-11-05), turned around: the camera is fixed (10-14 ArUco extrinsics, 10-07
intrinsics with the real lens), the known object (paper cup frustum: top 75 mm, bottom 50 mm, height 93 mm = sim/dr_config
cup_ref) is moved on the table (x, y in the sim world = the frame the replay puts the cup in) until its projected outline
(silhouette + rim ellipse) sits on the image edges (chamfer, robust cap). Frame 0: the arm is at HOME, not in front of the cup.

Output per episode: nominal placement (episode.json cup_xy_m), fitted x/y, offset in mm, chamfer cost at both, and an overlay JPG:
yellow dots = the image edge each model point was matched to (<= 10 px), red = outline at the nominal position, green = fitted.
Look at the overlay before trusting a number (docs: feedback "measure edges, show detected points").

    python3 scripts/s10_cup_position.py <dataset dir> <prepared dir with cl_epNN/episode.json> <out dir> <ep> [<ep> ...]
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rr

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "sim"), str(REPO / "scripts")]
import camera_distortion as cd  # noqa: E402
import compose_replay_ext_video as C  # noqa: E402

TOP_R, BOT_R, H = 0.0375, 0.025, 0.093
INTR = REPO / "calibration/2026-10-07_camera_intrinsics_front-left.json"
EXT = REPO / "calibration/2026-10-14_camera_extrinsics_front-left.json"
CAP_PX, MATCH_PX = 15.0, 10.0

ds, prep, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]); eps = [int(x) for x in sys.argv[4:]]
out.mkdir(parents=True, exist_ok=True)
M = cd.load_model(INTR); K = M.K
X = json.load(open(EXT)); q = X["quat_wxyz"]
T = np.eye(4); T[:3, :3] = Rr.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = X["pos_m"]; T[2, 3] -= X.get("table_top_z", 0.0)
CFW = np.linalg.inv(T)   # camera_from_world, OpenCV axes (same convention as sim/dr_render.py TFL)


def cone_points(x, y):
    az = np.linspace(0, 2 * np.pi, 90, endpoint=False)
    rings = []
    for t in np.linspace(0, 1, 7):
        r = BOT_R + t * (TOP_R - BOT_R); rings.append(np.c_[x + r * np.cos(az), y + r * np.sin(az), t * H * np.ones_like(az)])
    return np.vstack(rings), rings[-1]


def to_pixels(P):
    pc = P @ CFW[:3, :3].T + CFW[:3, 3]
    ideal = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
    return cd.distort_pixels(ideal, M)


def outline(x, y):
    """Silhouette (convex hull of the projected cone surface) plus the whole rim ring."""
    P, top = cone_points(x, y)
    uv = to_pixels(P).astype(np.float32); hull = cv2.convexHull(uv).reshape(-1, 2)
    # densify the hull polygon
    pts = []
    for a, b in zip(hull, np.roll(hull, -1, 0)):
        n = min(max(int(np.linalg.norm(b - a) / 2), 1), 2000); pts.append(a + (b - a) * (np.arange(n)[:, None] / n))
    return np.vstack(pts), to_pixels(top)


def make_cost(dt):
    h, w = dt.shape

    def cost(p):
        if np.max(np.abs(np.asarray(p) - NOM0)) > 0.07:   # search window = grid +-6 cm (+1 cm slack)
            return CAP_PX
        try:
            hull, rim = outline(*p)
        except (OverflowError, ValueError):   # a point at/behind the camera plane -> inf pixels
            return CAP_PX
        pts = np.vstack([hull, rim])
        # points outside the image carry no information (cup cut by the frame): skip them instead of clipping to the border
        ins = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
        if ins.mean() < 0.3 or ins.sum() < 40:
            return CAP_PX
        u = np.round(pts[ins, 0]).astype(int); v = np.round(pts[ins, 1]).astype(int)
        return float(np.mean(np.minimum(dt[v, u], CAP_PX)))
    return cost


rows = []
for ep in eps:
    E = json.load(open(prep / f"cl_ep{ep:02d}/episode.json")); nom = np.array(E["cup_xy_m"], float)
    frames = []
    for k, f in enumerate(C.real_frames(ds, ep)):
        frames.append(f)
        if k >= 2:
            break
    img = np.median(np.stack(frames), 0).astype(np.uint8)
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.2)
    edges = cv2.Canny(g, 30, 90)
    NOM0 = nom; dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3); cost = make_cost(dt)
    best = (cost(nom), nom)
    for dx in np.arange(-0.06, 0.0601, 0.005):
        for dy in np.arange(-0.06, 0.0601, 0.005):
            p = nom + [dx, dy]; c = cost(p)
            if c < best[0]:
                best = (c, p)
    r = minimize(cost, best[1], method="Nelder-Mead", options={"xatol": 5e-4, "fatol": 1e-4})
    fit = r.x; c_nom, c_fit = cost(nom), cost(fit)
    # robustness: spread of the fit over 8 starting points around the optimum (should come back to the same place)
    starts = [fit + 0.01 * np.array([np.cos(a), np.sin(a)]) for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    refit = np.array([minimize(cost, s0, method="Nelder-Mead", options={"xatol": 5e-4, "fatol": 1e-4}).x for s0 in starts])
    spread = float(np.max(np.linalg.norm(refit - fit, axis=1)) * 1000)
    # overlay
    ov = img.copy()
    for P, col in ((nom, (0, 0, 255)), (fit, (0, 255, 0))):
        hull, rim = outline(*P)
        cv2.polylines(ov, [np.round(hull).astype(np.int32)], True, col, 1, cv2.LINE_AA); cv2.polylines(ov, [np.round(rim).astype(np.int32)], True, col, 1, cv2.LINE_AA)
    hull, rim = outline(*fit); pts = np.vstack([hull, rim]); ys, xs = np.nonzero(edges)
    ep_pts = np.c_[xs, ys]; matched = 0
    for p in pts[::3]:
        d = np.linalg.norm(ep_pts - p, axis=1); i = int(d.argmin())
        if d[i] <= MATCH_PX:
            cv2.circle(ov, (int(ep_pts[i, 0]), int(ep_pts[i, 1])), 2, (0, 255, 255), -1); matched += 1
    cv2.putText(ov, f"ep{ep} {E['placement']} nominal red {c_nom:.1f}px  fit green {c_fit:.1f}px  offset ({(fit[0]-nom[0])*1000:+.0f},{(fit[1]-nom[1])*1000:+.0f}) mm  yellow = matched edge pts", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(ov, f"ep{ep} {E['placement']} nominal red {c_nom:.1f}px  fit green {c_fit:.1f}px  offset ({(fit[0]-nom[0])*1000:+.0f},{(fit[1]-nom[1])*1000:+.0f}) mm  yellow = matched edge pts", (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out / f"cup_ep{ep:02d}.jpg"), ov)
    rows.append({"episode": ep, "placement": E["placement"], "nominal_xy_m": nom.tolist(), "fit_xy_m": fit.tolist(), "offset_mm": ((fit - nom) * 1000).tolist(),
                 "chamfer_nominal_px": c_nom, "chamfer_fit_px": c_fit, "restart_spread_mm": spread})
    print(f"ep{ep:02d} {E['placement']:>4} nominal ({nom[0]:.3f},{nom[1]:.3f}) fit ({fit[0]:.3f},{fit[1]:.3f}) offset ({(fit[0]-nom[0])*1000:+.0f},{(fit[1]-nom[1])*1000:+.0f}) mm "
          f"| chamfer {c_nom:.1f} -> {c_fit:.1f} px | restart spread {spread:.1f} mm")
json.dump(rows, open(out / "cup_positions.json", "w"), indent=1)
