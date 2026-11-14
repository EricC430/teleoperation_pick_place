#!/usr/bin/env python3
"""Where is the bottle on the table at the start of a real episode? Outline fit on the calibrated front-left camera.

The plastic-bottle version of scripts/s10_cup_position.py (2026-11-11): same camera (10-14 ArUco extrinsics, 10-07
intrinsics with the real lens), same search (5 mm grid over +-6 cm around the nominal placement, then Nelder-Mead, then 8
restarts for the spread), same cost (mean distance from the projected outline to the nearest Canny edge, capped at 15 px).
Only the object changes: an upright solid of revolution with the profile of sim/dr_objects_bottle.json (bottle_ref), whose
outline is, per height, the left- and right-most projected point of that ring, plus the cap-top ring.

The bottle's label has many internal edges; the cost only pulls the model outline to its nearest edge, so internal edges
can still attract it. Look at the overlay (red = nominal, green = fitted, yellow = the image edge each model point
matched, <= 10 px) before trusting a number.

    python3 scripts/s9_bottle_position.py <dataset dir> <prepared dir with cl_epNN/episode.json> <out dir> <ep> [<ep> ...]
        [--frame K]   median of frames K..K+2 (default 0: the arm is at HOME)
Output: <out>/bottle_positions.json (nominal, fitted, offset mm, cost, restart spread) and bottle_epNN.jpg.
"""
import argparse
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
from bottle_mesh import radius_at  # noqa: E402

INTR = REPO / "calibration/2026-10-07_camera_intrinsics_front-left.json"
EXT = REPO / "calibration/2026-10-14_camera_extrinsics_front-left.json"
CAP_PX, MATCH_PX = 15.0, 10.0

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("dataset"); ap.add_argument("prep"); ap.add_argument("out"); ap.add_argument("eps", nargs="+", type=int)
ap.add_argument("--frame", type=int, default=0)
ap.add_argument("--objects", default=str(REPO / "sim/dr_objects_bottle.json"))
ap.add_argument("--parts", choices=["full", "cap"], default="full",
                help="cap: fit only the neck + cap (z >= shoulder_z1). The whole outline lets the fit trade distance for size "
                     "(a thinner, farther bottle on the label's edges, 2026-11-11); the cap's height is known and small, so it is not")
ap.add_argument("--method", choices=["outline", "base_color"], default="outline",
                help="base_color: lowest rows of the yellow bottle blob = front edge of the foot on the table -> ray cut with z = 0, plus one foot radius away from the camera (no fitting)")
ap.add_argument("--window_m", type=float, default=0.06, help="grid search half-width around the nominal placement")
a = ap.parse_args()
ds, prep, out = Path(a.dataset), Path(a.prep), Path(a.out); out.mkdir(parents=True, exist_ok=True)
OBJ = next(g for g in json.load(open(a.objects, encoding="utf-8"))["geometry"] if g["id"] == "bottle_ref")
Z0 = OBJ["shoulder_z1"] if a.parts == "cap" else 0.0
ZS = np.linspace(Z0, OBJ["height"], 47 if a.parts == "full" else 13); RS = radius_at(OBJ, ZS)
M = cd.load_model(INTR); K = M.K
X = json.load(open(EXT)); q = X["quat_wxyz"]
T = np.eye(4); T[:3, :3] = Rr.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); T[:3, 3] = X["pos_m"]; T[2, 3] -= X.get("table_top_z", 0.0)
CFW = np.linalg.inv(T)   # camera_from_world, OpenCV axes


def to_pixels(P):
    pc = P @ CFW[:3, :3].T + CFW[:3, 3]
    ideal = np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]
    return cd.distort_pixels(ideal, M)


AZ = np.linspace(0, 2 * np.pi, 72, endpoint=False)


def outline(x, y):
    """Left/right silhouette per ring (densified between rings) and the cap-top ring, in pixels."""
    left, right = [], []
    for z, r in zip(ZS, RS):
        uv = to_pixels(np.c_[x + r * np.cos(AZ), y + r * np.sin(AZ), np.full_like(AZ, z)])
        left.append(uv[np.argmin(uv[:, 0])]); right.append(uv[np.argmax(uv[:, 0])])
    pts = []
    for side in (np.array(left), np.array(right)):
        for p0, p1 in zip(side, side[1:]):
            n = min(max(int(np.linalg.norm(p1 - p0) / 2), 1), 200); pts.append(p0 + (p1 - p0) * (np.arange(n)[:, None] / n))
    top = to_pixels(np.c_[x + RS[-1] * np.cos(AZ), y + RS[-1] * np.sin(AZ), np.full_like(AZ, ZS[-1])])
    return np.vstack(pts), top


def make_cost(dt, nom):
    h, w = dt.shape

    def cost(p):
        if np.max(np.abs(np.asarray(p) - nom)) > a.window_m + 0.01:
            return CAP_PX
        try:
            sil, top = outline(*p)
        except (OverflowError, ValueError):
            return CAP_PX
        pts = np.vstack([sil, top])
        ins = (pts[:, 0] >= 0) & (pts[:, 0] <= w - 1) & (pts[:, 1] >= 0) & (pts[:, 1] <= h - 1)
        if ins.mean() < 0.3 or ins.sum() < (40 if a.parts == "full" else 15):
            return CAP_PX
        u = np.round(pts[ins, 0]).astype(int); v = np.round(pts[ins, 1]).astype(int)
        return float(np.mean(np.minimum(dt[v, u], CAP_PX)))
    return cost


def ray_to_plane(uv, z):
    """World point where the camera ray through pixel uv (distorted, as in the image) meets the plane Z = z."""
    und = cd.undistort_pixels(np.asarray(uv, float).reshape(1, 2), M)[0]
    d_cam = np.array([(und[0] - K[0, 2]) / K[0, 0], (und[1] - K[1, 2]) / K[1, 1], 1.0])
    d = T[:3, :3] @ d_cam; o = T[:3, 3]
    s = (z - o[2]) / d[2]
    return o + s * d


def full_outline(x, y):
    zs = np.linspace(0.0, OBJ["height"], 47); rs = radius_at(OBJ, zs); L, Rt = [], []
    for z, r in zip(zs, rs):
        uv = to_pixels(np.c_[x + r * np.cos(AZ), y + r * np.sin(AZ), np.full_like(AZ, z)])
        L.append(uv[np.argmin(uv[:, 0])]); Rt.append(uv[np.argmax(uv[:, 0])])
    return np.array(L), np.array(Rt)


def cap_color(ep, E, nom, img):
    """The cap is the top of the yellow blob; its pixel centroid is taken as the image of the cap's centre
    (on the bottle axis at z = height - cap_h / 2), and the camera ray through it is cut with that plane."""
    H, hc = OBJ["height"], OBJ["cap_h"]; zc = H - hc / 2
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    yel = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 40) & (hsv[..., 1] >= 110) & (hsv[..., 2] >= 120)).astype(np.uint8)
    yel = cv2.morphologyEx(yel, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(yel)
    # the bottle's blob: the largest one whose box contains the nominal cap's projection (within 60 px)
    pc = to_pixels(np.array([[nom[0], nom[1], zc]]))[0]
    cand = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 300 and
            st[i, 0] - 60 <= pc[0] <= st[i, 0] + st[i, 2] + 60 and st[i, 1] - 60 <= pc[1] <= st[i, 1] + st[i, 3] + 60]
    ov = img.copy()
    if not cand:
        print(f"ep{ep:02d} {E['placement']:>4}: no yellow blob near the nominal cap -> skipped", flush=True); return
    b = max(cand, key=lambda i: st[i, cv2.CC_STAT_AREA]); top = st[b, 1]
    # cap height in pixels at the nominal place (projected z H - hc .. H on the axis)
    hpx = abs(to_pixels(np.array([[nom[0], nom[1], H - hc]]))[0, 1] - to_pixels(np.array([[nom[0], nom[1], H]]))[0, 1])
    m = (lab == b); m[int(top + hpx * 1.15):, :] = False
    vs, us = np.nonzero(m); uv_cap = np.array([us.mean(), vs.mean()])
    # The camera looks almost level: a ray through the cap runs nearly parallel to the cap's plane (1 px -> metres), so the
    # position comes from the BASE instead. The lowest rows of the bottle blob are the base's front edge on the table
    # (z = 0, the point of the foot circle nearest the camera); the ray through it is cut with z = 0 and the centre is one
    # foot radius further along the horizontal direction away from the camera.
    # the clear neck separates the cap's blob from the label's: take every yellow blob below the cap whose columns overlap it
    cu = uv_cap[0]; w = max(st[b, 2], 20)
    body = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] > 300 and st[i, 1] >= top and
            st[i, 0] <= cu + 2 * w and st[i, 0] + st[i, 2] >= cu - 2 * w]
    vlim = to_pixels(np.array([[nom[0], nom[1], 0.0]]))[0, 1] + 40   # nothing of this bottle is far below its nominal foot
    bl = np.isin(lab, body); bl[int(vlim):, :] = False; vb = np.nonzero(bl)[0]; vmax = np.percentile(vb, 99.5)
    rowsel = bl & (np.arange(bl.shape[0])[:, None] >= vmax - 3)
    vs2, us2 = np.nonzero(rowsel); uv = np.array([np.median(us2), vmax])
    Pb = ray_to_plane(uv, 0.0); away = Pb[:2] - T[:2, 3]; away /= np.linalg.norm(away)
    fit = Pb[:2] + radius_at(OBJ, np.array([0.0]))[0] * away; off = (fit - nom) * 1000
    m = m | rowsel
    ov[m] = (0.5 * ov[m] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
    for xy, col in ((nom, (0, 0, 255)), (fit, (0, 255, 0))):
        L, Rt = full_outline(*xy)
        cv2.polylines(ov, [np.round(L).astype(np.int32)], False, col, 1, cv2.LINE_AA); cv2.polylines(ov, [np.round(Rt).astype(np.int32)], False, col, 1, cv2.LINE_AA)
    cv2.drawMarker(ov, (int(round(uv[0])), int(round(uv[1]))), (255, 255, 255), cv2.MARKER_CROSS, 12, 1)
    txt = f"ep{ep} {E['placement']} base_color f{a.frame}: magenta = cap + base rows used, red nominal, green from the base  offset ({off[0]:+.0f},{off[1]:+.0f}) mm"
    cv2.putText(ov, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3); cv2.putText(ov, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out / f"bottle_ep{ep:02d}.jpg"), ov)
    rows.append({"episode": ep, "placement": E["placement"], "frame": a.frame, "method": "base_color", "nominal_xy_m": nom.tolist(),
                 "fit_xy_m": fit.tolist(), "offset_mm": off.tolist(), "cap_px": int(m.sum()), "cap_uv": uv.tolist()})
    print(f"ep{ep:02d} {E['placement']:>4} nominal ({nom[0]:.3f},{nom[1]:.3f}) from base ({fit[0]:.3f},{fit[1]:.3f}) offset ({off[0]:+.0f},{off[1]:+.0f}) mm | base pixel ({uv[0]:.0f},{uv[1]:.0f}) -> table ({Pb[0]:.3f},{Pb[1]:.3f})", flush=True)


rows = []
for ep in a.eps:
    E = json.load(open(prep / f"cl_ep{ep:02d}/episode.json")); nom = np.array(E["cup_xy_m"], float)
    frames = []
    for k, f in enumerate(C.real_frames(ds, ep)):
        if k >= a.frame:
            frames.append(f)
        if k >= a.frame + 2:
            break
    img = np.median(np.stack(frames), 0).astype(np.uint8)
    if a.method == "base_color":
        cap_color(ep, E, nom, img); continue
    edges = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (0, 0), 1.2), 30, 90)
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3); cost = make_cost(dt, nom)
    best = (cost(nom), nom)
    for dx in np.arange(-a.window_m, a.window_m + 1e-4, 0.005):
        for dy in np.arange(-a.window_m, a.window_m + 1e-4, 0.005):
            p = nom + [dx, dy]; c = cost(p)
            if c < best[0]:
                best = (c, p)
    fit = minimize(cost, best[1], method="Nelder-Mead", options={"xatol": 5e-4, "fatol": 1e-4}).x
    c_nom, c_fit = cost(nom), cost(fit)
    starts = [fit + 0.01 * np.array([np.cos(t), np.sin(t)]) for t in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
    refit = np.array([minimize(cost, s0, method="Nelder-Mead", options={"xatol": 5e-4, "fatol": 1e-4}).x for s0 in starts])
    spread = float(np.max(np.linalg.norm(refit - fit, axis=1)) * 1000)
    ov = img.copy()
    for P, col in ((nom, (0, 0, 255)), (fit, (0, 255, 0))):
        sil, top = outline(*P)
        for poly in (sil, top):
            cv2.polylines(ov, [np.round(poly).astype(np.int32)], False, col, 1, cv2.LINE_AA)
    sil, top = outline(*fit); ys, xs = np.nonzero(edges); E_pts = np.c_[xs, ys]
    for p in np.vstack([sil, top])[::3]:
        d = np.linalg.norm(E_pts - p, axis=1); i = int(d.argmin())
        if d[i] <= MATCH_PX:
            cv2.circle(ov, (int(E_pts[i, 0]), int(E_pts[i, 1])), 2, (0, 255, 255), -1)
    off = (fit - nom) * 1000
    txt = f"ep{ep} {E['placement']} {a.parts} f{a.frame} nominal red {c_nom:.1f}px  fit green {c_fit:.1f}px  offset ({off[0]:+.0f},{off[1]:+.0f}) mm"
    cv2.putText(ov, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3); cv2.putText(ov, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imwrite(str(out / f"bottle_ep{ep:02d}.jpg"), ov)
    rows.append({"episode": ep, "placement": E["placement"], "frame": a.frame, "parts": a.parts, "nominal_xy_m": nom.tolist(), "fit_xy_m": fit.tolist(),
                 "offset_mm": off.tolist(), "chamfer_nominal_px": c_nom, "chamfer_fit_px": c_fit, "restart_spread_mm": spread})
    print(f"ep{ep:02d} {E['placement']:>4} nominal ({nom[0]:.3f},{nom[1]:.3f}) fit ({fit[0]:.3f},{fit[1]:.3f}) offset ({off[0]:+.0f},{off[1]:+.0f}) mm "
          f"| chamfer {c_nom:.1f} -> {c_fit:.1f} px | restart spread {spread:.1f} mm", flush=True)
json.dump(rows, open(out / "bottle_positions.json", "w"), indent=1)
