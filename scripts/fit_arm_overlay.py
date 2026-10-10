#!/usr/bin/env python
"""Fit where the arm sits in the front-left image: project the URDF meshes through FK and match the dark arm.

Both fits use FK (joint readings -> joint_mapping -> URDF chain) and the 10/07 ArUco front-left camera:
  (A) base offset   camera fixed at its ArUco (paper-frame) pose; solve the arm base's planar pose on the
                    paper (dx, dy, yaw) = "compare the sim arm with the video arm".
  (B) camera-from-arm  base fixed at the paper origin; solve the camera's 6-DoF pose from the arm silhouette
                    alone, then compare it with the ArUco camera = "locate the camera relative to the arm".
Reading it: a rigid base/paper offset gives the SAME (dx, dy, yaw) on every subset of frames (low/high,
left/right) in (A); an FK error (joint zeros, link lengths, droop) makes the subsets disagree, because its
image effect changes with the pose. (B) has more freedom: camera rotation can absorb part of an FK error.

Silhouette: per link, the convex hull of the projected mesh vertices (over-fills concave brackets; the same
for every candidate, so it biases IoU levels, not which candidate wins). Real arm: pixels with V < 50, minus
pixels dark in most sampled frames (the chair arm). Cables are dark too and count against every candidate.

    python3 scripts/fit_arm_overlay.py            # writes calibration/<date>_arm_overlay_fit.json, outputs/arm_overlay/
"""
from __future__ import annotations

import datetime as dt
import json
import math
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import minimize

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "sim"))
import camera_distortion as cd  # noqa: E402
import joint_mapping as jm  # noqa: E402

MESH_DIR = Path.home() / "isaaclab_volume/assets/open_manipulator_description/meshes/omx_f"
MESHES = ["follower_01_base", "follower_02_base_tilt_Revised", "follower_03_middle_verticle", "follower_04_middle_horizontal",
          "follower_05_tip", "follower_06_pan_Revised", "follower_07_gripper_motorized", "follower_08_gripper_gear"]
CHAIN = [((-0.01125, 0, 0.034), "z"), ((0, 0, 0.0635), "y"), ((0.0415, 0, 0.11315), "y"), ((0.162, 0, 0), "y"), ((0.0287, 0, 0), "x")]
DATASET = _REPO / "data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_normal_A1"
EXTR = _REPO / "calibration/2026-10-07_camera_extrinsics_front-left.json"
INTR = _REPO / "calibration/2026-10-07_camera_intrinsics_front-left.json"
RISER = 0.146   # [柏宇說 2026-10-07] underside of the base above the table
SCALE = 0.5     # masks at half resolution
DARK_V = 50


def load_stl(path: Path, keep: int = 1500, seed: int = 0) -> np.ndarray:
    data = path.read_bytes()
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v = np.unique(np.round(rec["v"].reshape(-1, 3).astype(np.float64) * 0.001, 4), axis=0)
    return v[np.random.default_rng(seed).choice(len(v), min(keep, len(v)), replace=False)]


def tr(xyz):
    t = np.eye(4); t[:3, 3] = xyz; return t


def rot(axis: str, a: float):
    c, s = math.cos(a), math.sin(a); r = np.eye(4)
    i, j = {"x": (1, 2), "y": (2, 0), "z": (0, 1)}[axis]
    r[i, i], r[i, j], r[j, i], r[j, j] = c, -s, s, c
    return r


def link_poses(state6, base: np.ndarray) -> list[np.ndarray]:
    q = jm.lerobot_to_urdf_rad(np.asarray(state6, dtype=np.float64))
    T, out = base.copy(), [base.copy()]
    for (o, ax), qi in zip(CHAIN, q[:5]):
        T = T @ tr(o) @ rot(ax, qi); out.append(T.copy())
    out.append(T @ tr((0.0295, 0.0075, 0)) @ rot("z", q[5]))
    out.append(T @ tr((0.0295, -0.0108, 0)) @ rot("z", -q[5]))
    return out


class Camera:
    def __init__(self, extr: Path, intr: Path):
        e = json.loads(extr.read_text(encoding="utf-8"))
        w, x, y, z = e["quat_wxyz"]
        self.R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                           [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                           [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        self.t, self.table_z, self.model = np.array(e["pos_m"]), e["table_top_z"], cd.load_model(intr)

    def moved(self, d_cm, rvec_deg) -> "Camera":
        c = object.__new__(Camera)
        c.R = cv2.Rodrigues(np.radians(np.asarray(rvec_deg, dtype=np.float64)))[0] @ self.R
        c.t, c.table_z, c.model = self.t + np.asarray(d_cm) / 100.0, self.table_z, self.model
        return c

    def project(self, X: np.ndarray):
        v = (X - self.t) @ self.R
        ok = v[:, 2] > 0.02
        K = self.model.K
        xn, yn = v[ok, 0] / v[ok, 2], v[ok, 1] / v[ok, 2]
        ideal = np.stack([xn * K[0, 0] + K[0, 2], yn * K[1, 1] + K[1, 2]], 1)
        # the lens polynomial is only meaningful inside the field of view; far outside it diverges (inf/NaN)
        inside = xn * xn + yn * yn < 2.0
        if inside.any():
            ideal[inside] = cd.distort_pixels(ideal[inside], self.model)
        return np.clip(ideal, -4000, 4000)


def silhouette(cam: Camera, meshes, state6, base: np.ndarray, shape) -> np.ndarray:
    m = np.zeros(shape, np.uint8)
    for T, v in zip(link_poses(state6, base), meshes):
        p = cam.project(v @ T[:3, :3].T + T[:3, 3])
        if len(p) >= 3:
            cv2.fillConvexPoly(m, cv2.convexHull((p * SCALE).astype(np.int32)), 1)
    return m


def base_pose(cam: Camera, dx_cm=0.0, dy_cm=0.0, yaw_deg=0.0) -> np.ndarray:
    return tr((dx_cm / 100.0, dy_cm / 100.0, cam.table_z + RISER)) @ rot("z", math.radians(yaw_deg))


def load_frames(stride: int, max_frames: int):
    import pyarrow.parquet as pq

    t = pq.read_table(next((DATASET / "data").rglob("*.parquet")), columns=["index", "episode_index", "frame_index", "observation.state"]).to_pydict()
    want = {int(i): (int(e), int(f), s) for i, e, f, s in zip(t["index"], t["episode_index"], t["frame_index"], t["observation.state"]) if int(i) % stride == 0}
    cap = cv2.VideoCapture(str(next((DATASET / "videos/observation.images.front-left").rglob("*.mp4"))))
    out, i = [], 0
    while True:
        ok, img = cap.read()
        if not ok:
            break
        if i in want:
            e, f, s = want[i]
            hsv = cv2.cvtColor(cv2.resize(img, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2HSV)
            out.append({"index": i, "episode": e, "frame": f, "state": np.array(s), "dark": (hsv[..., 2] < DARK_V).astype(np.uint8), "img": img})
        i += 1
    static = np.mean([f["dark"] for f in out], axis=0) > 0.5
    for f in out:
        f["dark"][static] = 0
    rng = np.random.default_rng(0)
    return [out[k] for k in sorted(rng.choice(len(out), min(max_frames, len(out)), replace=False))], static


def iou_terms(pred, frame, roi):
    p, d = pred.astype(bool) & roi, frame["dark"].astype(bool) & roi
    return (p & d).sum(), (p | d).sum()


def mean_iou(frames, rois, sil_fn) -> float:
    vals = []
    for f, roi in zip(frames, rois):
        inter, union = iou_terms(sil_fn(f), f, roi)
        vals.append(inter / union if union else 0.0)
    return float(np.mean(vals))


def main() -> int:
    cam = Camera(EXTR, INTR)
    meshes = [load_stl(MESH_DIR / f"{n}.stl") for n in MESHES]
    frames, static = load_frames(stride=5, max_frames=120)
    shape = frames[0]["dark"].shape
    nominal = base_pose(cam)
    keep, rois = [], []
    for f in frames:  # frames where the arm should be in view; ROI = generous band around the nominal arm
        s = silhouette(cam, meshes, f["state"], nominal, shape)
        if s.sum() > 400 and f["dark"].sum() > 200:
            f["pinch_z_cm"] = (link_poses(f["state"], nominal)[5] @ np.array([0.088, 0, 0, 1]))[2] * 100 - (cam.table_z + RISER) * 100 + RISER * 100
            f["pan"] = f["state"][0]
            keep.append(f); rois.append(cv2.dilate(s, np.ones((61, 61), np.uint8)).astype(bool) & ~static)
    frames = keep
    print(f"{len(frames)} frames with the arm in view (normal_A1, every 5th frame sampled, front-left ArUco camera, riser {RISER} m)")

    def fit_A(sel):
        F, Rr = [frames[i] for i in sel], [rois[i] for i in sel]
        f = lambda x: -mean_iou(F, Rr, lambda fr: silhouette(cam, meshes, fr["state"], base_pose(cam, *x), shape))
        grid = [(dx, dy, yw) for dx in range(-4, 5, 2) for dy in range(-4, 5, 2) for yw in (-4, 0, 4)]
        x0 = min(grid, key=f)
        r = minimize(f, x0, method="Powell", options={"xtol": 0.1, "ftol": 1e-4, "maxfev": 400})
        return r.x, -r.fun, -f((0, 0, 0))

    allsel = list(range(len(frames)))
    xA, iouA, iou0 = fit_A(allsel)
    print(f"\n(A) base offset on the paper, all frames: dx {xA[0]:+.2f} cm  dy {xA[1]:+.2f} cm  yaw {xA[2]:+.2f} deg   IoU {iou0:.3f} -> {iouA:.3f}")
    z = np.array([f["pinch_z_cm"] for f in frames]); pan = np.array([f["pan"] for f in frames])
    subsets = {"low (pinch < 12 cm above table)": np.where(z < 12)[0], "high (pinch >= 12 cm)": np.where(z >= 12)[0],
               "left (pan > 0)": np.where(pan > 0)[0], "right (pan <= 0)": np.where(pan <= 0)[0]}
    sub_res = {}
    for name, sel in subsets.items():
        if len(sel) < 8:
            print(f"  {name:<34} only {len(sel)} frames, skipped"); continue
        x, i1, i0 = fit_A(list(sel))
        sub_res[name] = {"n": int(len(sel)), "dx_cm": float(x[0]), "dy_cm": float(x[1]), "yaw_deg": float(x[2]), "iou_nominal": i0, "iou_fit": i1}
        print(f"  {name:<34} n={len(sel):3d}  dx {x[0]:+.2f}  dy {x[1]:+.2f}  yaw {x[2]:+.2f}   IoU {i0:.3f} -> {i1:.3f}")

    fB = lambda x: -mean_iou(frames, rois, lambda fr: silhouette(cam.moved(x[:3], x[3:]), meshes, fr["state"], nominal, shape))
    rB = minimize(fB, np.zeros(6), method="Powell", options={"xtol": 0.05, "ftol": 1e-4, "maxfev": 900})
    camB = cam.moved(rB.x[:3], rB.x[3:])
    ang = math.degrees(math.acos(np.clip((np.trace(camB.R.T @ cam.R) - 1) / 2, -1, 1)))
    print(f"\n(B) camera solved from the arm (base at the paper origin): moved {np.round(rB.x[:3], 2)} cm, rotated {ang:.2f} deg"
          f" (rvec {np.round(rB.x[3:], 2)} deg)   IoU {-fB(np.zeros(6)):.3f} -> {-rB.fun:.3f}")

    out_dir = _REPO / "outputs/arm_overlay"; out_dir.mkdir(parents=True, exist_ok=True)
    show = [frames[k] for k in np.linspace(0, len(frames) - 1, 12).astype(int)]
    tiles = []
    for f in show:
        img = cv2.resize(f["img"], None, fx=SCALE, fy=SCALE)
        for col, sil in (((0, 0, 255), silhouette(cam, meshes, f["state"], nominal, shape)),
                         ((0, 255, 0), silhouette(cam, meshes, f["state"], base_pose(cam, *xA), shape))):
            cs, _ = cv2.findContours(sil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cs, -1, col, 1)
        cv2.putText(img, f"ep{f['episode']} f{f['frame']}", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(img)
    png = out_dir / f"{dt.date.today().isoformat()}_normal_A1_arm_overlay.png"
    cv2.imwrite(str(png), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 12, 4)]))

    res = {"what": "arm silhouette (URDF meshes by FK) vs dark pixels in front-left frames of normal_A1",
           "camera": str(EXTR.relative_to(_REPO)), "riser_m": RISER, "n_frames": len(frames),
           "A_base_offset_on_paper": {"dx_cm": float(xA[0]), "dy_cm": float(xA[1]), "yaw_deg": float(xA[2]), "iou_nominal": iou0, "iou_fit": iouA, "subsets": sub_res},
           "B_camera_from_arm": {"move_cm": [float(v) for v in rB.x[:3]], "rvec_deg": [float(v) for v in rB.x[3:]], "rotation_deg": ang,
                                 "iou_nominal": float(-fB(np.zeros(6))), "iou_fit": float(-rB.fun)},
           "joint_mapping_offsets": dict(jm.OFFSET_RAD), "overlay_png": str(png.relative_to(_REPO))}
    out = _REPO / "calibration" / f"{dt.date.today().isoformat()}_arm_overlay_fit.json"
    out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {out.relative_to(_REPO)}, {png.relative_to(_REPO)} (red = arm at the paper origin, green = fitted base offset)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
