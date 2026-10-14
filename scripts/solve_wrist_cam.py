#!/usr/bin/env python
"""Wrist camera mount (link5_from_camera) from the 10-14 marker-session wrist stills, with the touch-pinned arm model.

Why (2026-10-15): the 10-07 wrist extrinsics were solved through joint_mapping's old zeros (incl. a hand-tuned +24 deg
on wrist_flex), and its two poses disagree by 3.6 cm. The link-marker fit, pinned by 16 camera-free touch points
(0.4 cm rms on the mat), now gives link5 in the mat frame for each 10-14 still, and some wrist stills see the paper
markers, so the mount is solved directly in link5.

Paper-marker corners: centre from the points CSV; each marker's own rotation on the mat (they are taped by hand --
ID0 sits ~6 deg off the grid) is measured from the front-left session stills with the fitted session camera.
Unknown: link5_from_camera (6 dof). Residual: the wrist stills' marker corners, ideal-pinhole px (wrist lens undone).

    uv run python scripts/solve_wrist_cam.py --out calibration/2026-10-15_camera_extrinsics_wrist.json
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO))
import camera_distortion as cd  # noqa: E402
import solve_link_markers as SL  # noqa: E402
import touch_test as TT  # noqa: E402

PAPER = (0, 2, 3, 4, 7)


def detector():
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), prm)


def paper_marker_poses(fit: dict, centres: dict, fl_model, fl_stills: list[Path]) -> dict[int, np.ndarray]:
    """world_from_marker for each paper marker: centre from the CSV, in-plane rotation measured on the front-left."""
    K = np.asarray(fl_model.K, float); C = np.array(fit["world_from_camera"])
    det = detector(); yaws = {i: [] for i in PAPER}
    for img in fl_stills:
        corners, ids, _ = det.detectMarkers(cv2.imread(str(img), cv2.IMREAD_GRAYSCALE))
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            if int(i) not in PAPER:
                continue
            ok, rv, tv = cv2.solvePnP(np.c_[SL.CORNERS[:, :2], np.zeros(4)], cd.undistort_pixels(c.reshape(4, 2).astype(float), fl_model),
                                      K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            wm = C[:3, :3] @ cv2.Rodrigues(rv)[0]          # marker axes in the mat frame
            x_axis = wm[:, 0]
            yaws[int(i)].append(np.arctan2(x_axis[1], x_axis[0]))
    out = {}
    for i in PAPER:
        if not yaws[i]:
            continue
        a = float(np.angle(np.mean(np.exp(1j * np.array(yaws[i])))))
        T = np.eye(4); T[:3, :3] = R.from_euler("z", a).as_matrix(); T[:3, 3] = centres[i]
        out[i] = T
        print(f"  paper ID{i}: rotation on the mat {np.degrees(a):+7.2f} deg (from {len(yaws[i])} front-left stills)")
    return out


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fit", default="calibration/2026-10-14_link_marker_fit.json")
    ap.add_argument("--captures", default="calibration/2026-10-14_link_markers_captures.csv")
    ap.add_argument("--points", default="calibration/2026-10-14_aruco_points.csv")
    ap.add_argument("--fl-intrinsics", default="calibration/2026-10-07_camera_intrinsics_front-left.json")
    ap.add_argument("--wrist-intrinsics", default="calibration/2026-10-07_camera_intrinsics_wrist.json")
    ap.add_argument("--compare", default="calibration/2026-10-07_camera_extrinsics_wrist.json,calibration/2026-10-07_camera_extrinsics_wrist_poseB.json")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    fit = json.loads((REPO / args.fit).read_text(encoding="utf-8"))
    model = TT.FitModel(REPO / args.fit)
    centres = {int(r["marker_id"]): np.array([float(r["x_m"]), float(r["y_m"]), float(r["height_above_table_m"])])
               for r in csv.DictReader((REPO / args.points).open(encoding="utf-8"))}
    rows = list(csv.DictReader((REPO / args.captures).open(encoding="utf-8")))
    last = {}
    for r in rows:
        last[(r["pose_id"], r["camera"])] = r
    fl_imgs = [REPO / Path(r["image"].replace("\\", "/")) for (pid, cam), r in last.items() if cam == "front-left"
               and not pid.startswith(("B0", "P00", "P01"))]       # after the 20:16 camera move only
    fl_model = cd.load_model(REPO / args.fl_intrinsics)
    print("paper markers:")
    paper = paper_marker_poses(fit, centres, fl_model, fl_imgs)

    wm = cd.load_model(REPO / args.wrist_intrinsics); Kw = np.asarray(wm.K, float)
    det = detector(); obs = []
    for (pid, cam), r in last.items():
        if cam != "wrist":
            continue
        corners, ids, _ = det.detectMarkers(cv2.imread(str(REPO / Path(r["image"].replace("\\", "/"))), cv2.IMREAD_GRAYSCALE))
        if ids is None:
            continue
        st = np.array([float(r["state_" + j]) for j in TT.JOINTS])
        T5 = model.link5(st)
        for c, i in zip(corners, ids.ravel()):
            if int(i) in paper:
                pw = (paper[int(i)] @ np.c_[SL.CORNERS, np.ones(4)].T).T[:, :3]
                obs.append((pid, int(i), T5, pw, cd.undistort_pixels(c.reshape(4, 2).astype(float), wm)))
    print(f"wrist observations: {len(obs)} markers in {len({o[0] for o in obs})} stills: {sorted({(o[0], o[1]) for o in obs})}")

    def project(X, T5, pw):
        cw = np.linalg.inv(T5 @ X)
        pc = (cw[:3, :3] @ pw.T).T + cw[:3, 3]
        uv = (Kw @ pc.T).T
        return uv[:, :2] / uv[:, 2:3]

    def res(p):
        X = SL.T(p[:3], p[3:])
        return np.concatenate([(project(X, T5, pw) - uv).ravel() for _, _, T5, pw, uv in obs])

    # initial guess: each observation's own PnP, mapped into link5, averaged
    inits = []
    for _, _, T5, pw, uv in obs:
        ok, rv, tv = cv2.solvePnP(pw, uv, Kw, None, flags=cv2.SOLVEPNP_ITERATIVE)
        cfw = np.eye(4); cfw[:3, :3] = cv2.Rodrigues(rv)[0]; cfw[:3, 3] = tv.ravel()
        inits.append(np.linalg.inv(T5) @ np.linalg.inv(cfw))
    rot0 = R.from_matrix(np.array([m[:3, :3] for m in inits])).mean()
    p0 = np.r_[rot0.as_rotvec(), np.median(np.array([m[:3, 3] for m in inits]), 0)]
    sol = least_squares(res, p0, loss="soft_l1", f_scale=2.0)
    X = SL.T(sol.x[:3], sol.x[3:])
    e = np.linalg.norm(sol.fun.reshape(-1, 2), axis=1)
    print(f"\nlink5_from_camera: position {np.round(X[:3, 3] * 100, 2)} cm, reprojection median {np.median(e):.2f} px, max {e.max():.2f} px")
    for k, (pid, i, *_r) in enumerate(obs):
        print(f"   {pid} ID{i}: mean corner error {e[4 * k:4 * k + 4].mean():.2f} px")
    J = sol.jac; s2 = (1.4826 * np.median(np.abs(sol.fun))) ** 2
    sd = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J) * s2), 0, None))
    print(f"   1-sigma: rotation {np.round(np.degrees(sd[:3]), 2)} deg, position {np.round(sd[3:] * 1000, 1)} mm")
    for f in [c for c in args.compare.split(",") if c]:
        d = json.loads((REPO / f).read_text(encoding="utf-8"))
        q = d["quat_wxyz"]; Ro = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
        print(f"   vs {Path(f).name}: position differs {np.linalg.norm(np.array(d['pos_m']) - X[:3, 3]) * 100:.2f} cm, "
              f"rotation {np.degrees(R.from_matrix(Ro.T @ X[:3, :3]).magnitude()):.2f} deg")
    if args.out:
        q = R.from_matrix(X[:3, :3]).as_quat()
        out = {"camera": "wrist", "frame": "link5", "pos_m": X[:3, 3].tolist(), "quat_wxyz": [q[3], q[0], q[1], q[2]],
               "convention": "OpenCV camera axes (x right, y down, z forward), like the 10-07 file", "fit": args.fit,
               "observations": [(o[0], o[1]) for o in obs], "reprojection_median_px": float(np.median(e)), "measured_at": "2026-10-14 stills, solved 2026-10-15"}
        (REPO / args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
