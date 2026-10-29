#!/usr/bin/env python
"""Markerless front-left camera: its 6 DoF from the arm's CAD outline + joint readings, the arm model held fixed.

Why (2026-10-29): the camera gets bumped between sessions; with the arm parameters (base, joint offsets, sag) held at
a link-marker fit, the camera alone is well determined (docs/worklog/2026-10-29_fl_markerless.md §4, with markers).
This drops the markers: the arm itself is the target.

Method (contour tracking, iterated):
  1. at the current camera, render every still's arm (full URDF meshes posed by FK from the joint readings; not
     convex hulls) into a mask; the mesh vertices that land on the mask's OUTER contour are the silhouette points;
  2. each one searches along the contour's outward normal (+-gate px) for the image edge with the arm's polarity
     (dark inside, brighter outside) in the undistorted image;
  3. least squares on the camera's 6 DoF, point-to-line along that normal; gate shrinks 12 -> 4 px over the rounds.
Marker cards on the 10-14 arm (and the paper markers) hide the arm and carry strong edges: their quads, dilated,
are masked out. The markers only say WHERE to mask; they never enter the fit. Paper markers are the check.

    python3 scripts/fit_camera_from_cad.py --arm-from calibration/2026-10-29_fl_markerless/arm_fit_H2.json \
        --holdout 23,4,39,43,38,17,5,15,12 --perturb-camera-init 5,3 --out <json>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "sim"))
import camera_distortion as cd  # noqa: E402
import joint_mapping as jm  # noqa: E402
import solve_link_markers as SL  # noqa: E402

MESH_DIR = Path.home() / "isaaclab_volume/assets/open_manipulator_description/meshes/omx_f"
MESHES = ["follower_01_base", "follower_02_base_tilt_Revised", "follower_03_middle_verticle", "follower_04_middle_horizontal",
          "follower_05_tip", "follower_06_pan_Revised", "follower_07_gripper_motorized", "follower_08_gripper_gear"]  # link0..link7
FINGERS = ((0.0295, 0.0075, 0.0, +1), (0.0295, -0.0108, 0.0, -1))   # omx_f.urdf gripper_joint_1/2 on link5


def load_stl(path: Path):
    data = path.read_bytes()
    n = int(np.frombuffer(data[80:84], np.uint32)[0])
    rec = np.frombuffer(data[84:84 + n * 50], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    v, tri = np.unique(np.round(rec["v"].reshape(-1, 3).astype(np.float64) * 0.001, 5), axis=0, return_inverse=True)
    return v, tri.reshape(-1, 3)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0, 0], [s, c, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1.0]])


def arm_world_points(model, x, state, meshes):
    """All mesh vertices in the world (paper) frame for one still, and the triangle index into them."""
    frames, _ = model.link_frames(x, state)
    base = model.unpack(x)[1]
    q_grip = jm.lerobot_to_urdf_rad(np.asarray(state, float))[5]
    poses = [base] + frames
    for ox, oy, oz, sgn in FINGERS:
        t = np.eye(4); t[:3, 3] = (ox, oy, oz)
        poses.append(frames[4] @ t @ rotz(sgn * q_grip))
    pts, tris, off = [], [], 0
    for P, (v, tri) in zip(poses, meshes):
        pts.append(v @ P[:3, :3].T + P[:3, 3]); tris.append(tri + off); off += len(v)
    return np.concatenate(pts), np.concatenate(tris)


def project(K, cam_from_world, X):
    pc = X @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]
    return np.c_[K[0, 0] * pc[:, 0] / pc[:, 2] + K[0, 2], K[1, 1] * pc[:, 1] / pc[:, 2] + K[1, 2]]


def cam_pose(base_wfc, p):
    """world_from_camera = base * perturbation p (rotvec deg, translation mm, in the camera frame)."""
    d = np.eye(4); d[:3, :3] = R.from_rotvec(np.radians(p[:3])).as_matrix(); d[:3, 3] = np.asarray(p[3:]) / 1000
    return base_wfc @ d


def bilinear(img, u, v):
    h, w = img.shape
    u = np.clip(u, 0, w - 1.001); v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int); a, b = u - u0, v - v0
    return (img[v0, u0] * (1 - a) * (1 - b) + img[v0, u0 + 1] * a * (1 - b)
            + img[v0 + 1, u0] * (1 - a) * b + img[v0 + 1, u0 + 1] * a * b)


class Still:
    def __init__(self, s, model, x, meshes, K, maps, args):
        self.s = s
        self.X, self.tri = arm_world_points(model, x, s["state"], meshes)
        img = cv2.remap(cv2.imread(str(s["image"])), maps[0], maps[1], cv2.INTER_LINEAR)
        g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32), (0, 0), args.blur)
        self.gx, self.gy = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 8, cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 8
        self.img = img
        h, w = g.shape
        self.excl = np.zeros((h, w), np.uint8)       # marker cards (arm) and paper markers: no edges, no points
        for i, c in s["det"].items():
            ctr = c.mean(0); sc = args.card_scale if i in SL.ARM_LINK else args.paper_scale
            cv2.fillConvexPoly(self.excl, np.round(ctr + sc * (c - ctr)).astype(np.int32), 1)
        border = np.zeros((h, w), np.uint8); border[4:-4, 4:-4] = 1
        self.excl[border == 0] = 1

    def silhouette(self, K, cfw, max_pts):
        """Outer-contour vertices of the rendered arm and their outward normals."""
        h, w = self.excl.shape
        uv = np.clip(project(K, cfw, self.X), -1e4, 1e4)
        mask = np.zeros((h, w), np.uint8)
        cv2.fillPoly(mask, np.round(uv[self.tri] * 8).astype(np.int32).reshape(-1, 3, 1, 2), 1, lineType=cv2.LINE_8, shift=3)
        cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        edge = np.zeros((h, w), np.uint8)
        cv2.drawContours(edge, cont, -1, 1, 1)
        ui, vi = np.round(uv[:, 0]).astype(int), np.round(uv[:, 1]).astype(int)
        ok = (ui >= 0) & (ui < w) & (vi >= 0) & (vi < h)
        idx = np.flatnonzero(ok)
        idx = idx[edge[vi[idx], ui[idx]] == 1]
        idx = idx[self.excl[vi[idx], ui[idx]] == 0]
        if len(idx) > max_pts:
            idx = np.random.default_rng(len(idx)).choice(idx, max_pts, replace=False)
        sm = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
        nx, ny = -cv2.Sobel(sm, cv2.CV_32F, 1, 0, ksize=3), -cv2.Sobel(sm, cv2.CV_32F, 0, 1, ksize=3)
        n = np.c_[bilinear(nx, uv[idx, 0], uv[idx, 1]), bilinear(ny, uv[idx, 0], uv[idx, 1])]
        nn = np.linalg.norm(n, axis=1)
        keep = nn > 1e-3
        return idx[keep], uv[idx[keep]], n[keep] / nn[keep, None], mask

    def match(self, p, n, gate, min_grad):
        """Signed offset along n to the arm-polarity edge (brighter outward); NaN if none."""
        ts = np.arange(-gate, gate + 0.125, 0.25)
        U = p[:, 0, None] + ts[None] * n[:, 0, None]; V = p[:, 1, None] + ts[None] * n[:, 1, None]
        resp = bilinear(self.gx, U, V) * n[:, 0, None] + bilinear(self.gy, U, V) * n[:, 1, None]
        h, w = self.excl.shape
        inside = (U >= 0) & (U < w) & (V >= 0) & (V < h)
        Ui, Vi = np.clip(np.round(U).astype(int), 0, w - 1), np.clip(np.round(V).astype(int), 0, h - 1)
        resp[~inside | (self.excl[Vi, Ui] == 1)] = 0
        score = resp * np.exp(-0.5 * (ts[None] / (0.6 * gate)) ** 2)
        k = score.argmax(1)
        t = ts[k].astype(float)
        t[resp[np.arange(len(k)), k] < min_grad] = np.nan
        return t


def solve(stills, K, wfc0, p0, args, log=print):
    p = np.asarray(p0, float)
    gates = np.geomspace(args.gate_start, args.gate_end, args.rounds)
    for r, gate in enumerate(gates):
        cfw = np.linalg.inv(cam_pose(wfc0, p))
        data = []
        for st in stills:
            idx, uv, n, _ = st.silhouette(K, cfw, args.max_pts)
            t = st.match(uv, n, gate, args.min_grad)
            ok = np.isfinite(t)
            data.append((st.X[idx[ok]], uv[ok] + t[ok, None] * n[ok], n[ok]))
        Xs = np.concatenate([d[0] for d in data]); Q = np.concatenate([d[1] for d in data]); N = np.concatenate([d[2] for d in data])

        def res(q):
            return ((project(K, np.linalg.inv(cam_pose(wfc0, q)), Xs) - Q) * N).sum(1)
        r0 = res(p)
        sol = least_squares(res, p, loss="soft_l1", f_scale=args.f_scale, x_scale=[0.2, 0.2, 0.2, 2, 2, 2], diff_step=1e-4)
        step = np.abs(sol.x - p)
        p = sol.x
        log(f"  round {r + 1:2d} gate {gate:5.1f} px: {len(Q):6d} matches, |r| median {np.median(np.abs(r0)):.2f} -> "
            f"{np.median(np.abs(sol.fun)):.2f} px; step {step[:3].max():.3f} deg {step[3:].max():.2f} mm")
    return p


def overlay_residual(stills, K, wfc, args):
    """Median |offset| from the rendered outline to the matched edge at a fixed small gate: the 'how well does it overlay' number."""
    cfw = np.linalg.inv(wfc); out, found = [], 0
    for st in stills:
        idx, uv, n, _ = st.silhouette(K, cfw, args.max_pts)
        t = st.match(uv, n, args.gate_eval, args.min_grad)
        found += np.isfinite(t).sum(); out.append(t[np.isfinite(t)])
        if not len(idx):
            continue
    a = np.abs(np.concatenate(out)) if out else np.zeros(0)
    return float(np.median(a)), float(np.percentile(a, 90)), int(found)


def paper_check(stills, K, wfc, paper_w):
    e = [np.linalg.norm(SL.project(K, wfc, paper_w[i][None])[0] - s["det"][i].mean(0))
         for s in stills for i in SL.PAPER_IDS if i in s["det"]]
    return float(np.median(e)), float(np.percentile(e, 90)), float(np.max(e)), len(e)


def pose_diff(A, B):
    D = np.linalg.inv(A) @ B
    return float(np.linalg.norm(D[:3, 3]) * 1000), float(np.degrees(np.linalg.norm(R.from_matrix(D[:3, :3]).as_rotvec())))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm-from", required=True, help="solve_link_markers --out JSON: base, offsets, sag (and its camera = the start)")
    ap.add_argument("--compare", default="", help="other camera JSONs (solve_link_markers --out) to report the difference to, comma-separated")
    ap.add_argument("--captures", default="calibration/2026-10-14_link_markers_captures.csv")
    ap.add_argument("--excluded", default="calibration/2026-10-14_link_markers_excluded.csv")
    ap.add_argument("--plan", default="configs/link_marker_plan_farwrist.json")
    ap.add_argument("--points", default="calibration/2026-10-14_aruco_points.csv")
    ap.add_argument("--intrinsics", default="calibration/2026-10-07_camera_intrinsics_front-left.json")
    ap.add_argument("--holdout", default="", help="episodes NOT used here (e.g. the arm fit's own episodes)")
    ap.add_argument("--max-elbow-reading", type=float, default=51.0)
    ap.add_argument("--max-stills", type=int, default=0)
    ap.add_argument("--perturb-camera-init", default="", help="deg,cm away from the --arm-from camera")
    ap.add_argument("--perturb-seed", type=int, default=0)
    ap.add_argument("--rounds", type=int, default=12)
    ap.add_argument("--gate-start", type=float, default=12.0)
    ap.add_argument("--gate-end", type=float, default=4.0)
    ap.add_argument("--gate-eval", type=float, default=6.0)
    ap.add_argument("--min-grad", type=float, default=6.0, help="grey levels per px along the normal")
    ap.add_argument("--blur", type=float, default=1.0)
    ap.add_argument("--f-scale", type=float, default=1.5)
    ap.add_argument("--max-pts", type=int, default=400, help="silhouette points per still")
    ap.add_argument("--card-scale", type=float, default=1.9, help="arm marker card mask, x the marker's black square")
    ap.add_argument("--paper-scale", type=float, default=1.7)
    ap.add_argument("--split-check", action="store_true", help="also fit each half of the stills (by episode) and score the other half")
    ap.add_argument("--overlays", default="", help="directory: write a few outline overlays (start and fitted)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    prev = json.loads((REPO / args.arm_from).read_text(encoding="utf-8"))
    cm = cd.load_model(REPO / args.intrinsics); K = np.asarray(cm.K, float)
    paper_w = {int(r["marker_id"]): np.array([float(r["x_m"]), float(r["y_m"]), float(r["height_above_table_m"])])
               for r in SL.csv.DictReader((REPO / args.points).open(encoding="utf-8"))}
    stills = SL.load_stills(REPO / args.captures, REPO / args.excluded, REPO / args.plan)
    SL.detect(stills, cm, "apriltag")
    stills = [s for s in stills if s["state"][2] <= args.max_elbow_reading]
    hold = {int(e) for e in args.holdout.split(",") if e}
    stills = [s for s in stills if s["ep"] not in hold]
    if args.max_stills and args.max_stills < len(stills):
        pick = np.random.default_rng(args.perturb_seed + 1000).choice(len(stills), args.max_stills, replace=False)
        stills = [stills[k] for k in sorted(pick)]

    off0 = np.array([prev["offset_rad"][n] for n in jm.LEROBOT_NAMES[:5]])
    model = SL.Model(sum(n.startswith("id1m") and n.endswith("_r0") for n in prev["names"]), False, SL.urdf_inertials(), off0)
    if model.names != prev["names"]:
        print(f"🔴 {args.arm_from}: parameter list differs from this model"); return 1
    model.gravity_in_base = prev["args"].get("gravity_in_base", False)
    x = np.array(prev["x"])
    wfc_ref = np.array(prev["world_from_camera"])
    meshes = [load_stl(MESH_DIR / f"{m}.stl") for m in MESHES]
    us, vs = np.meshgrid(np.arange(cm.width, dtype=np.float64), np.arange(cm.height, dtype=np.float64))
    src = cd.distort_pixels(np.stack([us.ravel(), vs.ravel()], 1), cm).reshape(cm.height, cm.width, 2).astype(np.float32)
    maps = (src[..., 0], src[..., 1])
    S = [Still(s, model, x, meshes, K, maps, args) for s in stills]
    print(f"{len(S)} stills (episodes held out here: {sorted(hold) or 'none'}); arm from {args.arm_from}")

    p0 = np.zeros(6)
    if args.perturb_camera_init:
        d_deg, d_cm = (float(v) for v in args.perturb_camera_init.split(","))
        rng = np.random.default_rng(args.perturb_seed)
        a, b = rng.normal(size=3), rng.normal(size=3)
        p0 = np.r_[d_deg * a / np.linalg.norm(a), d_cm * 10 * b / np.linalg.norm(b)]
        print(f"camera start perturbed by {d_deg} deg, {d_cm} cm (seed {args.perturb_seed})")
    w_start = cam_pose(wfc_ref, p0)
    print("start:  overlay |offset| median {:.2f} p90 {:.2f} px ({} matches); paper {:.2f} px".format(
        *overlay_residual(S, K, w_start, args), paper_check(stills, K, w_start, paper_w)[0]))
    p = solve(S, K, wfc_ref, p0, args)
    wfc = cam_pose(wfc_ref, p)
    ov = overlay_residual(S, K, wfc, args)
    pc = paper_check(stills, K, wfc, paper_w)
    print(f"fitted: overlay |offset| median {ov[0]:.2f} p90 {ov[1]:.2f} px ({ov[2]} matches)")
    print(f"paper-marker centres, INDEPENDENT: median {pc[0]:.2f} px, p90 {pc[1]:.2f}, max {pc[2]:.2f} over {pc[3]} detections")
    dref = pose_diff(wfc_ref, wfc)
    print(f"camera vs the --arm-from camera (paper-fitted on its episodes): {dref[0]:.1f} mm, {dref[1]:.2f} deg")
    report = {"overlay_px": {"median": ov[0], "p90": ov[1], "n": ov[2]}, "paper_px": {"median": pc[0], "p90": pc[1], "max": pc[2], "n": pc[3]},
              "vs_arm_from": {"mm": dref[0], "deg": dref[1]}}
    for f in [f for f in args.compare.split(",") if f]:
        d = pose_diff(np.array(json.loads((REPO / f).read_text(encoding="utf-8"))["world_from_camera"]), wfc)
        print(f"camera vs {f}: {d[0]:.1f} mm, {d[1]:.2f} deg"); report[f"vs:{f}"] = {"mm": d[0], "deg": d[1]}

    if args.split_check:
        eps = sorted({st.s["ep"] for st in S if st.s["ep"] is not None})
        halves = (set(eps[0::2]), set(eps[1::2]))
        sols = []
        for k, hs in enumerate(halves):
            fit = [st for st in S if st.s["ep"] in hs]; test = [st for st in S if st.s["ep"] not in hs]
            q = solve(fit, K, wfc_ref, p0, args, log=lambda *_: None)
            w = cam_pose(wfc_ref, q); sols.append(w)
            ovt = overlay_residual(test, K, w, args)
            print(f"half {k} (episodes {sorted(hs)}, {len(fit)} stills): held-out overlay median {ovt[0]:.2f} p90 {ovt[1]:.2f} px; "
                  f"paper {paper_check([st.s for st in test], K, w, paper_w)[0]:.2f} px")
            report[f"half{k}"] = {"episodes": sorted(hs), "held_out_overlay_median": ovt[0], "held_out_overlay_p90": ovt[1]}
        dh = pose_diff(*sols)
        print(f"half 0 vs half 1 camera: {dh[0]:.1f} mm, {dh[1]:.2f} deg"); report["halves"] = {"mm": dh[0], "deg": dh[1]}

    if args.overlays:
        od = Path(args.overlays); od.mkdir(parents=True, exist_ok=True)
        for st in S[:: max(1, len(S) // 6)][:6]:
            im = st.img.copy()
            for wc, col in ((w_start, (0, 0, 255)), (wfc, (0, 255, 0))):
                _, _, _, mask = st.silhouette(K, np.linalg.inv(wc), args.max_pts)
                cont, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(im, cont, -1, col, 1)
            im[st.excl == 1] = (im[st.excl == 1] * 0.5).astype(np.uint8)
            cv2.imwrite(str(od / f"{st.s['pose_id']}.png"), im)
        print(f"overlays (red = start, green = fitted, dimmed = masked, undistorted image) in {od}")

    if args.out:
        res = {"what": "scripts/fit_camera_from_cad.py", "args": vars(args), "world_from_camera": wfc.tolist(),
               "perturbation_from_arm_from": p.tolist(), "report": report}
        (REPO / args.out if not Path(args.out).is_absolute() else Path(args.out)).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
