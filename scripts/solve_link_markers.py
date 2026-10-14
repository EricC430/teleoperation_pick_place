#!/usr/bin/env python
"""Solve the arm's kinematic constants from the 10-14 link-marker stills (scripts/capture_link_markers.py).

What is measured: front-left stills of ArUco markers taped to the upper arm (ID1, link2), forearm (ID5, link3) and
gripper servo body (ID6, link5), with the joint readings averaged at each still, plus the paper markers (0, 2, 3, 4, 7)
whose centres are known on the placement mat.

Model (all in the mat frame: origin on the pan axis at the table, +x ahead, +y left, +z up):
  camera       6 dof, fitted from the paper-marker centres in EVERY still (the camera did not move: <= 1.3 px)
  base         6 dof: link0 in the mat frame (pan-axis offset, riser height, tilt). Base yaw absorbs the pan offset,
               so the pan offset stays at its joint_mapping value.
  offsets      shoulder_lift, elbow_flex, wrist_flex (rad). wrist_roll's offset is absorbed by ID6's mount (ID6 is on
               the link after roll) and stays fixed. Scale is the encoder's 1.7996 deg/unit, exact.
  sag          link angle = encoder angle + c_i * gravity torque on joint i (lift, elbow, flex), URDF masses: play
               between the encoder and the link that the readings cannot see (S7: 2-6 deg, gravity dependent).
  mounts       each arm marker's pose in its link, 6 dof. ID1 gets one mount per taping (re-taped after S24 and
               after S39 [柏宇說]); the spread between them is reported.
  lengths      optional (--free-lengths): a scale on the upper arm (joint2->3 vector, direction kept: its angle is the
               lift offset) and on the forearm (joint3->4). joint5's x is not identifiable (ID6's mount absorbs it).
               First free fit (10-14, every origin free) put both links ~4 mm (3.4 %) longer than the URDF.
  marker size  optional (--free-marker-size): one scale on the 60 mm of the three ARM markers. 10-14: with free link
               scales the fit said upper arm 124.4 / forearm 165.7 mm, but a ruler (axis to axis, [柏宇說]) reads
               120.5 / 163 mm -- URDF 120.5 / 162. A marker smaller than assumed makes the whole arm look bigger.
  mat scale    optional (--free-mat-scale): true metres per mat metre. The paper centres are in mat units (a printed
               grid), the arm-marker corners in true metres (60 mm), so a printer-scaled mat shows up here.
Residuals: the 4 corners of each arm marker (60 mm [柏宇說 2026-10-14: all markers are 6 cm]) and the centre of each
paper marker, in ideal-pinhole pixels (detections undistorted with the camera's own model), soft-L1 loss.

    uv run python scripts/solve_link_markers.py --holdout 14,38
    uv run python scripts/solve_link_markers.py --holdout 14,38 --free-lengths

Held-out episodes are fitted with nothing of theirs and scored afterwards: that, not the training residual, is how far
to trust the constants. Output: calibration/2026-10-14_link_marker_fit.json (--out).
"""
from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "sim"))
sys.path.insert(0, str(REPO))
import camera_distortion as cd  # noqa: E402
import joint_mapping as jm  # noqa: E402
from reach_logger import fk  # noqa: E402

MARKER_M = 0.060
PAPER_IDS = (0, 2, 3, 4, 7)
ARM_LINK = {1: 2, 5: 3, 6: 5}          # marker id -> link index (fk.link_transform)
SAG_JOINTS = (1, 2, 3)                 # chain index of shoulder_lift, elbow_flex, wrist_flex
G = 9.81


# ------------------------------------------------------------------------------------------------ data
def load_stills(captures: Path, excluded: Path, plan: Path):
    rows = [r for r in csv.DictReader(captures.open(encoding="utf-8")) if r["camera"] == "front-left"]
    last = {}
    for r in rows:                     # last row per pose_id wins (retakes)
        last[r["pose_id"]] = r
    excl = list(csv.DictReader(excluded.open(encoding="utf-8"))) if excluded.exists() else []
    src = {p["id"]: int(p["source"].split()[0][2:]) for p in json.loads(plan.read_text(encoding="utf-8"))["poses"]}
    order = {r["pose_id"]: n for n, r in enumerate(rows)}
    stills = []
    for pid, r in last.items():
        img = REPO / PureWindowsPath(r["image"]).as_posix()
        drop = {e["marker"] for e in excl if fnmatch.fnmatch(img.stem, e["image_glob"])}
        if "all" in drop:
            continue
        m = re.match(r"[ES](\d+)_", pid)
        ep = int(m.group(1)) if m else src.get(pid)
        stills.append({"pose_id": pid, "image": img, "ep": ep, "order": order[pid],
                       "state": np.array([float(r["state_" + j]) for j in jm.LEROBOT_NAMES]),
                       "drop_markers": {int(d) for d in drop}})
    stills.sort(key=lambda s: s["order"])
    return stills


def detect(stills, model, refine="none"):
    prm = cv2.aruco.DetectorParameters()
    prm.cornerRefinementMethod = {"none": cv2.aruco.CORNER_REFINE_NONE, "subpix": cv2.aruco.CORNER_REFINE_SUBPIX,
                                  "contour": cv2.aruco.CORNER_REFINE_CONTOUR, "apriltag": cv2.aruco.CORNER_REFINE_APRILTAG}[refine]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50), prm)
    for s in stills:
        corners, ids, _ = det.detectMarkers(cv2.imread(str(s["image"]), cv2.IMREAD_GRAYSCALE))
        s["det"] = {}
        if ids is None:
            continue
        for c, i in zip(corners, ids.ravel()):
            i = int(i)
            if (i in PAPER_IDS or i in ARM_LINK) and i not in s["drop_markers"]:
                s["det"][i] = cd.undistort_pixels(c.reshape(4, 2).astype(float), model)


def id1_segments(stills):
    """Mount index for ID1: re-taped after the S24 scan and after the S39 scan [柏宇說]."""
    seg, cur = {}, 0
    for s in stills:
        if s["pose_id"].startswith("S30_") and cur == 0:
            cur = 1
        if s["pose_id"].startswith("S43_") and cur == 1:
            cur = 2
        seg[s["pose_id"]] = cur
    return seg


def urdf_inertials():
    root = ET.parse(REPO / "assets/omx_f/omx_f.urdf").getroot()
    inert = {}
    for link in root.findall("link"):
        m = link.find("inertial/mass")
        if m is not None:
            inert[link.get("name")] = (float(m.get("value")), np.array([float(v) for v in link.find("inertial/origin").get("xyz").split()]))
    # fingers ride on link5 at their pivots (gripper rotation ignored: ~1% of the wrist torque)
    m5, c5 = inert["link5"]
    mf = inert["link6"][0] + inert["link7"][0]
    c6 = np.array([0.0295, 0.0075, 0.0]) + inert["link6"][1]
    c7 = np.array([0.0295, -0.0108, 0.0]) + inert["link7"][1]
    inert["link5"] = (m5 + mf, (m5 * c5 + inert["link6"][0] * c6 + inert["link7"][0] * c7) / (m5 + mf))
    return [inert[f"link{k}"] for k in range(1, 6)]  # links 1..5


# ------------------------------------------------------------------------------------------------ model
def T(rotvec, t):
    out = np.eye(4)
    out[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    out[:3, 3] = t
    return out


def chain_transforms(q, chain):
    """Frames of link1..link5 in the base frame, like fk.link_transform, with an editable chain."""
    out, t = [], np.eye(4)
    for (origin, axis), a in zip(chain, q):
        t = t @ fk._translation(origin) @ fk._rotation(axis, a)
        out.append(t)
    return out


def gravity_torques(frames, inertials, g_base):
    """Gravity torque on joints 2, 3, 4 (about their +Y), in the base frame."""
    coms = [f[:3, :3] @ c + f[:3, 3] for f, (_, c) in zip(frames, inertials)]
    taus = []
    for j in SAG_JOINTS:                       # frames[j] is link j+1, whose origin sits on joint j+1
        p, ax = frames[j][:3, 3], frames[j][:3, 1]
        tau = sum(np.dot(np.cross(coms[k] - p, m * g_base), ax) for k, (m, _) in enumerate(inertials) if k >= j)
        taus.append(tau)
    return np.array(taus)


class Model:
    def __init__(self, n_id1, free_lengths, inertials, base_q_offset):
        self.free_lengths = free_lengths
        self.inertials = inertials
        self.q_off_fixed = base_q_offset.copy()
        self.base_chain = [tuple(c) for c in fk._CHAIN]
        names = ["cam_r0", "cam_r1", "cam_r2", "cam_x", "cam_y", "cam_z",
                 "base_r0", "base_r1", "base_r2", "base_x", "base_y", "base_z",
                 "off_lift", "off_elbow", "off_flex", "sag_lift", "sag_elbow", "sag_flex"]
        for k in range(n_id1):
            names += [f"id1m{k}_{a}" for a in ("r0", "r1", "r2", "x", "y", "z")]
        for mid in (5, 6):
            names += [f"id{mid}_{a}" for a in ("r0", "r1", "r2", "x", "y", "z")]
        if free_lengths:
            names += ["scale_upper", "scale_fore"]
        names += ["mat_scale", "marker_scale", "focal_scale", "jscale_lift", "jscale_elbow", "jscale_flex"]
        self.names = names
        self.idx = {n: i for i, n in enumerate(names)}
        self.n_id1 = n_id1

    def unpack(self, x):
        g = lambda n: x[self.idx[n]]  # noqa: E731
        cam = T(x[0:3], x[3:6])                         # world_from_camera
        base = T(x[6:9], x[9:12])                       # world_from_link0
        off = self.q_off_fixed.copy()
        off[1:4] = [g("off_lift"), g("off_elbow"), g("off_flex")]
        sag = np.array([g("sag_lift"), g("sag_elbow"), g("sag_flex")])
        mounts = {("id1", k): T(x[self.idx[f"id1m{k}_r0"]:self.idx[f"id1m{k}_r0"] + 3], x[self.idx[f"id1m{k}_x"]:self.idx[f"id1m{k}_x"] + 3]) for k in range(self.n_id1)}
        for mid in (5, 6):
            mounts[(f"id{mid}", 0)] = T(x[self.idx[f"id{mid}_r0"]:self.idx[f"id{mid}_r0"] + 3], x[self.idx[f"id{mid}_x"]:self.idx[f"id{mid}_x"] + 3])
        chain = [list(c) for c in self.base_chain]
        if self.free_lengths:
            chain[2][0] = tuple(g("scale_upper") * v for v in self.base_chain[2][0])
            chain[3][0] = tuple(g("scale_fore") * v for v in self.base_chain[3][0])
        return cam, base, off, sag, mounts, chain

    def link_frames(self, x, state, cache=None):
        cam, base, off, sag, mounts, chain = self.unpack(x)
        js = np.ones(5)
        js[1:4] = x[self.idx["jscale_lift"]:self.idx["jscale_flex"] + 1]
        q = jm.SCALE_RAD_PER_UNIT["shoulder_lift"] * js * state[:5] + off
        frames0 = chain_transforms(q, chain)
        g_base = np.array([0.0, 0.0, -G]) if getattr(self, "gravity_in_base", False) else base[:3, :3].T @ np.array([0.0, 0.0, -G])
        tau = gravity_torques(frames0, self.inertials, g_base)
        q2 = q.copy()
        q2[1:4] += sag * tau
        return [base @ f for f in chain_transforms(q2, chain)], q2


CORNERS = np.array([[-1, 1, 0], [1, 1, 0], [1, -1, 0], [-1, -1, 0]], float) * MARKER_M / 2  # ArUco corner order


def project(Kmat, world_from_cam, pts_w):
    cam_from_world = np.linalg.inv(world_from_cam)
    pc = (cam_from_world[:3, :3] @ pts_w.T).T + cam_from_world[:3, 3]
    uv = (Kmat @ pc.T).T
    return uv[:, :2] / uv[:, 2:3]


def residuals(x, model, stills, Kmat, paper_w, seg, which=None):
    cam, base, off, sag, mounts, chain = model.unpack(x)
    Kmat = Kmat.copy()
    Kmat[0, 0] *= x[model.idx["focal_scale"]]
    Kmat[1, 1] *= x[model.idx["focal_scale"]]
    res = []
    for s in stills:
        d = s["det"]
        pp = [i for i in PAPER_IDS if i in d]
        if pp:
            ms = x[model.idx["mat_scale"]]
            res.append((project(Kmat, cam, ms * np.array([paper_w[i] for i in pp])) - np.array([d[i].mean(0) for i in pp])).ravel())
        arm = [i for i in ARM_LINK if i in d]
        if not arm:
            continue
        frames, _ = model.link_frames(x, s["state"])
        for i in arm:
            if which is not None and i not in which:
                continue
            key = ("id1", seg[s["pose_id"]]) if i == 1 else (f"id{i}", 0)
            mw = frames[ARM_LINK[i] - 1] @ mounts[key]
            pw = (mw[:3, :3] @ (x[model.idx["marker_scale"]] * CORNERS).T).T + mw[:3, 3]
            res.append((project(Kmat, cam, pw) - d[i]).ravel())
    if which is None and getattr(model, "touch", None):
        res.append(touch_residuals(x, model, model.touch) * model.touch_weight / TOUCH_SIGMA_MM)
    return np.concatenate(res) if res else np.zeros(0)


TOUCH_SIGMA_MM = 2.5     # reading a 1 mm grid by eye from above, plumb settled
TIP_H = np.array([0.0945, -0.00165, 0.0, 1.0])   # omx_constants GRIPPER_TIP_X_M / MIDLINE_Y_M (closed)


def touch_point(x, model, readings, plumb):
    frames, _ = model.link_frames(x, readings)
    t = (frames[-1] @ TIP_H)[:3]
    if plumb:
        base = model.unpack(x)[1]
        g = base[:3, :3] @ np.array([0.0, 0.0, -1.0]) if getattr(model, "gravity_in_base", False) else np.array([0.0, 0.0, -1.0])
        t = t + (t[2] / -g[2]) * g
    return t[:2]


def touch_residuals(x, model, rows):
    """mm, mat frame: where the model puts the closed fingertip (or the plumb weight) minus where it was read."""
    return np.concatenate([(touch_point(x, model, rd, pl) - xy) * 1000 for rd, xy, pl in rows])


def load_touch(paths_plumb):
    rows = []
    for path, plumb in paths_plumb:
        for r in csv.DictReader(open(path, encoding="utf-8")):
            rows.append((np.array([float(r["state_" + j]) for j in jm.LEROBOT_NAMES]),
                         np.array([float(r["measured_x_cm"]), float(r["measured_y_cm"])]) / 100, plumb))
    return rows


# ------------------------------------------------------------------------------------------------ init
def init_camera(stills, Kmat, paper_w):
    obj, img = [], []
    s = max(stills, key=lambda s: sum(i in s["det"] for i in PAPER_IDS))
    for i in PAPER_IDS:
        if i in s["det"]:
            obj.append(paper_w[i]); img.append(s["det"][i].mean(0))
    ok, rv, tv = cv2.solvePnP(np.array(obj), np.array(img), Kmat, None, flags=cv2.SOLVEPNP_ITERATIVE)
    Rc = cv2.Rodrigues(rv)[0]
    wfc = np.eye(4); wfc[:3, :3] = Rc.T; wfc[:3, 3] = -Rc.T @ tv.ravel()
    return wfc


def init_mounts(stills, model, x, Kmat, seg):
    cam = model.unpack(x)[0]
    obj = CORNERS[:, :2].copy()
    acc = {}
    for s in stills:
        frames, _ = model.link_frames(x, s["state"])
        for i in ARM_LINK:
            if i not in s["det"]:
                continue
            ok, rv, tv = cv2.solvePnP(np.c_[obj, np.zeros(4)], s["det"][i], Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            cfm = T(rv.ravel(), tv.ravel())
            link_from_marker = np.linalg.inv(frames[ARM_LINK[i] - 1]) @ cam @ cfm
            key = ("id1", seg[s["pose_id"]]) if i == 1 else (f"id{i}", 0)
            acc.setdefault(key, []).append(link_from_marker)
    out = {}
    for key, ms in acc.items():
        rot = R.from_matrix(np.array([m[:3, :3] for m in ms])).mean()
        out[key] = (rot.as_rotvec(), np.median(np.array([m[:3, 3] for m in ms]), axis=0), len(ms))
    return out


# ------------------------------------------------------------------------------------------------ main
def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--captures", default="calibration/2026-10-14_link_markers_captures.csv")
    ap.add_argument("--excluded", default="calibration/2026-10-14_link_markers_excluded.csv")
    ap.add_argument("--plan", default="configs/link_marker_plan_farwrist.json")
    ap.add_argument("--points", default="calibration/2026-10-14_aruco_points.csv")
    ap.add_argument("--intrinsics", default="calibration/2026-10-07_camera_intrinsics_front-left.json")
    ap.add_argument("--holdout", default="14,38", help="episodes kept out of the fit and scored afterwards")
    ap.add_argument("--start", choices=("s7", "current"), default="s7",
                    help="initial joint offsets: S7 (calibration/2026-10-10_link_tilt_fit.json) or joint_mapping.py")
    ap.add_argument("--free-lengths", action="store_true")
    ap.add_argument("--free-mat-scale", action="store_true")
    ap.add_argument("--free-marker-size", action="store_true")
    ap.add_argument("--free-focal", action="store_true", help="one scale on fx, fy (device-read intrinsics)")
    ap.add_argument("--refine", choices=("none", "subpix", "contour", "apriltag"), default="none",
                    help="ArUco corner refinement; without it the corners of a ~50 px marker can sit a fraction of a pixel inside")
    ap.add_argument("--base-tilt-deg", default=None,
                    help="roll,pitch of the base relative to the mat, MEASURED (phone level) and held fixed. [柏宇說 10-15] "
                         "base y -0.45, desk -2.85 (near) / -2.65 (far): base is ~2.3 deg off the mat plane")
    ap.add_argument("--gravity-in-base", action="store_true",
                    help="gravity straight down in the base frame (base level within 0.45 deg [柏宇說]); the mat itself is "
                         "tilted ~2.75 deg, so 'down' in the mat frame is not gravity")
    ap.add_argument("--fix-base-tilt", action="store_true",
                    help="base level with the mat (yaw only): base pitch and the lift offset trade off (corr -0.5); "
                         "10-13 level check: base within ~0.5 deg of level [柏宇說]")
    ap.add_argument("--touch-tip", default="", help="touch_test.py CSV, fingertip read directly (10-15)")
    ap.add_argument("--touch-plumb", default="", help="touch_test.py --plumb CSV(s), comma-separated")
    ap.add_argument("--touch-weight", type=float, default=0.0,
                    help="weight of each touch residual (mm / 2.5) against marker pixels; 0 = report only")
    ap.add_argument("--touch-holdout", choices=("none", "even", "odd"), default="none",
                    help="keep every other touch point out of the fit and score it afterwards")
    ap.add_argument("--max-elbow-reading", type=float, default=None,
                    help="drop stills with the elbow reading above this. [柏宇說 10-15] at HOME the forearm rests on the "
                         "shoulder-lift servo: past ~52.8 the link stops while the reading goes on (photos: -2..-5 deg at "
                         "54-55; sim with self-collision stops 2.6 deg short at B0)")
    ap.add_argument("--free-joint-scales", action="store_true",
                    help="effective deg/unit for lift, elbow, flex (the encoder's is exact; play can make it look otherwise, S7)")
    ap.add_argument("--fore-mm", type=float, default=None, help="forearm joint3->4 length to use instead of the URDF's")
    ap.add_argument("--no-sag", action="store_true")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model_cam = cd.load_model(REPO / args.intrinsics)
    Kmat = np.asarray(model_cam.K, float)
    paper_w = {int(r["marker_id"]): np.array([float(r["x_m"]), float(r["y_m"]), float(r["height_above_table_m"])])
               for r in csv.DictReader((REPO / args.points).open(encoding="utf-8"))}
    stills = load_stills(REPO / args.captures, REPO / args.excluded, REPO / args.plan)
    detect(stills, model_cam, args.refine)
    seg = id1_segments(stills)
    if args.max_elbow_reading is not None:
        n0 = len(stills)
        stills = [s for s in stills if s["state"][2] <= args.max_elbow_reading]
        print(f"elbow reading <= {args.max_elbow_reading}: {n0 - len(stills)} stills dropped (forearm on the shoulder servo)")
    hold = {int(e) for e in args.holdout.split(",") if e}
    train = [s for s in stills if s["ep"] not in hold]
    test = [s for s in stills if s["ep"] in hold]
    count = lambda ss, i: sum(i in s["det"] for s in ss)  # noqa: E731
    print(f"stills: {len(stills)} usable ({len(train)} fit / {len(test)} held out: episodes {sorted(hold)})")
    for i in (1, 5, 6):
        print(f"  ID{i}: {count(train, i)} fit / {count(test, i)} held out")

    if args.start == "s7":
        s7 = json.loads((REPO / "calibration/2026-10-10_link_tilt_fit.json").read_text(encoding="utf-8"))["S7 offsets only"]["offset"]
        off0 = np.array([s7[n] for n in jm.LEROBOT_NAMES[:5]])
    else:
        off0 = np.array([jm.OFFSET_RAD[n] for n in jm.LEROBOT_NAMES[:5]])
    off0[0] = jm.OFFSET_RAD["shoulder_pan"]
    model = Model(3, args.free_lengths, urdf_inertials(), off0)
    if args.fore_mm:
        model.base_chain[3] = ((args.fore_mm / 1000.0, 0.0, 0.0), fk._CHAIN[3][1])
    x = np.zeros(len(model.names))
    wfc = init_camera(stills, Kmat, paper_w)
    x[0:3] = R.from_matrix(wfc[:3, :3]).as_rotvec(); x[3:6] = wfc[:3, 3]
    x[9:12] = [-fk._PAN_AXIS_XY[0], 0.0, 0.146]        # pan axis on the mat origin, riser 14.6 cm
    x[model.idx["off_lift"]:model.idx["off_flex"] + 1] = off0[1:4]
    if args.free_lengths:
        x[model.idx["scale_upper"]] = x[model.idx["scale_fore"]] = 1.0
    x[model.idx["mat_scale"]] = x[model.idx["marker_scale"]] = x[model.idx["focal_scale"]] = 1.0
    x[model.idx["jscale_lift"]] = x[model.idx["jscale_elbow"]] = x[model.idx["jscale_flex"]] = 1.0
    inits = init_mounts(train, model, x, Kmat, seg)
    for key in [("id1", 0), ("id1", 1), ("id1", 2), ("id5", 0), ("id6", 0)]:
        if key not in inits:
            print(f"🔴 no fit-set detection for mount {key}: change --holdout"); return 1
        print(f"  mount {key}: initialised from {inits[key][2]} detections")
    for key, (rv, t, n) in inits.items():
        pre = f"id1m{key[1]}" if key[0] == "id1" else key[0]
        x[model.idx[f"{pre}_r0"]:model.idx[f"{pre}_r0"] + 3] = rv
        x[model.idx[f"{pre}_x"]:model.idx[f"{pre}_x"] + 3] = t

    lo = np.full(len(x), -np.inf); hi = np.full(len(x), np.inf)
    if args.base_tilt_deg:
        r_, p_ = (np.radians(float(v)) for v in args.base_tilt_deg.split(","))
        for n, v in (("base_r0", r_), ("base_r1", p_)):
            lo[model.idx[n]], hi[model.idx[n]] = v - 1e-12, v + 1e-12
            x[model.idx[n]] = v
    model.gravity_in_base = args.gravity_in_base
    if args.fix_base_tilt:
        for n in ("base_r0", "base_r1"):
            lo[model.idx[n]], hi[model.idx[n]] = -1e-12, 1e-12
            x[model.idx[n]] = 0.0
    if not args.free_joint_scales:
        for n in ("jscale_lift", "jscale_elbow", "jscale_flex"):
            lo[model.idx[n]], hi[model.idx[n]] = 1.0 - 1e-12, 1.0 + 1e-12
    if not args.free_focal:
        lo[model.idx["focal_scale"]], hi[model.idx["focal_scale"]] = 1.0 - 1e-12, 1.0 + 1e-12
    if not args.free_marker_size:
        lo[model.idx["marker_scale"]], hi[model.idx["marker_scale"]] = 1.0 - 1e-12, 1.0 + 1e-12
    if not args.free_mat_scale:
        lo[model.idx["mat_scale"]], hi[model.idx["mat_scale"]] = 1.0 - 1e-12, 1.0 + 1e-12
    if args.no_sag:
        for n in ("sag_lift", "sag_elbow", "sag_flex"):
            lo[model.idx[n]], hi[model.idx[n]] = -1e-12, 1e-12
    touch = load_touch([(REPO / args.touch_tip, False)] * bool(args.touch_tip)
                       + [(REPO / f, True) for f in args.touch_plumb.split(",") if f])
    t_fit = [t for k, t in enumerate(touch) if args.touch_holdout == "none" or (k % 2 == 1) == (args.touch_holdout == "even")]
    t_out = [t for t in touch if not any(t is u for u in t_fit)]
    model.touch = t_fit if args.touch_weight > 0 else []
    model.touch_weight = args.touch_weight
    def touch_report(tag):
        for name, rows in (("fit", t_fit if model.touch else []), ("held out" if model.touch else "all (not fitted)", t_out if model.touch else touch)):
            if rows:
                e = touch_residuals(x, model, rows).reshape(-1, 2) / 10
                print(f"  touch {tag} {name:16s}: n={len(rows)}  rms {np.sqrt((e ** 2).sum(1).mean()):.2f} cm  mean (model-read) {np.round(e.mean(0), 2)} cm")
    if touch:
        touch_report("before")
    r0 = residuals(x, model, train, Kmat, paper_w, seg)
    print(f"initial: median |r| {np.median(np.abs(r0)):.2f} px over {r0.size} residuals")
    sol = least_squares(residuals, x, args=(model, train, Kmat, paper_w, seg), loss="soft_l1", f_scale=1.5,
                        bounds=(lo, hi), x_scale="jac", max_nfev=200, verbose=0)
    x = sol.x
    if touch:
        touch_report("after ")
    rt = residuals(x, model, train, Kmat, paper_w, seg)
    print(f"fit:     median |r| {np.median(np.abs(rt)):.2f} px, p95 {np.percentile(np.abs(rt), 95):.2f} px  ({sol.message})")

    # covariance from the Jacobian, scaled by the robust residual spread
    J = sol.jac
    s2 = (1.4826 * np.median(np.abs(rt))) ** 2
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        sd = np.full(len(x), np.nan)

    cam, base, off, sag, mounts, chain = model.unpack(x)
    deg = np.degrees
    print("\njoint offsets (deg)      fit ± sd        S7        joint_mapping")
    s7o = json.loads((REPO / "calibration/2026-10-10_link_tilt_fit.json").read_text(encoding="utf-8"))["S7 offsets only"]["offset"]
    for k, (n, pn) in enumerate((("shoulder_lift", "off_lift"), ("elbow_flex", "off_elbow"), ("wrist_flex", "off_flex"))):
        print(f"  {n:<14} {deg(x[model.idx[pn]]):8.2f} ± {deg(sd[model.idx[pn]]):.2f}   {deg(s7o[n]):8.2f}   {deg(jm.OFFSET_RAD[n]):8.2f}")
    print("sag (deg per N.m of gravity torque):", {n: f"{deg(x[model.idx[n]]):.2f} ± {deg(sd[model.idx[n]]):.2f}" for n in ("sag_lift", "sag_elbow", "sag_flex")})
    bt = base[:3, 3]
    tilt = R.from_matrix(base[:3, :3]).as_euler("xyz", degrees=True)
    print(f"base: link0 at ({bt[0]*100:.2f}, {bt[1]*100:.2f}, {bt[2]*100:.2f}) cm -> pan axis at "
          f"({(bt[0] + fk._PAN_AXIS_XY[0])*100:.2f}, {bt[1]*100:.2f}) cm; rot xyz {np.round(tilt, 2)} deg (z = yaw incl. pan offset)")
    print(f"camera: at {np.round(cam[:3, 3]*100, 1)} cm")
    if args.free_lengths:
        up, fo = np.linalg.norm(fk._CHAIN[2][0]), fk._CHAIN[3][0][0]
        for n, u in (("scale_upper", up), ("scale_fore", fo)):
            print(f"{n}: {x[model.idx[n]]:.4f} ± {sd[model.idx[n]]:.4f} -> {x[model.idx[n]]*u*1000:.1f} mm (URDF {u*1000:.1f})")
    if args.free_joint_scales:
        print("effective joint scale (x 1.7996 deg/unit):", {n: f"{x[model.idx[n]]:.4f} ± {sd[model.idx[n]]:.4f}" for n in ("jscale_lift", "jscale_elbow", "jscale_flex")})
    if args.free_focal:
        print(f"focal scale: {x[model.idx['focal_scale']]:.4f} ± {sd[model.idx['focal_scale']]:.4f} (fx {Kmat[0, 0]:.1f} -> {Kmat[0, 0] * x[model.idx['focal_scale']]:.1f})")
    if args.free_marker_size:
        ms_ = x[model.idx["marker_scale"]]
        print(f"arm marker size: {ms_ * MARKER_M * 1000:.2f} ± {sd[model.idx['marker_scale']] * MARKER_M * 1000:.2f} mm (assumed 60)")
    if args.free_mat_scale:
        print(f"mat_scale (true m per mat m): {x[model.idx['mat_scale']]:.4f} ± {sd[model.idx['mat_scale']]:.4f}")
    i_lift, i_pitch = model.idx["off_lift"], model.idx["base_r1"]
    if np.isfinite(sd[i_lift]) and sd[i_lift] > 0 and sd[i_pitch] > 0:
        print(f"corr(lift offset, base pitch) = {cov[i_lift, i_pitch] / (sd[i_lift] * sd[i_pitch]):+.2f};"
              f" lift offset + base pitch = {deg(x[i_lift] + x[i_pitch]):.2f} deg")
    m1 = [mounts[("id1", k)] for k in range(3)]
    for k in (1, 2):
        dt = np.linalg.norm(m1[k][:3, 3] - m1[0][:3, 3]) * 1000
        dr = deg(R.from_matrix(m1[0][:3, :3].T @ m1[k][:3, :3]).magnitude())
        print(f"ID1 taping {k} vs 0: {dt:.1f} mm, {dr:.1f} deg")

    def arm_only(ss, i):
        cam_, *_ = model.unpack(x)
        errs, pos_err = [], []
        mounts_ = model.unpack(x)[4]
        for s in ss:
            if i not in s["det"]:
                continue
            frames, _ = model.link_frames(x, s["state"])
            key = ("id1", seg[s["pose_id"]]) if i == 1 else (f"id{i}", 0)
            mw = frames[ARM_LINK[i] - 1] @ mounts_[key]
            pw = (mw[:3, :3] @ (x[model.idx["marker_scale"]] * CORNERS).T).T + mw[:3, 3]
            errs.append(np.linalg.norm(project(Kmat, cam_, pw) - s["det"][i], axis=1).mean())
            ok, rv, tv = cv2.solvePnP(np.c_[x[model.idx["marker_scale"]] * CORNERS[:, :2], np.zeros(4)], s["det"][i], Kmat, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
            meas = cam_ @ T(rv.ravel(), tv.ravel())
            pos_err.append(np.linalg.norm(meas[:3, 3] - mw[:3, 3]) * 1000)
        return np.array(errs), np.array(pos_err)

    print("\nper-marker error           fit set: px (median / p90)  mm*      held out: px (median / p90)  mm*")
    report = {}
    for i in (1, 5, 6):
        a, am = arm_only(train, i)
        b, bm = arm_only(test, i)
        f = lambda v: f"{np.median(v):5.2f} / {np.percentile(v, 90):5.2f}" if v.size else "  -  "  # noqa: E731
        g = lambda v: f"{np.median(v):5.1f}" if v.size else "  - "  # noqa: E731
        print(f"  ID{i}   n={a.size:3d}/{b.size:3d}        {f(a)}       {g(am)}           {f(b)}       {g(bm)}")
        report[f"ID{i}"] = {"fit_px_median": float(np.median(a)) if a.size else None, "held_px_median": float(np.median(b)) if b.size else None,
                            "fit_mm_median": float(np.median(am)) if am.size else None, "held_mm_median": float(np.median(bm)) if bm.size else None}
    print("  * marker-centre distance between the model and a single-marker PnP of the same still (depth-noisy)")

    out = Path(args.out) if args.out else None
    if out:
        res = {"what": "scripts/solve_link_markers.py", "args": vars(args),
               "offset_rad": {n: float(off[k]) for k, n in enumerate(jm.LEROBOT_NAMES[:5])},
               "offset_sd_rad": {"shoulder_lift": float(sd[model.idx["off_lift"]]), "elbow_flex": float(sd[model.idx["off_elbow"]]),
                                 "wrist_flex": float(sd[model.idx["off_flex"]])},
               "sag_rad_per_nm": {n: float(x[model.idx[n]]) for n in ("sag_lift", "sag_elbow", "sag_flex")},
               "world_from_link0": base.tolist(), "world_from_camera": cam.tolist(),
               "chain": [[list(o), a] for o, a in chain], "report": report, "x": x.tolist(), "names": model.names}
        (REPO / out).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
