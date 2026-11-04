#!/usr/bin/env python
"""Wrist-camera mount candidates rendered side by side with the real wrist video.

Three steps, so that only the rendering step depends on the simulator version:

  prep     (plain python, repo modules)  episode -> <job dir>: per-frame link5/6/7 and cup poses from the touch-pinned
           arm model, one link5_from_camera per candidate, the wrist lens as remap tables, the meshes. Nothing in the
           job needs the repo afterwards, so it can be copied to the Windows Isaac Sim box or into the container.
  render   --backend raster: numpy/OpenCV painter's rasteriser, no Isaac. For an Isaac render run
           sim/render_wrist_job.py on the job with that machine's Isaac python (it picks the camera API by version).
  compose  real video + one render set -> tiles and an mp4. The outlines always come from the raster masks: they are
           geometry only, identical for every backend.

    python3 scripts/render_wrist_candidates.py prep --dataset <lerobot root> --episode 0 --job <dir> [--candidates cad,cad_lens,fit1015]
    python3 scripts/render_wrist_candidates.py render --job <dir>                       # raster colour + masks
    sim/run_in_container.sh render_wrist_job.py --job /workspace/test_isaaclab/<dir>    # Isaac 4.5/5.x container
    <isaac-sim>\\python.bat <repo>\\sim\\render_wrist_job.py --job <dir>                 # Isaac 6.x standalone
    python3 scripts/render_wrist_candidates.py compose --job <dir> [--render isaac] [--frames 0,100]
    python3 scripts/render_wrist_candidates.py all --dataset ... --episode 0 --job <dir>   # prep + raster + compose

Candidates (link5_from_camera, OpenCV axes; Isaac Lab "ros" = OpenCV):
  cad       sim/scene_constants.py CAM_WRIST_OFFSET_* -- the CAD part's placement. Its origin is NOT on the lens: in the
            link5 STL (which carries the camera) the lens barrel axis is parallel to this z but 12.6 mm off, at -y.
  cad_lens  same rotation, origin moved to the centre of the lens front face found in that STL (barrel r 6.6 mm,
            from -8.2 to +13.8 mm along z). Where the pinhole centre sits inside the barrel is an assumption.
  fit1015   calibration/2026-10-15_camera_extrinsics_wrist.json (what sim/physics_replay.py uses).
The cup (top 7.5 / bottom 5.0 / height 9.3 cm, as sim/gen_paper_cup_usd.py) stands on the episode's placement until the
first gripper close, then follows link5 rigidly -- a stand-in, not where the real cup went. The table is a 5 cm
checkerboard in the mat frame, not the mat's printed pattern.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import struct
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R

REPO = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "sim")]
import camera_distortion as cd  # noqa: E402
import joint_mapping as jm  # noqa: E402
import scene_constants as SC  # noqa: E402
import touch_test as TT  # noqa: E402

MESH_DIR = Path("/home/boyuchen/isaaclab_volume/assets/open_manipulator_description/meshes/omx_f")
MESHES = {"link5": "follower_06_pan_Revised", "link6": "follower_07_gripper_motorized", "link7": "follower_08_gripper_gear"}
FIT = REPO / "calibration/2026-10-14_link_marker_fit.json"
INTR = REPO / "calibration/2026-10-07_camera_intrinsics_wrist.json"
CUP = dict(top_r=0.0375, bot_r=0.025, h=0.093)
GRIP_CLOSE_BELOW = 54.0     # gripper.pos; open ~59, on a paper cup ~47-50 (S5 §11-1)
FPS = 15
# lens front centre in the CAD camera frame (mm), from the barrel rings in follower_06_pan_Revised.stl
CAD_LENS_FRONT_MM = (0.0, -12.61, 13.8)
COLORS = {"link5": (70, 70, 70), "link6": (150, 150, 150), "link7": (150, 150, 150), "cup": (235, 235, 225),
          "table": (150, 175, 195)}   # BGR, shared with sim/render_wrist_job.py through the job
MASK_ID = {"cup": 1, "link5": 2, "link6": 3, "link7": 3}
OUTLINE = {"cad": (0, 0, 255), "cad_lens": (0, 200, 255), "fit1015": (255, 160, 0)}   # BGR


# ---------------------------------------------------------------------------------------------------------- prep
def load_stl(name):
    b = (MESH_DIR / f"{name}.stl").read_bytes()
    n = struct.unpack("<I", b[80:84])[0]
    a = np.frombuffer(b[84:84 + n * 50], dtype=np.dtype([("n", "<3f4"), ("v", "<9f4"), ("a", "<u2")]))
    return a["v"].reshape(-1, 3, 3).astype(np.float64) / 1000.0


def cup_tris(n=48, rings=6):
    t = []
    for i in range(n):
        a0, a1 = 2 * np.pi * i / n, 2 * np.pi * (i + 1) / n
        for k in range(rings):
            z0, z1 = CUP["h"] * k / rings, CUP["h"] * (k + 1) / rings
            r0 = CUP["bot_r"] + (CUP["top_r"] - CUP["bot_r"]) * k / rings
            r1 = CUP["bot_r"] + (CUP["top_r"] - CUP["bot_r"]) * (k + 1) / rings
            p = [(r0 * np.cos(a0), r0 * np.sin(a0), z0), (r0 * np.cos(a1), r0 * np.sin(a1), z0),
                 (r1 * np.cos(a1), r1 * np.sin(a1), z1), (r1 * np.cos(a0), r1 * np.sin(a0), z1)]
            t += [(p[0], p[1], p[2]), (p[0], p[2], p[3])]
    return np.array(t)


def mount(name):
    if name in ("cad", "cad_lens"):
        p, q = np.array(SC.CAM_WRIST_OFFSET_POS, float), SC.CAM_WRIST_OFFSET_ROT
        src = "sim/scene_constants.py CAM_WRIST_OFFSET_* (CAD part placement)"
    elif name == "fit1015":
        d = json.loads((REPO / "calibration/2026-10-15_camera_extrinsics_wrist.json").read_text())
        p, q = np.array(d["pos_m"], float), d["quat_wxyz"]
        src = "calibration/2026-10-15_camera_extrinsics_wrist.json"
    else:
        raise KeyError(name)
    X = np.eye(4); X[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); X[:3, 3] = p
    if name == "cad_lens":
        X[:3, 3] = p + X[:3, :3] @ (np.array(CAD_LENS_FRONT_MM) / 1000.0)
        src += f" + lens front centre {CAD_LENS_FRONT_MM} mm in its own frame"
    return X, src


def tf(origin, rz=0.0):
    T = np.eye(4); T[:3, :3] = R.from_euler("z", rz).as_matrix(); T[:3, 3] = origin
    return T


def read_episode(root: Path, ep: int):
    rows = []
    for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet"))):
        t = pq.read_table(f).to_pydict()
        rows += [{k: t[k][i] for k in t if not k.startswith("stats/")} for i in range(len(t["episode_index"]))]
    e = next(r for r in rows if r["episode_index"] == ep)
    data = []
    for f in sorted(glob.glob(str(root / "data/*/*.parquet"))):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"]).to_pydict()
        data += [(fi, st) for ei, fi, st in zip(t["episode_index"], t["frame_index"], t["observation.state"]) if ei == ep]
    data.sort()
    vid = root / "videos/observation.images.wrist" / f"chunk-{e['videos/observation.images.wrist/chunk_index']:03d}" / \
        f"file-{e['videos/observation.images.wrist/file_index']:03d}.mp4"
    return e, np.array([d[1] for d in data], float), vid


def placement_xy(meta_csv: Path, ep: int):
    r = next(x for x in csv.DictReader(open(meta_csv, encoding="utf-8-sig")) if int(x["episode_index"]) == ep)
    pid = r["placement_id"]
    pid = f"train_{int(pid[1:]):03d}" if pid.startswith("t") and pid[1:].isdigit() else pid
    for f in sorted((REPO / "configs/placements").glob("campA_136sym_*_*.csv")):
        for x in csv.DictReader(open(f, encoding="utf-8")):
            if x["placement_id"] == pid:
                return pid, float(x["x_cm"]) / 100, float(x["y_cm"]) / 100
    raise KeyError(pid)


def lens_tables(path):
    """Ideal pinhole canvas (same K, padded so every real pixel has a source) -> real lens, as physics_replay.py."""
    m = cd.load_model(Path(path)); W, H, K = m.width, m.height, m.K
    us, vs = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    ideal = cd.undistort_pixels(np.stack([us.ravel(), vs.ravel()], 1), m)
    cx, cy = K[0, 2], K[1, 2]
    hw = int(np.ceil(max(np.abs(ideal[:, 0] - cx).max(), W / 2))) + 2
    hh = int(np.ceil(max(np.abs(ideal[:, 1] - cy).max(), H / 2))) + 2
    map_x = (ideal[:, 0] - cx + hw).reshape(H, W).astype(np.float32)
    map_y = (ideal[:, 1] - cy + hh).reshape(H, W).astype(np.float32)
    rays = np.stack([(ideal[:, 0] - cx) / K[0, 0], (ideal[:, 1] - cy) / K[1, 1], np.ones(len(ideal))], 1)
    return dict(W=W, H=H, K=K.tolist(), hw=hw, hh=hh), dict(map_x=map_x, map_y=map_y, rays=rays.astype(np.float32))


def cmd_prep(a):
    root, job = Path(a.dataset), Path(a.job); job.mkdir(parents=True, exist_ok=True)
    model = TT.FitModel(FIT)
    e, st, vid = read_episode(root, a.episode)
    pid, cx, cy = placement_xy(REPO / "episode_meta" / f"{root.name}.csv", a.episode)
    cands = a.candidates.split(",")
    X = {k: mount(k) for k in cands}
    grip = jm.lerobot_to_urdf_rad(np.c_[np.zeros((len(st), 5)), st[:, 5]])[:, 5]
    closed = np.nonzero(st[:, 5] < GRIP_CLOSE_BELOW)[0]
    f_close = int(closed[0]) if len(closed) else None
    cup0 = tf((cx, cy, 0.0)); link5_cup = None
    frames = []
    for f in range(len(st)):
        W5 = model.link5(st[f, :5])
        if f_close is not None and f >= f_close:
            if link5_cup is None:
                link5_cup = np.linalg.inv(model.link5(st[f_close, :5])) @ cup0
            Wc = W5 @ link5_cup
        else:
            Wc = cup0
        frames.append(dict(link5=W5.tolist(), link6=(W5 @ tf((0.0295, 0.0075, 0.0), grip[f])).tolist(),
                           link7=(W5 @ tf((0.0295, -0.0108, 0.0), -grip[f])).tolist(), cup=Wc.tolist(),
                           cam={k: (W5 @ X[k][0]).tolist() for k in cands}, cup_attached=bool(link5_cup is not None)))
    lens, maps = lens_tables(INTR)
    np.savez_compressed(job / "lens_maps.npz", **maps)
    np.savez_compressed(job / "meshes.npz", cup=cup_tris(), **{k: load_stl(v) for k, v in MESHES.items()})
    info = dict(dataset=root.name, dataset_root=str(root.resolve()), episode=a.episode, fps=FPS, placement=pid,
                cup_xy_m=[cx, cy], first_close_frame=f_close, video=str(vid.resolve()),
                video_start_frame=int(round(e["videos/observation.images.wrist/from_timestamp"] * FPS)),
                near_m=a.near_m, lens=lens, colors_bgr=COLORS, mask_id=MASK_ID,
                candidates={k: dict(source=v[1], link5_from_camera=v[0].tolist()) for k, v in X.items()},
                arm_model=str(FIT.relative_to(REPO)), intrinsics=str(INTR.relative_to(REPO)), frames=frames)
    (job / "job.json").write_text(json.dumps(info))
    print(json.dumps(dict({k: info[k] for k in ("dataset", "episode", "placement", "first_close_frame")}, frames=len(frames))))


# ---------------------------------------------------------------------------------------------------------- raster
class Job:
    def __init__(self, d: Path):
        self.dir = d; self.j = json.loads((d / "job.json").read_text())
        L = self.j["lens"]; self.W, self.H, self.K, self.hw, self.hh = L["W"], L["H"], np.array(L["K"]), L["hw"], L["hh"]
        m = np.load(d / "lens_maps.npz"); self.map_x, self.map_y, self.rays = m["map_x"], m["map_y"], m["rays"].astype(np.float64)
        self.mesh = dict(np.load(d / "meshes.npz"))
        self.cands = list(self.j["candidates"])

    def parts(self, f):
        fr = self.j["frames"][f]
        out = []
        for name in ("cup", "link5", "link6", "link7"):
            T = np.array(fr[name]); M = self.mesh[name]
            out.append((name, (M.reshape(-1, 3) @ T[:3, :3].T + T[:3, 3]).reshape(-1, 3, 3)))
        return out


def table_image(job, cam_from_world, step=0.05):
    Rwc = cam_from_world[:3, :3].T; c = -Rwc @ cam_from_world[:3, 3]
    d = job.rays @ Rwc.T
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -c[2] / d[:, 2]
    hit = t > 0
    X = c[:2] + t[:, None] * d[:, :2]
    chk = ((np.floor(X[:, 0] / step) + np.floor(X[:, 1] / step)) % 2).astype(int)
    img = np.empty((len(t), 3)); img[:] = (40, 40, 40)
    base = np.array(COLORS["table"], float)
    img[hit] = np.where(chk[hit, None] == 1, base, 0.85 * base)
    return img.reshape(job.H, job.W, 3).astype(np.uint8)


def raster(job, parts, cam_from_world, near, colour=True):
    """-> (BGR image or None, label mask with MASK_ID values), both through the real lens."""
    CH, CW = 2 * job.hh, 2 * job.hw
    canvas = np.zeros((CH, CW, 3), np.uint8); lab = np.zeros((CH, CW), np.uint8)
    tris = []
    for name, T in parts:
        P = (T.reshape(-1, 3) @ cam_from_world[:3, :3].T + cam_from_world[:3, 3]).reshape(-1, 3, 3)
        P = P[(P[:, :, 2] > near).all(1)]
        if not len(P):
            continue
        uv = np.stack([job.K[0, 0] * P[..., 0] / P[..., 2] + job.hw, job.K[1, 1] * P[..., 1] / P[..., 2] + job.hh], -1)
        inb = ((uv[..., 0] > -CW) & (uv[..., 0] < 2 * CW) & (uv[..., 1] > -CH) & (uv[..., 1] < 2 * CH)).all(1)
        P, uv = P[inb], uv[inb]
        nrm = np.cross(P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]); nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
        view = -P.mean(1); view /= np.linalg.norm(view, axis=1, keepdims=True)
        shade = 0.35 + 0.65 * np.abs((nrm * view).sum(1))
        base = np.array(COLORS[name], float)
        tris += [(P[k, :, 2].mean(), uv[k], base * shade[k], MASK_ID[name]) for k in range(len(P))]
    tris.sort(key=lambda t: -t[0])
    for _, uv, c, mid in tris:
        pts = np.round(uv * 16).astype(np.int32)
        if colour:
            cv2.fillConvexPoly(canvas, pts, tuple(float(x) for x in c), lineType=cv2.LINE_AA, shift=4)
        cv2.fillConvexPoly(lab, pts, mid, shift=4)
    lab = cv2.remap(lab, job.map_x, job.map_y, cv2.INTER_NEAREST, borderValue=0)
    if not colour:
        return None, lab
    img = table_image(job, cam_from_world)
    fg = cv2.remap(canvas, job.map_x, job.map_y, cv2.INTER_LINEAR, borderValue=0)
    img[lab > 0] = fg[lab > 0]
    return img, lab


def frame_list(job, spec):
    n = len(job.j["frames"])
    return [int(x) for x in spec.split(",")] if spec else list(range(n))


def cmd_render(a, colour=True):
    job = Job(Path(a.job))
    for f in frame_list(job, a.frames):
        parts = job.parts(f)
        for k in job.cands:
            img, lab = raster(job, parts, np.linalg.inv(np.array(job.j["frames"][f]["cam"][k])), job.j["near_m"], colour)
            (job.dir / "masks" / k).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(job.dir / "masks" / k / f"f{f:04d}.png"), lab)
            if colour:
                (job.dir / "renders/raster" / k).mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(job.dir / "renders/raster" / k / f"f{f:04d}.png"), img)


# ---------------------------------------------------------------------------------------------------------- compose
def label(img, lines):
    for i, s in enumerate(lines):
        cv2.putText(img, s, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, s, (8, 22 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


def cmd_compose(a):
    job = Job(Path(a.job)); j = job.j
    frames = frame_list(job, a.frames)
    if not all((job.dir / "masks" / k / f"f{f:04d}.png").exists() for k in job.cands for f in frames):
        a2 = argparse.Namespace(job=a.job, frames=a.frames); cmd_render(a2, colour=False)
    cap = cv2.VideoCapture(j["video"]); cap.set(cv2.CAP_PROP_POS_FRAMES, j["video_start_frame"])
    real, k, need = {}, 0, set(frames)
    while k <= max(frames):
        ok, fr = cap.read()
        if not ok:
            break
        if k in need:
            real[k] = fr
        k += 1
    out = job.dir / f"tiles_{a.render}"; out.mkdir(exist_ok=True)
    for f in frames:
        rl = real.get(f, np.zeros((job.H, job.W, 3), np.uint8))
        top, bot = [rl.copy()], [rl.copy()]
        label(top[0], [f"real wrist  ep{j['episode']} f{f}", j["dataset"][:42]])
        att = "  cup attached" if j["frames"][f]["cup_attached"] else ""
        for c in job.cands:
            img = cv2.imread(str(job.dir / "renders" / a.render / c / f"f{f:04d}.png"))
            img = np.zeros_like(rl) if img is None else img
            label(img, [f"{a.render}: {c}", f"near {j['near_m'] * 1000:.0f} mm{att}"]); top.append(img)
            lab = cv2.imread(str(job.dir / "masks" / c / f"f{f:04d}.png"), cv2.IMREAD_GRAYSCALE)
            ov = rl.copy()
            for mid, col in ((1, (0, 255, 0)), (2, (255, 0, 255)), (3, OUTLINE.get(c, (255, 255, 255)))):
                cs, _ = cv2.findContours((lab == mid).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(ov, cs, -1, col, 1, cv2.LINE_AA)
            label(ov, [f"real + {c} outlines", "fingers / cup(green) / link5(magenta)"]); bot.append(ov)
        cv2.imwrite(str(out / f"f{f:04d}.png"), np.vstack([np.hstack(top), np.hstack(bot)]))
    if not a.frames:
        mp4 = job.dir / f"ep{j['episode']:02d}_wrist_{a.render}_{'_'.join(job.cands)}.mp4"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(j["fps"]), "-pattern_type", "glob",
                        "-i", str(out / "f*.png"), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", str(mp4)], check=True)
        print(mp4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prep", "render", "compose", "all"):
        p = sp.add_parser(name)
        p.add_argument("--job", required=True)
        if name in ("prep", "all"):
            p.add_argument("--dataset", required=True)
            p.add_argument("--episode", type=int, default=0)
            p.add_argument("--candidates", default="cad,cad_lens,fit1015")
            p.add_argument("--near-m", type=float, default=0.005, help="near clip, as physics_replay wrist_near_m")
        if name != "prep":
            p.add_argument("--frames", default="", help="comma list; default all (and an mp4 from compose)")
        if name in ("compose", "all"):
            p.add_argument("--render", default="raster", help="renders/<name>: raster, or isaac from sim/render_wrist_job.py")
    a = ap.parse_args()
    if a.cmd in ("prep", "all"):
        cmd_prep(a)
    if a.cmd in ("render", "all"):
        cmd_render(a)
    if a.cmd in ("compose", "all"):
        cmd_compose(a)


if __name__ == "__main__":
    main()
