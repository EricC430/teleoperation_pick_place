"""Where the paper cup really is in the wrist view vs where the arm model puts it (docs/worklog/2026-10-31_wrist_cup_offset.md).

    python3 scripts/prep_closed_loop_replay.py <prep dir> <dataset>          # joint radians, cup placement, gripper
    python3 calibration/2026-10-31_wrist_cup_offset/wrist_cup_chamfer.py <dataset dir> <prep dir> <tag> [<out dir> [<lag frames>]]

Camera: link5 (calibration/2026-10-14_link_marker_fit.json: offsets, sag, base) @ the CAD lens mount
(calibration/2026-10-30_camera_extrinsics_wrist_cad_lens.json), wrist intrinsics 2026-10-07.
Frames: before the first gripper close, every 2nd frame, FK camera >= 30 cm over the mat, the whole predicted cup inside
the image with a 40 px margin; at most 40 per episode.
Rule: the predicted cup contour (base circle r 2.5 cm on the mat + rim circle r 3.75 cm at 9.3 cm; rim points all used,
base points only where they lie on the silhouette hull) is shifted on the mat over a 0.25 cm grid within +-7 cm; the cost
is the mean distance from the contour points to the nearest Canny edge (blur 5, thresholds 20/50), capped at 15 px.
The best shift is the apparent cup offset on the mat (world frame, and radial = from the arm base toward the cup /
tangential = 90 deg left of it). Frames whose best shift hits the +-7 cm border are dropped. Per episode: medians.
Overlay sheet: one frame per episode - red dots = contour at the placement point, green = after the best shift,
orange = the Canny edges it was matched to.
"""
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "sim"), str(REPO)]
import camera_distortion as cd  # noqa: E402
import touch_test as TT  # noqa: E402

D, PREP, TAG = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT = Path(sys.argv[4]) if len(sys.argv) > 4 else Path(__file__).resolve().parent
LAG = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0   # camera pose from the state at f + LAG (linear in the readings) for image f (timing test)


def state_at(st, x):
    x = min(max(x, 0.0), len(st) - 1.0); i = min(int(np.floor(x)), len(st) - 2); a = x - i
    return (1 - a) * st[i] + a * st[i + 1]
M = cd.load_model(REPO / "calibration/2026-10-07_camera_intrinsics_wrist.json")
FM = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")
_m = json.load(open(REPO / "calibration/2026-10-30_camera_extrinsics_wrist_cad_lens.json"))
X5c = np.eye(4); q = _m["quat_wxyz"]; X5c[:3, :3] = Rot.from_quat([q[1], q[2], q[3], q[0]]).as_matrix(); X5c[:3, 3] = _m["pos_m"]
FPS = json.load(open(D / "meta/info.json"))["fps"]
VID = {}
for f in sorted(glob.glob(str(D / "meta/episodes/*/*.parquet"))):
    t = pq.read_table(f).to_pydict()
    for i, e in enumerate(t["episode_index"]):
        VID[e] = (t["videos/observation.images.wrist/chunk_index"][i], t["videos/observation.images.wrist/file_index"][i],
                  t["videos/observation.images.wrist/from_timestamp"][i])

A = np.linspace(0, 2 * np.pi, 72, endpoint=False); HS = np.linspace(0, 0.093, 12)
RIM_A = np.linspace(0, 2 * np.pi, 48, endpoint=False)
G = np.arange(-0.07, 0.0701, 0.0025); GX, GY = np.meshgrid(G, G); SH = np.c_[GX.ravel(), GY.ravel()]; K0 = len(SH) // 2


def proj_cam(Pc):
    x = Pc[..., :2] / Pc[..., 2:3]
    uv = np.stack([M.K[0, 0] * x[..., 0] + M.K[0, 2], M.K[1, 1] * x[..., 1] + M.K[1, 2]], -1)
    return cd.distort_pixels(uv.reshape(-1, 2), M).reshape(uv.shape)


def to_cam(P, T):
    return (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3]


plan = {}
for e in sorted(VID):
    p = PREP / f"cl_ep{e:02d}/episode.json"
    if not p.exists():
        continue
    E = json.load(open(p)); st = np.array(E["state"]); c = np.array(E["cup_xy_m"])
    gr = np.array(json.load(open(PREP / f"cl_ep{e:02d}/gripper.json"))["gripper_rad_from_state"])
    closed = np.nonzero(gr < gr[:30].mean() - 0.35)[0]; fc = int(closed[0]) if len(closed) else len(st)
    cup = np.array([[c[0] + r * np.cos(a), c[1] + r * np.sin(a), h] for h, r in zip(HS, 0.025 + 0.0125 * HS / 0.093) for a in A])
    fr = []
    for f in range(0, fc, 2):
        T = FM.link5(st[f]) @ X5c
        if T[2, 3] < 0.30:
            continue
        pc = to_cam(cup, T)
        if (pc[:, 2] < 0.02).any():
            continue
        uv = proj_cam(pc)
        if uv[:, 0].min() < 40 or uv[:, 0].max() > 600 or uv[:, 1].min() < 40 or uv[:, 1].max() > 440:
            continue
        fr.append((f, FM.link5(state_at(st, f + LAG)) @ X5c if LAG else T))
    plan[e] = dict(c=c, frames=fr[:40], place=E["placement"])

need = {}
for e, v in plan.items():
    ck, fi, t0 = VID[e]
    for f, _ in v["frames"]:
        need.setdefault((ck, fi), {})[int(round(t0 * FPS)) + f] = (e, f)
img = {}
for (ck, fi), want in need.items():   # sequential decode: seeking is not frame-accurate
    cap = cv2.VideoCapture(str(D / f"videos/observation.images.wrist/chunk-{ck:03d}/file-{fi:03d}.mp4")); k = 0
    while k <= max(want):
        ok, im = cap.read()
        if not ok:
            break
        if k in want:
            img[want[k]] = im
        k += 1

res = {}; tiles = []
for e, v in plan.items():
    c = v["c"]; rows = []; shown = False
    u = (c - FM.base[:2, 3]) / np.linalg.norm(c - FM.base[:2, 3]); tng = np.array([-u[1], u[0]])
    P = np.array([[c[0] + r * np.cos(a), c[1] + r * np.sin(a), h] for h, r in ((0.0, 0.025), (0.093, 0.0375)) for a in RIM_A])
    for f, T in v["frames"]:
        im = img.get((e, f))
        if im is None:
            continue
        ed = cv2.Canny(cv2.GaussianBlur(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY), (5, 5), 0), 20, 50)
        dt = np.minimum(cv2.distanceTransform((ed == 0).astype(np.uint8), cv2.DIST_L2, 3), 15)
        Pc = to_cam(P, T); d3 = np.c_[SH, np.zeros(len(SH))] @ np.linalg.inv(T)[:3, :3].T
        uvp = proj_cam(Pc[None] + d3[:, None])
        use = np.union1d(np.arange(48, 96), cv2.convexHull(uvp[K0].astype(np.float32), returnPoints=False).ravel())
        uu, vv = uvp[:, use, 0], uvp[:, use, 1]
        inside = (uu >= 0) & (uu < 640) & (vv >= 0) & (vv < 480)
        cost = np.where(inside, dt[np.clip(vv, 0, 479).astype(int), np.clip(uu, 0, 639).astype(int)], 15).mean(1)
        k = int(np.argmin(cost)); sh = SH[k]
        if np.abs(sh).max() > 0.069:
            continue
        rows.append(dict(f=f, dw_cm=(sh * 100).tolist(), rad_cm=float(sh @ u * 100), tan_cm=float(sh @ tng * 100),
                         cost0=float(cost[K0]), cost=float(cost[k]), cam_z=float(T[2, 3]),
                         tilt=float(np.degrees(np.arccos(-T[2, 2]))),
                         cam_reach=float(np.linalg.norm(T[:2, 3] - FM.base[:2, 3]) * 100)))
        if not shown and len(rows) >= 3:
            vis = im.copy(); vis[ed > 0] = (0, 200, 255)
            for kk, col in ((K0, (0, 0, 255)), (k, (0, 255, 0))):
                for p_ in uvp[kk, use]:
                    cv2.circle(vis, (int(p_[0]), int(p_[1])), 2, col, -1)
            cv2.putText(vis, f"ep{e} f{f} {v['place']} d({sh[0]*100:+.1f},{sh[1]*100:+.1f})cm", (6, 26), 0, 0.7, (255, 255, 0), 2)
            tiles.append(cv2.resize(vis, (320, 240))); shown = True
    if rows:
        dw = np.array([r["dw_cm"] for r in rows])
        med = lambda key: float(np.median([r[key] for r in rows]))  # noqa: E731
        res[e] = dict(place=v["place"], cup=c.tolist(), n=len(rows), dw_med=np.median(dw, 0).tolist(),
                      rad_med=med("rad_cm"), tan_med=med("tan_cm"), tilt_med=med("tilt"), reach_med=med("cam_reach"),
                      cam_z_med=med("cam_z"), cost0=med("cost0"), cost=med("cost"), rows=rows)
OUT.mkdir(parents=True, exist_ok=True)
json.dump(dict(dataset=D.name, prep=str(PREP), episodes=res), open(OUT / f"{TAG}.json", "w"), indent=0)
while len(tiles) % 6:
    tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT / f"{TAG}_overlay.jpg"), np.vstack([np.hstack(tiles[i:i + 6]) for i in range(0, len(tiles), 6)]),
            [cv2.IMWRITE_JPEG_QUALITY, 80])
RT = np.array([[r["rad_med"], r["tan_med"]] for r in res.values()])
print(f"{TAG}: {len(res)}/{len(plan)} episodes measured; radial median {np.median(RT[:, 0]):+.2f} sd {RT[:, 0].std():.2f}, "
      f"tangential median {np.median(RT[:, 1]):+.2f} sd {RT[:, 1].std():.2f} cm")
