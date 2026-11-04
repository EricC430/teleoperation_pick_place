"""Usage: fl_vs_wrist.py <tag> <dataset dir> <front-left extrinsics json> <wrist_cup_chamfer json> <prep dir> [<out dir>]

10/14 15:12 rollout: cup offset from the front-left (no arm involved) vs from the wrist (arm FK + mount), per episode.
front-left: frame 2 of each episode (arm at HOME), dark band bottom (H 15-35, S>120, V<110) vs the projected base circle's
lowest point, both back-projected onto the mat with 2026-10-14_camera_extrinsics_front-left.json (taken 19:01 the same day)."""
import json, sys, glob
from pathlib import Path
import cv2, numpy as np, pyarrow.parquet as pq
from scipy.spatial.transform import Rotation as R
sys.path[:0] = ['sim', 'scripts']
import camera_distortion as cd
TAG = sys.argv[1]; D = Path(sys.argv[2]); EXT = sys.argv[3]; WR = sys.argv[4]; PREP = Path(sys.argv[5])
OUT = Path(sys.argv[6]) if len(sys.argv) > 6 else Path(__file__).resolve().parent
M = cd.load_model(Path("calibration/2026-10-07_camera_intrinsics_front-left.json"))
X = json.load(open(EXT)); q = X["quat_wxyz"]; T = np.eye(4); T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
T[:3, 3] = np.array(X["pos_m"]) - [0, 0, X.get("table_top_z", 0.0)]
def proj(P):
    pc = np.linalg.inv(T) @ np.r_[P, 1]; x = pc[:2] / pc[2]
    return cd.distort_pixels(np.array([[M.K[0, 0] * x[0] + M.K[0, 2], M.K[1, 1] * x[1] + M.K[1, 2]]]), M)[0]
def ray_table(px):
    u = cd.undistort_pixels(np.array([px]), M)[0]; r = np.r_[(u[0] - M.K[0, 2]) / M.K[0, 0], (u[1] - M.K[1, 2]) / M.K[1, 1], 1]
    rw = T[:3, :3] @ r; o = T[:3, 3]; return o - o[2] / rw[2] * rw
VID = {}
for f in sorted(glob.glob(str(D / "meta/episodes/*/*.parquet"))):
    t = pq.read_table(f).to_pydict()
    for i, e in enumerate(t["episode_index"]):
        VID[e] = (t["videos/observation.images.front-left/file_index"][i], t["videos/observation.images.front-left/from_timestamp"][i])
FPS = json.load(open(D / "meta/info.json"))["fps"]
want = {}
for e, (fi, t0) in VID.items():
    want.setdefault(fi, {})[int(round(t0 * FPS)) + 2] = e
img = {}
for fi, w in want.items():
    cap = cv2.VideoCapture(str(D / f"videos/observation.images.front-left/chunk-000/file-{fi:03d}.mp4")); k = 0
    while k <= max(w):
        ok, im = cap.read()
        if not ok: break
        if k in w: img[w[k]] = im
        k += 1
wr = json.load(open(WR))["episodes"]
res = {}; tiles = []
for e in sorted(VID):
    E = json.load(open(PREP / f"cl_ep{e:02d}/episode.json")); c = np.array(E["cup_xy_m"]); im = img.get(e)
    if im is None: continue
    pts = np.array([proj(np.r_[c[0] + 0.025 * np.cos(a), c[1] + 0.025 * np.sin(a), 0]) for a in np.linspace(0, 2 * np.pi, 72, endpoint=False)])
    sb = pts[np.argmax(pts[:, 1])]
    if sb[0] < 10 or sb[0] > 838 or sb[1] < 10 or sb[1] > 470: continue
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    band = ((hsv[..., 0] >= 15) & (hsv[..., 0] <= 35) & (hsv[..., 1] > 120) & (hsv[..., 2] < 110)).astype(np.uint8)
    roi = np.zeros_like(band); roi[max(0, int(sb[1]) - 120):int(sb[1]) + 60, max(0, int(sb[0]) - 100):int(sb[0]) + 100] = 1
    band = cv2.morphologyEx(band * roi, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(band)
    ok_ = [i for i in range(1, n) if st[i, 4] >= 80]
    if not ok_: continue
    i = max(ok_, key=lambda i: st[i, 4]); by, bx = np.nonzero(lab == i); cx = bx.mean(); ry = by[np.abs(bx - cx) < 4].max()
    d_fl = (ray_table([cx, ry]) - ray_table(sb))[:2] * 100
    res[e] = dict(place=E["placement"], fl_cm=d_fl.tolist(), wrist_cm=wr[str(e)]["dw_med"] if str(e) in wr else None)
    v = im.copy(); v[lab == i] = (0, 255, 0); cv2.circle(v, (int(cx), int(ry)), 4, (255, 0, 255), -1); cv2.circle(v, (int(sb[0]), int(sb[1])), 4, (0, 0, 255), -1)
    x0 = int(min(max(cx - 80, 0), 848 - 160)); y0 = int(min(max(ry - 110, 0), 480 - 140))
    tl = cv2.resize(v[y0:y0 + 140, x0:x0 + 160], (240, 210)); cv2.putText(tl, f"ep{e} {E['placement']}", (4, 16), 0, 0.5, (0, 255, 255), 1); tiles.append(tl)
json.dump(res, open(OUT / f"fl_vs_wrist_{TAG}.json", "w"), indent=0)
while len(tiles) % 8: tiles.append(np.zeros_like(tiles[0]))
cv2.imwrite(str(OUT / f"fl_vs_wrist_{TAG}.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
F = np.array([r["fl_cm"] for r in res.values()])
print(f"front-left measured {len(res)} episodes; real - placement |d| median {np.median(np.linalg.norm(F, axis=1)):.2f} cm, vector median {np.median(F, 0).round(2)}")
both = [(e, r) for e, r in res.items() if r["wrist_cm"] is not None]
for e, r in both:
    print(f"ep{e:02d} {r['place']:>4} front-left ({r['fl_cm'][0]:+5.1f},{r['fl_cm'][1]:+5.1f})  wrist ({r['wrist_cm'][0]:+5.1f},{r['wrist_cm'][1]:+5.1f})  diff |{np.linalg.norm(np.subtract(r['wrist_cm'], r['fl_cm'])):.1f}|")
if both:
    A = np.array([r["fl_cm"] for _, r in both]); B = np.array([r["wrist_cm"] for _, r in both])
    print(f"n both {len(both)}: |front-left| median {np.median(np.linalg.norm(A,axis=1)):.2f}, |wrist| median {np.median(np.linalg.norm(B,axis=1)):.2f}, |wrist - front-left| median {np.median(np.linalg.norm(B-A,axis=1)):.2f} cm")
    print("corr x", np.corrcoef(A[:, 0], B[:, 0])[0, 1].round(2), " corr y", np.corrcoef(A[:, 1], B[:, 1])[0, 1].round(2))
