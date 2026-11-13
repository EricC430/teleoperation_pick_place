"""H16: in which frames of the human demos is the object inside the wrist view? (2026-11-12)

Why: Claude claimed that full-episode front-left dropout would mask mostly coarse-phase frames "where the wrist sees no
object"; Boyu objected: after the arm lifts, the object is usually in view. This counts it instead of assuming it.

Geometry (the 10-31 wrist_cup_offset setup, calibration/2026-10-31_wrist_cup_offset/wrist_cup_chamfer.py):
  camera pose = touch_test.FitModel(2026-10-14_link_marker_fit).link5(state readings) @ CAD lens mount (2026-10-30),
  intrinsics + distortion 2026-10-07, image 640 x 480; object at its placement (x_cm, y_cm of the placement table).
  cup: cylinder r 2.5 -> 3.75 cm, h 9.3 cm; bottle: r 3.47 cm, h 23 cm (docs/worklog/2026-11-11_bottle_model.md).
Per frame:  seen   = >= SEEN_FRAC (3%) of points on the object's surface project inside the image, in front of the camera
            (v1 = axis point at mid height inside: the check sheet showed it missing close-ups and objects in a corner)
            whole  = every surface point inside the image
Phases of an episode: home (before the arm has moved 3 deg in J0..J3 from frame 0), approach (moved, before the first
gripper close k, aim_error rule), window (the last 30 frames before k = where fldrop30 already masks), after (>= k).
Prints, per set and pooled per object: share of all frames in each phase, share 'seen' in each phase, and for the frames
that full-episode dropout would add (all frames outside the 30-frame window) how many have the object seen.
Check sheet: wrist frames with the projected axis point drawn, sampled over phases, to eyeball the projection.
Caveats: the arm model / mount are from 10/14 and 10/30; uvc_60 is 9/13 and the bottle demos 10/21 (10-31 worklog: 10/13
and earlier carry ~1.2 cm extra radial offset) -> cm-level error, matters only near the image border; the gripper
fingers can hide the object (not modelled); after the close the object is in the gripper ('seen' but irrelevant).
Human demo sets with placements only (the 60 base bottle demos have no episode -> placement table).

    uv run python scripts/input_reliance/h16_wrist_visibility.py
"""
import glob
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from matplotlib.path import Path as MplPath
from scipy.spatial.transform import Rotation as Rot

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "scripts"), str(REPO / "sim"), str(REPO), str(REPO / "scripts/input_reliance")]
import camera_distortion as cd  # noqa: E402
import touch_test as TT  # noqa: E402
import h14_approach_correction as P14  # noqa: E402

M = cd.load_model(REPO / "calibration/2026-10-07_camera_intrinsics_wrist.json")
FM = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")
_m = json.load(open(REPO / "calibration/2026-10-30_camera_extrinsics_wrist_cad_lens.json", encoding="utf-8"))
X5c = np.eye(4); _q = _m["quat_wxyz"]; X5c[:3, :3] = Rot.from_quat([_q[1], _q[2], _q[3], _q[0]]).as_matrix(); X5c[:3, 3] = _m["pos_m"]
W, HGT = 640, 480
HC = ".cache/lerobot/"
SETS = [("cup", "uvc_60 (A2)", "omx_pick_place_pilot_uvc_60"), ("cup", "normal_A1", "omx_pick_place_pilot_paper_cup_normal_A1"),
        ("cup", "far_wrist_A1", "omx_pick_place_pilot_paper_cup_far_wrist_A1"),
        ("bottle", "bottle normal_A1", "omx_pick_place_pilot_plastic_bottle_normal_A1"),
        ("bottle", "bottle precise_A1", "omx_pick_place_pilot_plastic_bottle_precise_A1")]
SHAPE = {"cup": (np.linspace(0, 0.093, 16), lambda z: 0.025 + 0.0125 * z / 0.093, 0.0465),
         "bottle": (np.linspace(0, 0.23, 24), lambda z: 0.0347 if z < 0.2 else 0.014, 0.12)}
ANG = np.linspace(0, 2 * np.pi, 48, endpoint=False)
SEEN_FRAC = 0.03  # v2: share of surface points inside the image (v1 used the axis point: missed close-ups and corners)
OUT = REPO / "data/logs/input_reliance"
WINDOW = 30


_e = np.linspace(0, 1, 120)
_BORDER = np.r_[np.c_[_e * (W - 1), np.zeros(120)], np.c_[np.full(120, W - 1), _e * (HGT - 1)],
                np.c_[(1 - _e) * (W - 1), np.full(120, HGT - 1)], np.c_[np.zeros(120), (1 - _e) * (HGT - 1)]]
FOV = MplPath(cd.undistort_pixels(_BORDER, M))  # the image border in ideal (pinhole) pixels


def project(P, T):
    """Inside = in front of the camera and the ideal (pinhole) pixel inside the undistorted image border. v2 tested the
    distorted pixel against the image rectangle: points far outside the field of view folded back into the image
    through the distortion polynomial (check sheet: dots on the bin at home)."""
    Pc = (np.linalg.inv(T) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
    front = Pc[:, 2] > 0.02
    x = Pc[:, :2] / np.where(front, Pc[:, 2], 1.0)[:, None]
    ideal = np.c_[M.K[0, 0] * x[:, 0] + M.K[0, 2], M.K[1, 1] * x[:, 1] + M.K[1, 2]]
    inside = front & FOV.contains_points(ideal)
    uv = np.full_like(ideal, np.nan)
    if inside.any():
        uv[inside] = cd.distort_pixels(ideal[inside], M)
    return uv, inside


def main() -> None:
    rows, keep = [], {}
    rng = np.random.default_rng(0)
    for obj, label, name in SETS:
        root = HC + name
        m = P14.meta(name + ".csv", root)
        t = pd.concat([pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
                       for f in sorted(Path(root, "data").rglob("*.parquet"))])
        zs, rad, zc = SHAPE[obj]
        for r in m.itertuples():
            g = t[t.episode_index == r.episode_index].sort_values("frame_index")
            if g.empty:
                continue
            s = np.stack(g["observation.state"].values); a = np.stack(g["action"].values)
            gr = a[:, 5]; depth = gr[0] - gr.min()
            k = int(np.argmax(gr < gr[0] - 0.6 * depth)) if depth >= 10 else len(s)
            moved = np.flatnonzero(np.abs(s[:, :4] - s[0, :4]).max(1) > 3)
            on = int(moved[0]) if len(moved) else len(s)
            # placement world xy (= x_cm, y_cm of the placement table, which equal r, theta within 0.01 cm)
            x, y = r.r / 100 * np.cos(np.deg2rad(r.th)), r.r / 100 * np.sin(np.deg2rad(r.th))
            ring = np.array([[x + rad(z) * np.cos(q), y + rad(z) * np.sin(q), z] for z in zs for q in ANG])
            axis = np.array([[x, y, zc]])
            keep[(label, int(r.episode_index))] = (s, ring)
            for f in range(len(s)):
                T = FM.link5(s[f]) @ X5c
                _, ins = project(ring, T)
                uv, c_in = project(axis, T)
                c_in = [ins.mean() >= SEEN_FRAC]
                ph = "home" if f < on else ("after" if f >= k else ("window" if f >= k - WINDOW else "approach"))
                rows.append(dict(object=obj, set=label, ep=r.episode_index, pid=r.placement_id, f=f, k=k, phase=ph,
                                 seen=bool(c_in[0]), whole=bool(ins.all()), frac_pts=float(ins.mean()),
                                 cam_z=T[2, 3], u=uv[0, 0], v=uv[0, 1]))
        print(f"{obj:6s} {label:18s}: {m.episode_index.nunique()} episodes", flush=True)
    d = pd.DataFrame(rows)
    d.to_parquet(OUT / "h16_wrist_visibility_frames.parquet")
    pd.set_option("display.width", 220)
    out = []
    for keys, g in list(d.groupby("object")) + list(d.groupby(["object", "set"])):
        keys = keys if isinstance(keys, tuple) else (keys, "(pooled)")
        row = dict(object=keys[0], set=keys[1], frames=len(g))
        for ph in ("home", "approach", "window", "after"):
            h = g[g.phase == ph]
            row[f"{ph}_share"] = len(h) / len(g)
            row[f"{ph}_seen"] = h.seen.mean() if len(h) else np.nan
        add = g[g.phase != "window"]  # frames full-episode dropout adds on top of fldrop30's window
        row["added_seen_share"] = add.seen.mean()
        row["added_preclose_share"] = add.phase.isin(["home", "approach"]).mean()
        row["added_preclose_seen"] = add[add.phase.isin(["home", "approach"])].seen.mean()
        out.append(row)
    sm = pd.DataFrame(out)
    sm.to_csv(OUT / "h16_wrist_visibility_summary.csv", index=False)
    print(sm.round(2).to_string(index=False))
    # approach timeline: seconds from motion onset to first 'seen', and seen share in the approach by time to close
    ap = d[d.phase.isin(["approach", "window"])].copy()
    ap["t_to_close"] = (ap.k - ap.f) / 15
    ap["bin"] = pd.cut(ap.t_to_close, [0, 2, 4, 6, 8, 12, 60], right=True)
    print("\nseen share before the close by time to close [s] (approach + window frames)")
    print(ap.groupby(["object", "bin"], observed=True).seen.agg(["mean", "size"]).round(2).unstack(0).to_string())
    # check sheet: per object, 3 frames per phase from random episodes, projected axis point drawn
    sheet = []
    for obj, label, name in SETS:
        root = Path(HC + name)
        eps = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(str(root / "meta/episodes/*/*.parquet")))]).set_index("episode_index")
        g = d[d.set == label]
        for ph in ("home", "approach", "window"):
            h = g[g.phase == ph]
            if h.empty:
                continue
            for _, rr in h.sample(min(2, len(h)), random_state=int(rng.integers(1e6))).iterrows():
                e = int(rr.ep); ci = int(eps.loc[e, "videos/observation.images.wrist/chunk_index"])
                fi = int(eps.loc[e, "videos/observation.images.wrist/file_index"]); t0 = float(eps.loc[e, "videos/observation.images.wrist/from_timestamp"])
                ts = t0 + rr.f / 15
                with av.open(str(root / f"videos/observation.images.wrist/chunk-{ci:03d}/file-{fi:03d}.mp4")) as c:
                    st = c.streams.video[0]; c.seek(int(max(0, ts - 1) / st.time_base), stream=st, backward=True)
                    img = None
                    for fr in c.decode(st):
                        if float(fr.pts * st.time_base) >= ts - 0.5 / 15:
                            img = fr.to_ndarray(format="bgr24"); break
                if img is None:
                    continue
                st_, ring_ = keep[(label, e)]
                uvp, insp = project(ring_, FM.link5(st_[int(rr.f)]) @ X5c)
                for u_, v_ in uvp[insp]:
                    cv2.circle(img, (int(u_), int(v_)), 2, (0, 0, 255), -1)
                cv2.putText(img, f"{label} ep{e} {ph} f{int(rr.f)} seen={rr.seen} {rr.frac_pts:.2f}", (6, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                sheet.append(cv2.resize(img, (320, 240)))
    while len(sheet) % 6:
        sheet.append(np.zeros((240, 320, 3), np.uint8))
    cv2.imwrite(str(OUT / "h16_wrist_visibility_check.jpg"), np.vstack([np.hstack(sheet[i:i + 6]) for i in range(0, len(sheet), 6)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"frames: {OUT / 'h16_wrist_visibility_frames.parquet'}\nsummary: {OUT / 'h16_wrist_visibility_summary.csv'}\ncheck sheet: {OUT / 'h16_wrist_visibility_check.jpg'}")


if __name__ == "__main__":
    main()
