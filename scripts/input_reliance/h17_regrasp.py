"""H17: what happens after a missed grasp? First-try success, retries, and whether a retry moves toward the object;
models vs humans, M-full vs single-object, TE vs nas30 on the same model. Joint data only. (2026-11-12)

Why: M-full paper cup 29/36 vs 34/36; four of the extra failures (c12 c21 c26 c32) are Boyu's `repetition_loop` (same spot,
grasp again, no progress), Boyu: 'recovery 一塌糊塗'. h13-h16 ruled out the other candidates.

Close events: hysteresis on the gripper ACTION (lo = g0 - 0.6 depth, hi = g0 - 0.3 depth, g0 = first value, depth = g0 - min
>= 10): a close starts when the command goes below lo, ends when it comes back above hi. One successful grasp = 1 close
(the gripper opens at the bin). Check: human normal demos should give 1, recovery_A1_tight 2.
Per close: lateral aim = r_object * (theta_object - theta_arm) in cm, theta_arm from the base joint (h8 human cup fit,
linear in shoulder_pan; robust - h14). Between consecutive closes: lateral move (cm) and whether |lateral error| shrank by
> 0.5 cm ('toward'). Radial is not used (cup FK radius failed its check in h15).
Per run: first-try success (success with 1 close), missed first try (>= 2 closes, or failure), recovered (success with
>= 2 closes), P(recovered | >= 2 closes), median lateral move per retry, share of retries 'toward'.
Reading rules, fixed before the run:
  M-full vs 10/16 augfl (cup): first-try rates similar but P(recovered | retry) lower for M-full -> the gap is recovery;
  first-try lower and recovery similar -> the gap is the first attempt.
  TE vs nas30 on the same model (10/06 vs 10/07 b1_uvc60; 10/13 ntcfA1aug c1-c13): retries under TE clearly smaller lateral
  moves (median less than half) -> supports 'TE makes the policy grasp again at the same spot'.
Caveats: small n per run; mechanism labels differ by day (repetition_loop only on 11-11); TE and nas30 runs are on
different days; lateral only. Writes h17_*.csv and a wrist sheet (each close of the 11-11 M-full cup loop episodes and
of 10/16 augfl at the same placements) in data/logs/input_reliance.

    uv run python scripts/input_reliance/h17_regrasp.py
"""
import glob
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "scripts/input_reliance")
sys.path.insert(0, "scripts")
import h14_approach_correction as P14  # noqa: E402

H = P14.H
R, HC = H.ROOT, ".cache/lerobot/"
FPS = 15
SETS = [  # object, group, label, inference, root, csv
    ("cup", "human", "uvc_60 (A2)", "-", HC + "omx_pick_place_pilot_uvc_60", "omx_pick_place_pilot_uvc_60.csv"),
    ("cup", "human", "normal_A1", "-", HC + "omx_pick_place_pilot_paper_cup_normal_A1", "omx_pick_place_pilot_paper_cup_normal_A1.csv"),
    ("cup", "human", "recovery_A1_tight", "-", HC + "omx_pick_place_pilot_paper_cup_recovery_A1_tight", "omx_pick_place_pilot_paper_cup_recovery_A1_tight.csv"),
] + [("cup", "single", rl, "nas30", R + d, csv) for rl, d, csv in H.ROLLOUTS] + [
    ("cup", "single", "10-07 b1_uvc60 TE", "te+0.01", R + "rollout_omx_b1_uvc60_100k_te001_A1_paper_cup_20261007_205814",
     "rollout_omx_b1_uvc60_100k_A1_temperal_ensemble_paper_cup_20261007_close_loop.csv"),
    ("cup", "single", "10-13 ntcfA1aug TE", "te-0.01", R + "rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_100k_nas1_te001_paper_cup_20261013_221009", None),
    ("cup", "mfull", "11-11 mfull", "te-0.01", R + "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_30s_paper_cup_20261111_163453", None),
    ("bottle", "human", "bottle normal_A1", "-", HC + "omx_pick_place_pilot_plastic_bottle_normal_A1", "omx_pick_place_pilot_plastic_bottle_normal_A1.csv"),
    ("bottle", "human", "bottle recovery_A1_tight", "-", HC + "omx_pick_place_pilot_plastic_bottle_recovery_A1_tight", "omx_pick_place_pilot_plastic_bottle_recovery_A1_tight.csv"),
    ("bottle", "single", "10-22 b136 TE", "te-0.01", R + "rollout_omx_b4a_plastic_bottle_136_augfl_fldrop30_100k_nas1_te-001_20261022_104207", None),
    ("bottle", "mfull", "11-11 mfull bottle", "te-0.01", R + "rollout_omx_mfull_cup129_bottle135_augfl_fldrop30_200k_nas1_te-001_plastic_bottle_20261111_175328", None),
]
for i, s in enumerate(SETS):  # h8.ROLLOUTS 10-13 ntcfA1aug is nas30; make the label say so
    if s[2] in ("10-06 b1_uvc60", "10-13 ntcfA1aug"):
        SETS[i] = s[:2] + (s[2] + " nas30",) + s[3:]
OUT = H.OUT
SHEET_EPS = {"11-11 mfull": ["c6", "c12", "c21", "c26", "c32"], "10-16 augfl": ["c12", "c21", "c26", "c32"]}


def closes(g):
    g0 = g[0]; depth = g0 - g.min()
    if depth < 10:
        return []
    lo, hi = g0 - 0.6 * depth, g0 - 0.3 * depth
    ev, closed = [], False
    for t in range(len(g)):
        if not closed and g[t] < lo:
            closed = True; ev.append(t)
        elif closed and g[t] > hi:
            closed = False
    return ev


def main() -> None:
    eps, retries = [], []
    sheet_src = {}
    for obj, grp, label, inf, root, csv in SETS:
        m = P14.meta(csv, root)
        t = pd.concat([pq.read_table(f, columns=["episode_index", "frame_index", "action", "observation.state"]).to_pandas()
                       for f in sorted(Path(root, "data").rglob("*.parquet"))])
        for r in m.itertuples():
            g = t[t.episode_index == r.episode_index].sort_values("frame_index")
            if g.empty:
                continue
            a = np.stack(g["action"].values); s = np.stack(g["observation.state"].values)
            ev = closes(a[:, 5])
            th = np.polyval(H.C[0], s[:, 0])
            lat = [r.r * np.deg2rad(r.th - th[k]) for k in ev]
            out = str(getattr(r, "outcome", "")); mech = str(getattr(r, "mechanism", ""))
            succ = out == "success"
            eps.append(dict(object=obj, group=grp, set=label, inference=inf, ep=r.episode_index, pid=r.placement_id,
                            outcome=out, mechanism=mech, loop="repetition_loop" in mech, n_close=len(ev),
                            t_close=" ".join(f"{k / FPS:.1f}" for k in ev), lat=" ".join(f"{x:+.1f}" for x in lat),
                            first_try=succ and len(ev) == 1, retried=len(ev) >= 2, recovered=succ and len(ev) >= 2))
            for i in range(1, len(ev)):
                retries.append(dict(object=obj, group=grp, set=label, inference=inf, ep=r.episode_index, pid=r.placement_id,
                                    outcome=out, i=i, dt=(ev[i] - ev[i - 1]) / FPS, lat_prev=lat[i - 1], lat_now=lat[i],
                                    move=abs(lat[i] - lat[i - 1]), toward=abs(lat[i]) < abs(lat[i - 1]) - 0.5))
            if label in SHEET_EPS and r.placement_id in SHEET_EPS[label]:
                sheet_src[(label, r.placement_id)] = (root, int(r.episode_index), ev, out, lat)
        print(f"{obj:6s} {label:24s}: {sum(1 for e in eps if e['set'] == label)} episodes", flush=True)
    E, Rt = pd.DataFrame(eps), pd.DataFrame(retries)
    E.to_csv(OUT / "h17_regrasp_episodes.csv", index=False); Rt.to_csv(OUT / "h17_regrasp_retries.csv", index=False)
    rows = []
    for (obj, label), g in E.groupby(["object", "set"], sort=False):
        rr = Rt[Rt.set == label] if len(Rt) else Rt
        rows.append(dict(object=obj, set=label, inference=g.inference.iloc[0], n=len(g), success=int((g.outcome == "success").sum()),
                         first_try=int(g.first_try.sum()), retried=int(g.retried.sum()), recovered=int(g.recovered.sum()),
                         fail_1close=int(((g.outcome != "success") & (g.n_close == 1)).sum()), fail_0close=int((g.n_close == 0).sum()),
                         loops=int(g.loop.sum()), n_retries=len(rr),
                         med_move_cm=rr.move.median() if len(rr) else np.nan, toward_share=rr.toward.mean() if len(rr) else np.nan,
                         med_dt_s=rr.dt.median() if len(rr) else np.nan))
    S = pd.DataFrame(rows)
    S["p_recover"] = S.recovered / S.retried.replace(0, np.nan)
    S.to_csv(OUT / "h17_regrasp_summary.csv", index=False)
    pd.set_option("display.width", 240)
    print(S.round(2).to_string(index=False))
    print("\nper placement, loop / retried failures of the 11-11 M-full cup run and 10-16 augfl at the same points:")
    for label in ("11-11 mfull", "10-16 augfl"):
        print(E[(E.set == label) & (E.pid.isin(SHEET_EPS["11-11 mfull"]) | (E.set == label) & E.retried)][
            ["set", "pid", "outcome", "mechanism", "n_close", "t_close", "lat"]].to_string(index=False))
    # wrist sheet: each close (and 1 s before the first) of the chosen episodes
    tiles = []
    for (label, pid), (root, e, ev, out, lat) in sheet_src.items():
        epm = pd.concat([pq.read_table(f).to_pandas() for f in sorted(glob.glob(str(Path(root) / "meta/episodes/*/*.parquet")))]).set_index("episode_index")
        ci = int(epm.loc[e, "videos/observation.images.wrist/chunk_index"]); fi = int(epm.loc[e, "videos/observation.images.wrist/file_index"])
        t0 = float(epm.loc[e, "videos/observation.images.wrist/from_timestamp"])
        want = ([max(ev[0] - FPS, 0)] if ev else []) + ev[:5]
        imgs = {}
        with av.open(str(Path(root) / f"videos/observation.images.wrist/chunk-{ci:03d}/file-{fi:03d}.mp4")) as c:
            st = c.streams.video[0]; c.seek(int(max(0, t0 - 1) / st.time_base), stream=st, backward=True)
            targets = sorted(want)
            for fr in c.decode(st):
                ts = float(fr.pts * st.time_base) - t0
                while targets and ts >= targets[0] / FPS - 0.5 / FPS:
                    imgs[targets.pop(0)] = fr.to_ndarray(format="bgr24")
                if not targets:
                    break
        row = []
        for j, k in enumerate(want):
            im = cv2.resize(imgs.get(k, np.zeros((480, 640, 3), np.uint8)), (320, 240))
            tag = "1 s before" if j == 0 else f"close {j} lat {lat[j - 1]:+.1f}"
            cv2.putText(im, f"{label} {pid} {out}", (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(im, f"{tag} t={k / FPS:.1f}s", (4, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            row.append(im)
        while len(row) < 6:
            row.append(np.zeros((240, 320, 3), np.uint8))
        tiles.append(np.hstack(row[:6]))
    if tiles:
        cv2.imwrite(str(OUT / "h17_regrasp_wrist.jpg"), np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"episodes: {OUT / 'h17_regrasp_episodes.csv'}\nsummary: {OUT / 'h17_regrasp_summary.csv'}\nwrist sheet: {OUT / 'h17_regrasp_wrist.jpg'}")


if __name__ == "__main__":
    main()
