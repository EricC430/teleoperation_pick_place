#!/usr/bin/env python
"""Candidate cut frames for "offset then correct" demos (aimfix), to be checked by eye and then trimmed.

An aimfix episode is: approach -> stop 1-2 cm off on purpose -> correct while watching the wrist view -> close.
The approach part teaches the policy to aim wrong (same reason as scripts/trim_recovery_prefix.py), so each episode
should start at the offset. The offset cannot be told from "centred, about to close" by joints alone, so this only
PROPOSES a frame; the operator confirms or corrects it in lerobot-dataset-viz (2026-11-15, Boyu: a pause is not
required as long as the cut frame is marked).

Rule (leader action, arm joints 0-4, degrees; gripper = action[:, 5]):
  close  = LAST downward crossing of the gripper's mid-range (the grasp that lifts)
  trial  = if there is an earlier crossing, the operator closed at the offset first (every 短 episode on 11/15):
           cut = first frame the gripper is within 0.5 of its lowest value between the two closes (the trial close
           is complete, closed on nothing), source=trial_min [Boyu 2026-11-15: 短 all cut at the trial-close minimum]
  ball  = earliest frame t such that every frame from t to close is within --ball deg (L1 over the 5 arm joints)
           of the close pose -> the arm has arrived near the object and only small moves remain
  pause  = a still run (max joint step < --still deg/frame, 3-frame median, >= --min_pause s) that starts
           at or after ball - 1 s; if there is one, cut = its LAST frame (the correction starts right after)
  cut    = pause end if found, else ball
Writes episode_index, cut_frame, cut_s, close_frame, close_s, source (pause/ball/trial_min), note. Existing rows
with source=manual are kept (cut_s is recomputed from cut_frame) unless listed in --redo. A note the operator typed
on a non-manual row (any non-ASCII text) is kept after the new note.

    uv run python scripts/aimfix_cut_candidates.py
    uv run python scripts/aimfix_cut_candidates.py --root .cache/lerobot/<dataset> --out episode_meta/<dataset>_cut.csv
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

DEFAULT = "omx_pick_place_pilot_plastic_bottle_aimfix_A1"


def close_events(g):
    """Frames where the gripper action crosses down through the midpoint of its range."""
    thr = (g.max() + g.min()) / 2
    return (np.where((g[:-1] >= thr) & (g[1:] < thr))[0] + 1).tolist()


def candidate(a, fps, ball_deg, still_deg, min_pause_s):
    ev = close_events(a[:, 5])
    if not ev:
        return None
    k = ev[-1]
    if len(ev) >= 2:
        # trial close at the offset, then reopen and correct (all 短 episodes did this). Keep from the moment the
        # trial close is complete (gripper at its lowest = closed on nothing), not the closing motion itself, which
        # would teach "close when short" (Boyu 2026-11-15; same idea as trim_recovery_prefix keeping the empty close).
        g = a[ev[0]: ev[-1], 5]
        cut = ev[0] + int(np.argmax(g <= g.min() + 0.5))
        return dict(cut_frame=cut, close_frame=k, source="trial_min",
                    note=f"trial close from {ev[0]}, lowest gripper {g.min():.1f} at {cut}")
    dist = np.abs(a[: k + 1, :5] - a[k, :5]).sum(1)
    outside = np.where(dist > ball_deg)[0]
    ball = int(outside[-1] + 1) if len(outside) else 0
    v = np.r_[0, np.abs(np.diff(a[:, :5], axis=0)).max(1)]
    v = pd.Series(v).rolling(3, center=True, min_periods=1).median().to_numpy()
    lo, i, pause = max(0, ball - fps), None, None
    i = lo
    while i < k:
        if v[i] < still_deg:
            j = i
            while j < k and v[j] < still_deg:
                j += 1
            if j - i >= min_pause_s * fps and j < k - 1:
                pause = (i, j - 1)
                break  # first qualifying pause near arrival = the offset stop
            i = j
        else:
            i += 1
    cut, src = (pause[1], "pause") if pause else (ball, "ball")
    return dict(cut_frame=cut, close_frame=k, source=src,
                note=f"pause {pause[0]}-{pause[1]}" if pause else "no pause near arrival")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=f".cache/lerobot/{DEFAULT}")
    ap.add_argument("--out", default=f"episode_meta/{DEFAULT}_cut.csv")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--ball", type=float, default=12.0, help="deg, L1 over 5 arm joints, from the close pose")
    ap.add_argument("--still", type=float, default=0.3, help="deg/frame, max over 5 arm joints")
    ap.add_argument("--min_pause", type=float, default=0.5, help="s")
    ap.add_argument("--redo", type=int, nargs="*", default=[], help="recompute these even if source=manual")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.root}/data/*/*.parquet"))
    if not files:
        raise SystemExit(f"no parquet under {args.root}/data")
    t = pd.concat([pq.read_table(f, columns=["episode_index", "frame_index", "action"]).to_pandas() for f in files])
    out = Path(args.out)
    manual, typed = {}, {}
    if out.exists():
        old = pd.read_csv(out, encoding="utf-8-sig", dtype=str).fillna("")
        for r in old.to_dict("records"):
            e = int(r["episode_index"])
            if r["source"] == "manual" and e not in args.redo:
                manual[e] = r
            elif any(ord(ch) > 127 for ch in r["note"]):
                typed[e] = r["note"]

    rows = []
    for e, g in t.groupby("episode_index"):
        if e in manual:
            r = dict(manual[e])
            try:
                r["cut_s"] = round(int(r["cut_frame"]) / args.fps, 2)
            except ValueError:
                print(f"ep {e}: cut_frame {r['cut_frame']!r} is not a frame number - fix it")
            rows.append(r)
            continue
        a = np.stack(g.sort_values("frame_index")["action"].to_numpy())
        c = candidate(a, args.fps, args.ball, args.still, args.min_pause)
        if c is None:
            rows.append(dict(episode_index=e, cut_frame="", cut_s="", close_frame="", close_s="", source="none",
                             note="no close found"))
            continue
        rows.append(dict(episode_index=e, cut_frame=c["cut_frame"], cut_s=round(c["cut_frame"] / args.fps, 2),
                         close_frame=c["close_frame"], close_s=round(c["close_frame"] / args.fps, 2),
                         source=c["source"], note=c["note"] + (f" | {typed[e]}" if e in typed else "")))
    df = pd.DataFrame(rows)[["episode_index", "cut_frame", "cut_s", "close_frame", "close_s", "source", "note"]]
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))
    print(f"\nwrote {out}  (kept {len(manual)} manual rows)")
    print("Check each in lerobot-dataset-viz; when you fix one, set cut_frame and source=manual.")


if __name__ == "__main__":
    main()
