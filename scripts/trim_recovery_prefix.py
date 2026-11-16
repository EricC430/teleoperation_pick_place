#!/usr/bin/env python
"""Save a copy of a recovery dataset with each episode's deliberate-miss approach cut off.

A recovery demo is "drive off on purpose -> close on nothing -> notice -> release -> redo".
The first part teaches the policy to aim wrong, so this keeps each episode from the onset of its
FIRST gripper close (the empty one) onwards. The empty-close state itself is kept: that is the
state recovery is supposed to start from (docs/worklog/2026-10-07.md §118, 2026-10-09 discussion).

The source dataset is not touched. The copy shares the source's video bytes exactly: the mp4 files
are copied unchanged and only the per-episode `from_timestamp` moves forward, so no frame is
re-encoded. Data parquet rows before the cut are dropped and frame_index / timestamp / index are
renumbered; per-episode stats (numeric and image) and meta/stats.json are recomputed for the kept
frames. Image stats follow lerobot's compute_episode_stats (same sampling, same downsampling).

Cut rule (gripper = action[:, 5], smaller = more closed):
  first close = first event of the close-event definition in scripts/analyze_rcvry_closed_loop_20261008.py
                (below the midpoint of the 2nd..98th percentile, debounced 0.2 s)
  cut         = walk back from it while the gripper is more than --open_tol below its value at frame 0
Episodes without a close event, or listed in --keep_whole, are copied whole.

--cuts <csv>: take each episode's cut from a hand-checked table (episode_index, cut_frame) instead of the rule
above - used for "offset then correct" demos (aimfix, 2026-11-15), where the cut is at the offset, picked by eye:
    uv run python scripts/trim_recovery_prefix.py \
        --src .cache/lerobot/omx_pick_place_pilot_plastic_bottle_aimfix_A1 \
        --dst .cache/lerobot/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut \
        --cuts episode_meta/omx_pick_place_pilot_plastic_bottle_aimfix_A1_cut.csv

    ./scripts/run_container.sh python scripts/trim_recovery_prefix.py \
        --src data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_recovery_A1_tight \
        --dst data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_recovery_A1_tight_cut

The cut table may also carry `end_frame` (inclusive; blank = to the end of the episode), and the same episode_index
may appear on more than one row: the first row keeps that episode's index, each further row becomes a NEW episode
appended after the last source episode (numbered in CSV order), sharing the same video bytes. Used to split an
"overshoot then correct back" demo into two (aimfix ep4 / ep16, 2026-11-16): [cut .. first correction reached] and
[end of overshoot .. end].

The cut table is written to <dst>/meta/trim_recovery_prefix.json (keyed by OUTPUT episode, with src_episode).
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from lerobot.datasets.compute_stats import (
    aggregate_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
from lerobot.datasets.io_utils import write_stats
from lerobot.datasets.video_utils import decode_video_frames

GRIPPER = 5


def close_events(g, fps, close_below=None):
    """Copied from scripts/analyze_rcvry_closed_loop_20261008.py (that file runs its analysis on import).
    close_below: fixed threshold instead of the percentile midpoint (bottle demos open to ~85 before grasping,
    so the midpoint lands above the resting ~59 and frame 0 counts as closed; 2026-10-21)."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    closed = g < ((lo + hi) / 2 if close_below is None else close_below)
    k = max(1, int(0.2 * fps))
    runs, cur, n = [], closed[0], 0
    for c in closed:
        if c == cur:
            n += 1
        else:
            runs.append((cur, n)); cur, n = c, 1
    runs.append((cur, n))
    merged = []
    for c, n in runs:
        if merged and (n < k or merged[-1][0] == c):
            merged[-1] = (merged[-1][0], merged[-1][1] + n)
        else:
            merged.append((c, n))
    t, out = 0, []
    for c, n in merged:
        if c:
            out.append((t, n))  # (start frame, hold frames)
        t += n
    return out


def cut_frame(g, fps, open_tol, close_below=None):
    ev = close_events(g, fps, close_below)
    if not ev:
        return None, None
    first = ev[0][0]
    o = first
    while o > 0 and g[o - 1] < g[0] - open_tol:
        o -= 1
    return o, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--open_tol", type=float, default=1.0, help="gripper drop below frame 0 that counts as closing")
    ap.add_argument("--keep_whole", type=int, nargs="*", default=[], help="episodes to copy uncut")
    ap.add_argument("--close_below", type=float, default=None,
                    help="fixed gripper threshold for a close (default: midpoint of the 2nd..98th percentile)")
    ap.add_argument("--cuts", default=None,
                    help="CSV with episode_index,cut_frame (e.g. episode_meta/<dataset>_cut.csv from "
                         "scripts/aimfix_cut_candidates.py, checked by eye): cut there instead of the close rule")
    args = ap.parse_args()
    given = None  # src episode -> [(cut, end_inclusive or None), ...] in CSV order
    if args.cuts:
        cdf = pd.read_csv(args.cuts, encoding="utf-8-sig")
        given = {}
        for _, r in cdf.iterrows():
            end = None if "end_frame" not in cdf.columns or pd.isna(r.end_frame) else int(r.end_frame)
            given.setdefault(int(r.episode_index), []).append((int(r.cut_frame), end))
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache"))

    info = json.loads((dst / "meta/info.json").read_text())
    fps, features = info["fps"], info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    def chunk_file(p):  # .../chunk-000/file-001.parquet -> (0, 1)
        return int(p.parent.name.split("-")[1]), int(p.stem.split("-")[1])

    # --- plan the output episodes: (cut, end) per row; extra rows of one episode become new episodes ---
    data_files = sorted((dst / "data").glob("*/*.parquet"))
    tables = {p: pq.read_table(p) for p in data_files}
    src_rows, src_file = {}, {}
    for p, t in tables.items():
        df = t.to_pandas()
        for ep in sorted(df.episode_index.unique()):
            src_rows[int(ep)] = df[df.episode_index == ep].sort_values("frame_index")
            src_file[int(ep)] = p
    n_src = len(src_rows)
    if given is not None and set(given) - set(src_rows):
        raise SystemExit(f"{args.cuts} has episodes not in the source: {sorted(set(given) - set(src_rows))}")
    cuts, extras = {}, []  # extras: output episode numbers appended after the source ones
    for ep, x in src_rows.items():
        g = np.stack(x["action"].values)[:, GRIPPER]
        if ep in args.keep_whole:
            segs, first = [(0, None)], None
        elif given is not None:
            if ep not in given:
                raise SystemExit(f"episode {ep} missing from {args.cuts}")
            segs, first = given[ep], None
        else:
            c, first = cut_frame(g, fps, args.open_tol, args.close_below)
            segs = [(int(c or 0), None)]
        for k, (c, e) in enumerate(segs):
            e = len(x) if e is None else e + 1  # exclusive
            if not 0 <= c < e <= len(x):
                raise SystemExit(f"episode {ep}: segment {c}..{e - 1} outside 0..{len(x) - 1} or empty")
            out = ep if k == 0 else n_src + len(extras)
            if k:
                extras.append(out)
            cuts[out] = dict(src_episode=ep, cut=int(c), end=int(e), first_close=None if first is None else int(first),
                             orig_len=len(x), new_len=int(e - c))

    # --- data: keep [cut, end) of each output episode, renumber; extras go at the end of the last data file ---
    next_index, kept = 0, {}
    for p in data_files:
        outs = [ep for ep in sorted(src_rows) if src_file[ep] == p] + (extras if p == data_files[-1] else [])
        parts = []
        for out in outs:
            c = cuts[out]
            x = src_rows[c["src_episode"]]
            x = x[(x.frame_index >= c["cut"]) & (x.frame_index < c["end"])].copy()
            x["episode_index"] = np.int64(out)
            x["frame_index"] = np.arange(len(x), dtype=np.int64)
            x["timestamp"] = (x["frame_index"] / fps).astype(np.float32)
            x["index"] = np.arange(next_index, next_index + len(x), dtype=np.int64)
            c["from_index"], next_index = next_index, next_index + len(x)
            c["data_file"] = chunk_file(p)
            kept[out] = x
            parts.append(x)
        new = pd.concat(parts)
        for col in ("action", "observation.state"):
            new[col] = [list(map(float, v)) for v in new[col]]
        pq.write_table(pa.Table.from_pandas(new, schema=tables[p].schema, preserve_index=False), p)

    # --- episode meta: lengths, indices, video offsets, per-episode stats; extras appended to the last meta file ---
    meta_files = sorted((dst / "meta/episodes").glob("*/*.parquet"))
    metas = {p: (pq.read_table(p), pq.read_table(p).to_pandas()) for p in meta_files}
    src_meta = {int(r.episode_index): (m, i) for _, m in metas.values() for i, r in m.iterrows()}
    all_stats = []
    for p in meta_files:
        t, m = metas[p]
        if p == meta_files[-1] and extras:
            m = pd.concat([m] + [src_meta[cuts[out]["src_episode"]][0].loc[[src_meta[cuts[out]["src_episode"]][1]]]
                                 for out in extras], ignore_index=True)
            m.loc[len(m) - len(extras):, "episode_index"] = extras
        for i, row in m.iterrows():
            ep = int(row.episode_index)
            c, x = cuts[ep], kept[ep]
            src_row = src_meta[c["src_episode"]][0].loc[src_meta[c["src_episode"]][1]]
            m.at[i, "length"] = c["new_len"]
            m.at[i, "dataset_from_index"] = c["from_index"]
            m.at[i, "dataset_to_index"] = c["from_index"] + c["new_len"]
            m.at[i, "data/chunk_index"], m.at[i, "data/file_index"] = c["data_file"]
            m.at[i, "meta/episodes/chunk_index"], m.at[i, "meta/episodes/file_index"] = chunk_file(p)
            for vk in video_keys:
                m.at[i, f"videos/{vk}/from_timestamp"] = src_row[f"videos/{vk}/from_timestamp"] + c["cut"] / fps
                m.at[i, f"videos/{vk}/to_timestamp"] = src_row[f"videos/{vk}/from_timestamp"] + c["end"] / fps
            ep_stats = {}
            for key in ("action", "observation.state", "timestamp", "frame_index", "episode_index", "index", "task_index"):
                arr = np.stack(x[key].values) if key in ("action", "observation.state") else x[key].to_numpy()
                ep_stats[key] = get_feature_stats(arr, axis=0, keepdims=arr.ndim == 1)
            for vk in video_keys:
                vpath = dst / info["video_path"].format(
                    video_key=vk, chunk_index=int(row[f"videos/{vk}/chunk_index"]),
                    file_index=int(row[f"videos/{vk}/file_index"]))
                from_ts = m.at[i, f"videos/{vk}/from_timestamp"]
                ts = [float(from_ts + j / fps) for j in sample_indices(c["new_len"])]
                frames = decode_video_frames(vpath, ts, tolerance_s=1e-4, return_uint8=True).numpy()
                imgs = np.stack([auto_downsample_height_width(f) for f in frames])
                s = get_feature_stats(imgs, axis=(0, 2, 3), keepdims=True)
                ep_stats[vk] = {k: v if k == "count" else np.squeeze(v / 255.0, axis=0) for k, v in s.items()}
            for key, st in ep_stats.items():
                for name, val in st.items():
                    col = f"stats/{key}/{name}"
                    if col in m.columns:
                        m.at[i, col] = np.asarray(val).tolist()  # image stats (3, 1, 1), vectors (6,), scalars (1,)
            all_stats.append(ep_stats)
        pq.write_table(pa.Table.from_pandas(m, schema=t.schema, preserve_index=False), p)

    info["total_frames"] = next_index
    info["total_episodes"] = len(cuts)
    info["splits"] = {"train": f"0:{len(cuts)}"}
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    write_stats(aggregate_stats(all_stats), dst)
    (dst / "meta/trim_recovery_prefix.json").write_text(json.dumps(
        dict(src=str(src), cuts_csv=args.cuts, open_tol=args.open_tol, close_below=args.close_below,
             keep_whole=args.keep_whole, fps=fps, episodes=cuts), indent=1))
    for ep in sorted(cuts):
        c = cuts[ep]
        print(f"ep{ep:3d} <- src ep{c['src_episode']:3d} [{c['cut']:4d}, {c['end']:4d})  first_close@{c['first_close']}  "
              f"{c['orig_len']} -> {c['new_len']}")
    print(f"total frames {sum(len(x) for x in src_rows.values())} -> {next_index}, episodes {n_src} -> {len(cuts)}")


if __name__ == "__main__":
    main()
