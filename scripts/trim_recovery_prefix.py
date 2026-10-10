#!/usr/bin/env python
"""Save a copy of a recovery dataset with each episode's deliberate-miss approach cut off.

A recovery demo is "drive off on purpose -> close on nothing -> notice -> release -> redo".
The first part teaches the policy to aim wrong, so this keeps each episode from the onset of its
FIRST gripper close (the empty one) onwards. The empty-close state itself is kept: that is the
state recovery is supposed to start from (docs/meeting/2026-10-07.md §118, 2026-10-09 discussion).

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

    ./scripts/run_container.sh python scripts/trim_recovery_prefix.py \
        --src data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_recovery_A1_tight \
        --dst data/huggingface/lerobot/ericc430/omx_pick_place_pilot_paper_cup_recovery_A1_tight_cut

The cut table is written to <dst>/meta/trim_recovery_prefix.json.
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


def close_events(g, fps):
    """Copied from scripts/analyze_rcvry_closed_loop_20261008.py (that file runs its analysis on import)."""
    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    closed = g < (lo + hi) / 2
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


def cut_frame(g, fps, open_tol):
    ev = close_events(g, fps)
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
    args = ap.parse_args()
    src, dst = Path(args.src), Path(args.dst)
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns(".cache"))

    info = json.loads((dst / "meta/info.json").read_text())
    fps, features = info["fps"], info["features"]
    video_keys = [k for k, v in features.items() if v["dtype"] == "video"]

    # --- data: drop rows before the cut, renumber ---
    data_files = sorted((dst / "data").glob("*/*.parquet"))
    tables = {p: pq.read_table(p) for p in data_files}
    cuts, kept = {}, {}
    for p, t in tables.items():
        df = t.to_pandas()
        for ep in sorted(df.episode_index.unique()):
            x = df[df.episode_index == ep].sort_values("frame_index")
            g = np.stack(x["action"].values)[:, GRIPPER]
            c, first = (None, None) if ep in args.keep_whole else cut_frame(g, fps, args.open_tol)
            cuts[int(ep)] = dict(cut=int(c or 0), first_close=None if first is None else int(first),
                                 orig_len=len(x), new_len=len(x) - int(c or 0))
    next_index = 0
    for p in data_files:
        df = tables[p].to_pandas()
        parts = []
        for ep in sorted(df.episode_index.unique()):
            x = df[df.episode_index == ep].sort_values("frame_index")
            x = x[x.frame_index >= cuts[int(ep)]["cut"]].copy()
            x["frame_index"] = np.arange(len(x), dtype=np.int64)
            x["timestamp"] = (x["frame_index"] / fps).astype(np.float32)
            x["index"] = np.arange(next_index, next_index + len(x), dtype=np.int64)
            cuts[int(ep)]["from_index"], next_index = next_index, next_index + len(x)
            kept[int(ep)] = x
            parts.append(x)
        new = pd.concat(parts)
        for col in ("action", "observation.state"):
            new[col] = [list(map(float, v)) for v in new[col]]
        pq.write_table(pa.Table.from_pandas(new, schema=tables[p].schema, preserve_index=False), p)

    # --- episode meta: lengths, indices, video offsets, per-episode stats ---
    all_stats = []
    for p in sorted((dst / "meta/episodes").glob("*/*.parquet")):
        t = pq.read_table(p)
        m = t.to_pandas()
        for i, row in m.iterrows():
            ep = int(row.episode_index)
            c, x = cuts[ep], kept[ep]
            m.at[i, "length"] = c["new_len"]
            m.at[i, "dataset_from_index"] = c["from_index"]
            m.at[i, "dataset_to_index"] = c["from_index"] + c["new_len"]
            shift = c["cut"] / fps
            for vk in video_keys:
                m.at[i, f"videos/{vk}/from_timestamp"] = row[f"videos/{vk}/from_timestamp"] + shift
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
    (dst / "meta/info.json").write_text(json.dumps(info, indent=4))
    write_stats(aggregate_stats(all_stats), dst)
    (dst / "meta/trim_recovery_prefix.json").write_text(json.dumps(
        dict(src=str(src), open_tol=args.open_tol, keep_whole=args.keep_whole, fps=fps, episodes=cuts), indent=1))
    for ep, c in cuts.items():
        print(f"ep{ep:3d} cut@{c['cut']:4d} ({c['cut'] / fps:5.1f}s)  first_close@{c['first_close']}  "
              f"{c['orig_len']} -> {c['new_len']}")
    print(f"total frames {sum(c['orig_len'] for c in cuts.values())} -> {next_index}")


if __name__ == "__main__":
    main()
