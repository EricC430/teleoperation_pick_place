#!/usr/bin/env python
"""Contact sheets for checking aimfix cut frames without opening each episode in lerobot-dataset-viz.

One row per episode: front-left at cut-1.0 s, CUT, cut+0.6 s, just before close; wrist at CUT and just before close.
Reads the cut table written by scripts/aimfix_cut_candidates.py. A good cut: at CUT the arm has stopped at the
offset (fingers visibly off the bottle in the row's direction: 短 = short of it, 左/右 = to one side), the frame
before still shows approach motion, and the frames after it show the correction. Fix wrong rows in the CSV (cut_frame + source=manual) and re-run this to see them again.

    uv run python scripts/aimfix_cut_sheet.py                      # all episodes -> outputs/aimfix_cut_sheet/page_*.png
    uv run python scripts/aimfix_cut_sheet.py --episodes 0 4 17    # only these
"""
import argparse
from pathlib import Path

import av
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei", "DejaVu Sans"]
DEFAULT = "omx_pick_place_pilot_plastic_bottle_aimfix_A1"
# front-left first: at the offset the wrist view is mostly bottle, the side view shows where the fingers are.
COLS = [("front-left", -1.0, "cut-1.0s"), ("front-left", 0.0, "CUT"), ("front-left", 0.6, "cut+0.6s"),
        ("front-left", None, "閉爪前"), ("wrist", 0.0, "wrist @CUT"), ("wrist", None, "wrist 閉爪前")]


def grab(video, times):
    """{time_s: rgb array} for the frames nearest to each requested time in one mp4."""
    want = sorted(set(times))
    out, i = {}, 0
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            while i < len(want) and t >= want[i] - 0.5 / 15:
                out[want[i]] = fr.to_ndarray(format="rgb24")
                i += 1
            if i == len(want):
                break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=f".cache/lerobot/{DEFAULT}")
    ap.add_argument("--cuts", default=f"episode_meta/{DEFAULT}_cut.csv")
    ap.add_argument("--meta", default=f"episode_meta/{DEFAULT}.csv")
    ap.add_argument("--out", default="outputs/aimfix_cut_sheet")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--per_page", type=int, default=7)
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    root = Path(args.root)
    cuts = pd.read_csv(args.cuts, encoding="utf-8-sig")
    meta = pd.read_csv(args.meta, encoding="utf-8-sig") if Path(args.meta).exists() else None
    eps = pd.concat([pq.read_table(f).to_pandas() for f in sorted((root / "meta/episodes").glob("*/*.parquet"))])
    eps = eps.set_index("episode_index")
    if args.episodes:
        cuts = cuts[cuts.episode_index.isin(args.episodes)]
    cuts = cuts[cuts.cut_frame.notna()]

    # requested absolute times per video file
    req = {}
    plan = []
    for r in cuts.itertuples():
        e, cut, close = int(r.episode_index), int(r.cut_frame), int(r.close_frame)
        row = []
        for cam, dt, _ in COLS:
            f = close - 2 if dt is None else max(0, cut + round(dt * args.fps))
            key = f"videos/observation.images.{cam}"
            vid = root / key / f"chunk-{int(eps.loc[e, key + '/chunk_index']):03d}" / \
                f"file-{int(eps.loc[e, key + '/file_index']):03d}.mp4"
            t = round(float(eps.loc[e, key + "/from_timestamp"]) + f / args.fps, 4)
            req.setdefault(vid, []).append(t)
            row.append((vid, t, f))
        plan.append((e, cut, close, r.source, row))
    frames = {vid: grab(vid, ts) for vid, ts in req.items()}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in range(0, len(plan), args.per_page):
        chunk = plan[p: p + args.per_page]
        fig, axs = plt.subplots(len(chunk), len(COLS), figsize=(3.6 * len(COLS), 2.3 * len(chunk)), squeeze=False)
        for i, (e, cut, close, src, row) in enumerate(chunk):
            label = f"ep {e}"
            if meta is not None:
                m = meta[meta.episode_index == e].iloc[0]
                label += f" {m.placement_id} {str(m.notes).split('：')[1].split('（')[0] if '：' in str(m.notes) else ''}"
            for j, ((vid, t, f), (_, _, name)) in enumerate(zip(row, COLS)):
                ax = axs[i, j]
                ax.imshow(frames[vid][t])
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{name}  f{f}", fontsize=9, color="red" if name == "CUT" else "black")
                if name == "CUT":
                    for s in ax.spines.values():
                        s.set_edgecolor("red"); s.set_linewidth(3)
            axs[i, 0].set_ylabel(f"{label}\ncut {cut} ({src})\nclose {close}", fontsize=10, rotation=0, ha="right",
                                 va="center")
        fig.tight_layout()
        path = out / f"page_{p // args.per_page + 1}.png"
        fig.savefig(path, dpi=90)
        plt.close(fig)
        print(path)


if __name__ == "__main__":
    main()
