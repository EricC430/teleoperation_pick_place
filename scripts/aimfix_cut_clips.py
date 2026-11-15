#!/usr/bin/env python
"""One short mp4 per aimfix episode with the frame number burned in, to pick cut frames in any video player.

front-left and wrist side by side, from --before s before the candidate cut to --after s after the close
(cut table from scripts/aimfix_cut_candidates.py). The top-left text is the episode's frame_index = the number to
write into cut_frame; the candidate cut is shown in red for +-2 frames. Boyu 2026-11-15: watching video is faster
than a contact sheet or lerobot-dataset-viz per episode.

    uv run python scripts/aimfix_cut_clips.py                     # -> outputs/aimfix_cut_clips/ep00_t17_短.mp4 ...
    uv run python scripts/aimfix_cut_clips.py --episodes 0 4 --before 8
"""
import argparse
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont

DEFAULT = "omx_pick_place_pilot_plastic_bottle_aimfix_A1"
CAMS = ("front-left", "wrist")
H = 360  # output height of each view


def frames(video, t0, t1):
    out = []
    with av.open(str(video)) as c:
        s = c.streams.video[0]
        c.seek(int(max(0.0, t0 - 1.0) / s.time_base), stream=s, backward=True)
        for fr in c.decode(s):
            t = float(fr.pts * s.time_base)
            if t < t0 - 0.5 / 15:
                continue
            if t > t1 + 0.5 / 15:
                break
            out.append(fr.to_image())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=f".cache/lerobot/{DEFAULT}")
    ap.add_argument("--cuts", default=f"episode_meta/{DEFAULT}_cut.csv")
    ap.add_argument("--meta", default=f"episode_meta/{DEFAULT}.csv")
    ap.add_argument("--out", default="outputs/aimfix_cut_clips")
    ap.add_argument("--episodes", type=int, nargs="*")
    ap.add_argument("--before", type=float, default=5.0, help="s before the candidate cut")
    ap.add_argument("--after", type=float, default=1.0, help="s after the close")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()

    root = Path(args.root)
    cuts = pd.read_csv(args.cuts, encoding="utf-8-sig")
    meta = pd.read_csv(args.meta, encoding="utf-8-sig") if Path(args.meta).exists() else None
    eps = pd.concat([pq.read_table(f).to_pandas() for f in sorted((root / "meta/episodes").glob("*/*.parquet"))])
    eps = eps.set_index("episode_index")
    if args.episodes:
        cuts = cuts[cuts.episode_index.isin(args.episodes)]
    try:
        font = ImageFont.truetype("msjh.ttc", 26)
    except OSError:
        font = ImageFont.load_default()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for r in cuts.itertuples():
        e = int(r.episode_index)
        if pd.isna(r.cut_frame):
            print(f"ep {e}: no cut candidate, skipped")
            continue
        cut, close = int(r.cut_frame), int(r.close_frame)
        n_ep = int(eps.loc[e, "length"])
        f0, f1 = max(0, cut - round(args.before * args.fps)), min(n_ep - 1, close + round(args.after * args.fps))
        views = []
        for cam in CAMS:
            key = f"videos/observation.images.{cam}"
            vid = root / key / f"chunk-{int(eps.loc[e, key + '/chunk_index']):03d}" / \
                f"file-{int(eps.loc[e, key + '/file_index']):03d}.mp4"
            base = float(eps.loc[e, key + "/from_timestamp"])
            views.append(frames(vid, base + f0 / args.fps, base + f1 / args.fps))
        n = min(len(v) for v in views)
        label, pid, d = f"ep{e:02d}", "", ""
        if meta is not None:
            m = meta[meta.episode_index == e].iloc[0]
            pid = m.placement_id
            d = str(m.notes).split("：")[1][:1] if "：" in str(m.notes) else ""
        path = out_dir / f"{label}_{pid}_{d}.mp4"
        with av.open(str(path), "w") as oc:
            widths = [round(v[0].width * H / v[0].height) for v in views]
            st = oc.add_stream("libx264", rate=args.fps)
            st.width, st.height, st.pix_fmt = sum(widths) + (sum(widths) % 2), H, "yuv420p"
            for i in range(n):
                f = f0 + i
                canvas = Image.new("RGB", (st.width, H))
                x = 0
                for v, w in zip(views, widths):
                    canvas.paste(v[i].resize((w, H)), (x, 0))
                    x += w
                dr = ImageDraw.Draw(canvas)
                is_cut = abs(f - cut) <= 2
                txt = f"{label} {pid} {d}   frame {f}" + ("   ← 候選剪點" if is_cut else "") + \
                    ("   閉爪" if abs(f - close) <= 1 else "")
                dr.rectangle((0, 0, st.width, 36), fill=(0, 0, 0))
                dr.text((8, 3), txt, font=font, fill=(255, 60, 60) if is_cut else (255, 255, 255))
                for pkt in st.encode(av.VideoFrame.from_ndarray(np.asarray(canvas), format="rgb24")):
                    oc.mux(pkt)
            for pkt in st.encode():
                oc.mux(pkt)
        print(f"{path}  frames {f0}-{f1}  (cut {cut}, close {close})")


if __name__ == "__main__":
    main()
