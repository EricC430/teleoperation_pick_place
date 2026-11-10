#!/usr/bin/env python
"""Real front-left vs Isaac replays of the same episode rendered with DIFFERENT front-left extrinsics, as one video.

Why (2026-11-05): to compare two camera calibrations (e.g. the 10-14 ArUco one vs a CAD-outline fit on the session's
own video) where it matters -- on the replayed episode. Each replay is a sim/physics_replay.py render=1 output
(sim_front-left_<k>.jpg, already distorted like the real lens). Per frame: top row the real frame and each sim frame;
bottom row the real frame blended 50/50 with each sim frame, so an offset shows as a double image.

    python3 scripts/compose_replay_ext_video.py <dataset dir> <episode> <out.mp4> <label>=<render dir> [<label>=<render dir> ...]
        [--scale 0.5] [--every 1]
"""
from __future__ import annotations

import argparse
import glob
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

CAM = "front-left"


def real_frames(ds: Path, ep: int):
    meta = {}
    for f in sorted(glob.glob(str(ds / "meta/episodes/*/*.parquet"))):
        for k, v in pq.read_table(f).to_pydict().items():
            meta.setdefault(k, []).extend(v)
    n = meta["episode_index"].index(ep); key = f"videos/observation.images.{CAM}"
    mp4 = ds / "videos" / f"observation.images.{CAM}" / f"chunk-{meta[key + '/chunk_index'][n]:03d}" / f"file-{meta[key + '/file_index'][n]:03d}.mp4"
    f0 = int(round(meta[key + "/from_timestamp"][n] * 15)); length = meta["length"][n]
    cap = cv2.VideoCapture(str(mp4))
    for _ in range(f0):          # decode up to the episode: seeking an h264 file can land off by a few frames
        cap.grab()
    for _ in range(length):
        ok, img = cap.read()
        if not ok:
            break
        yield img


def label(img, text):
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
    cv2.putText(img, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return img


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset"); ap.add_argument("episode", type=int); ap.add_argument("out")
    ap.add_argument("renders", nargs="+", help="label=render dir")
    ap.add_argument("--scale", type=float, default=0.5)
    ap.add_argument("--every", type=int, default=1, help="keep every n-th frame")
    args = ap.parse_args()
    R = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in args.renders]
    tmp = Path(tempfile.mkdtemp()); n = 0
    for k, real in enumerate(real_frames(Path(args.dataset), args.episode)):
        if k % args.every:
            continue
        sims = [cv2.imread(str(d / f"sim_{CAM}_{k:04d}.jpg")) for _, d in R]
        if any(s is None for s in sims):
            continue
        top = [label(real.copy(), f"real ep{args.episode} f{k}")] + [label(s.copy(), f"sim {lab}") for (lab, _), s in zip(R, sims)]
        bot = [label(np.zeros_like(real), "bottom: real + sim 50/50")] + \
              [label(cv2.addWeighted(real, 0.5, s, 0.5, 0), f"real+sim {lab}") for (lab, _), s in zip(R, sims)]
        frame = np.vstack([np.hstack(top), np.hstack(bot)])
        frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(tmp / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90]); n += 1
    if not n:
        print("🔴 no frame had every sim image"); return 1
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(15 / args.every), "-i", str(tmp / "%05d.jpg"),
                    "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p", args.out], check=True)
    print(f"{n} frames -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
