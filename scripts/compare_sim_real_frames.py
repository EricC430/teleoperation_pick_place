#!/usr/bin/env python
"""Put `sim/render_state_replay.py`'s renders beside the real recorded video frames.

The sim side is posed directly at the real `observation.state` (S5 §4's kinematic replay), so if
`joint_mapping.SIGN` is right for every joint, the two arms should be in visibly the SAME
configuration at the same timestamp. A flipped sign shows up as a grossly different pose -- arm
reaching up where the real one reached down -- which is what this comparison is for.

🔴 **Not a calibration check.** The sim camera pose is PLACEHOLDER until S5 gap 4's T1/T2
measurements happen (`sim/scene_constants.py`), so the two views will NOT line up pixel-wise and
are not supposed to. Judge arm CONFIGURATION (which way each segment points), not overlap. This is
S4 §5-5's T4 smoke test, not its T3 acceptance test.

Video frame lookup uses the dataset's own episode metadata rather than assuming episode 0 starts
at t=0 of file 0: `meta/episodes/**.parquet` carries, per episode and per camera, the chunk/file
index and the `from_timestamp` that episode begins at inside that mp4.

    uv run python scripts/compare_sim_real_frames.py \\
        --dataset-root data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60 \\
        --renders ~/isaaclab_volume/omx_sim/state_replay_ep0 \\
        --out outputs/sign_check_ep0

Several render dirs put several sim variants in ONE strip, real frame first, each panel labelled --
which is what an undecided calibration question needs (S6 section 4-a: does `shoulder_lift` keep the
upper arm's 20.14 deg lean?). Label a variant with `label=dir`, and `--frames` cuts the strip down
to the frames that decide it instead of all 134:

    uv run python scripts/compare_sim_real_frames.py \\
        --dataset-root data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60 \\
        --renders "A committed=~/isaaclab_volume/omx_sim/s5_tcp_fingertip" \\
                  "B lift+20.14=~/isaaclab_volume/omx_sim/s5_tcp_fingertip_liftB" \\
        --frames 232 --out outputs/lift_AB_ep0

Reads both manifest shapes: `render_<camera>` (render_state_replay.py) and `image_<camera>`
(replay_render_episode.py).
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-root", required=True, help="the dataset dir containing meta/ and videos/")
    p.add_argument("--renders", required=True, nargs="+",
                   help="one or more render dirs (each needs manifest.json). 'label=dir' names the "
                        "panel; without a label the dir's basename is used.")
    p.add_argument("--frames", help="comma-separated frame indices to build (default: every frame "
                                    "in the first manifest)")
    p.add_argument("--camera", default="front-left", choices=["front-left", "wrist"])
    p.add_argument("--out", required=True)
    return p.parse_args()


def episode_video_location(dataset_root: str, episode: int, camera: str):
    """(mp4 path, from_timestamp) for one episode+camera, read from the dataset's own metadata."""
    import pyarrow.parquet as pq

    files = sorted(glob.glob(os.path.join(dataset_root, "meta", "episodes", "chunk-*", "file-*.parquet")))
    if not files:
        raise SystemExit(f"no episode metadata under {dataset_root}/meta/episodes/")
    key = f"videos/observation.images.{camera}"
    for f in files:
        d = pq.read_table(f).to_pydict()
        for i, ep in enumerate(d["episode_index"]):
            if ep != episode:
                continue
            mp4 = os.path.join(
                dataset_root,
                "videos",
                f"observation.images.{camera}",
                f"chunk-{d[key + '/chunk_index'][i]:03d}",
                f"file-{d[key + '/file_index'][i]:03d}.mp4",
            )
            if not os.path.exists(mp4):
                raise SystemExit(f"episode metadata points at {mp4}, which does not exist")
            return mp4, float(d[key + "/from_timestamp"][i])
    raise SystemExit(f"episode {episode} not found in {dataset_root}/meta/episodes/")


def extract_frame(mp4: str, timestamp_s: float, dest: str):
    """One frame at an absolute timestamp inside the mp4. -ss before -i seeks fast and is accurate
    enough here: at 15 fps a seek error under a frame cannot change an arm's configuration."""
    subprocess.run(
        ["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss", f"{timestamp_s:.4f}",
         "-i", mp4, "-frames:v", "1", dest],
        check=True,
    )


def strip(panels, dest, label):
    """panels: [(caption, png path)], left to right, scaled to a common height."""
    from PIL import Image, ImageDraw

    images = [Image.open(p).convert("RGB") for _, p in panels]
    h = max(im.height for im in images)
    images = [im.resize((round(im.width * h / im.height), h)) for im in images]

    band = 28
    canvas = Image.new("RGB", (sum(im.width for im in images), h + band), (16, 16, 16))
    draw = ImageDraw.Draw(canvas)
    x = 0
    for (caption, _), im in zip(panels, images):
        canvas.paste(im, (x, band))
        draw.text((x + 6, 7), caption, fill=(255, 255, 255))
        x += im.width
    draw.text((canvas.width - 8 * len(label) - 8, 7), label, fill=(180, 180, 180))
    canvas.save(dest)


def render_for(entry, camera):
    """The render's filename inside its dir, whichever manifest shape wrote it."""
    for key in (f"render_{camera}", f"image_{camera}"):
        if key in entry:
            return entry[key]
    return None


def load_renders(spec):
    """'label=dir' or 'dir' -> (label, dir, manifest)."""
    label, _, path = spec.rpartition("=")
    path = os.path.expanduser(path)
    label = label or os.path.basename(path.rstrip("/"))
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"{manifest_path} not found -- run the render first")
    return label, path, json.loads(Path(manifest_path).read_text())


def main():
    args = parse_args()
    variants = [load_renders(spec) for spec in args.renders]
    first = variants[0][2]

    episode = first["source_episode_id"]
    for label, _, m in variants[1:]:
        if m["source_episode_id"] != episode:
            raise SystemExit(f"{label} renders episode {m['source_episode_id']}, not {episode} -- "
                             "comparing different episodes side by side would be meaningless")
    mp4, from_ts = episode_video_location(args.dataset_root, episode, args.camera)
    print(f"episode {episode} {args.camera}: {mp4} starting at t={from_ts:.3f}s")
    for label, _, m in variants:
        if m.get("sign_overridden"):
            print(f"⚠️  {label}: overridden SIGN {m['sign']}")
        if m.get("offset_delta_deg"):
            print(f"⚠️  {label}: OFFSET_RAD shifted by {m['offset_delta_deg']} deg for that run")
    print(f"⚠️  {first.get('geometry', 'geometry unverified')}")

    wanted = None
    if args.frames:
        wanted = {int(x) for x in args.frames.replace(" ", "").split(",") if x}

    os.makedirs(args.out, exist_ok=True)
    made = []
    for entry in first["frames"]:
        frame = entry["frame"]
        if wanted is not None and frame not in wanted:
            continue
        real_png = os.path.join(args.out, f"f{frame:05d}_real.png")
        extract_frame(mp4, from_ts + entry["timestamp_s"], real_png)
        panels = [("REAL", real_png)]
        for label, path, m in variants:
            match = next((e for e in m["frames"] if e["frame"] == frame), None)
            name = render_for(match, args.camera) if match else None
            png = os.path.join(path, name) if name else None
            if png is None or not os.path.exists(png):
                print(f"  frame {frame}: {label} has no render, skipped")
                continue
            panels.append((f"SIM {label}", png))
        if len(panels) == 1:
            continue
        dest = os.path.join(args.out, f"f{frame:05d}_compare.png")
        strip(panels, dest, f"ep{episode} f{frame}  t={entry['timestamp_s']:.2f}s")
        made.append(dest)
        print(f"  frame {frame:>5}  -> {dest}")

    if wanted:
        missing = wanted - {int(os.path.basename(p)[1:6]) for p in made}
        if missing:
            print(f"⚠️  asked for frames not in the manifest: {sorted(missing)}")
    print(f"\n{len(made)} comparison image(s) in {args.out}")
    print("Judge arm CONFIGURATION, not pixel overlap -- the sim camera pose is a placeholder (gap 4).")


if __name__ == "__main__":
    main()
