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
    p.add_argument("--renders", required=True, help="sim/render_state_replay.py's --out dir (needs manifest.json)")
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


def side_by_side(real_path: str, sim_path: str, dest: str, label: str):
    from PIL import Image, ImageDraw

    real = Image.open(real_path).convert("RGB")
    sim = Image.open(sim_path).convert("RGB")
    h = max(real.height, sim.height)
    real = real.resize((round(real.width * h / real.height), h))
    sim = sim.resize((round(sim.width * h / sim.height), h))

    band = 28
    canvas = Image.new("RGB", (real.width + sim.width, h + band), (16, 16, 16))
    canvas.paste(real, (0, band))
    canvas.paste(sim, (real.width, band))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 7), f"REAL  {label}", fill=(255, 255, 255))
    draw.text((real.width + 6, 7), f"SIM (kinematic replay)  {label}", fill=(255, 255, 255))
    canvas.save(dest)


def main():
    args = parse_args()
    manifest_path = os.path.join(args.renders, "manifest.json")
    if not os.path.exists(manifest_path):
        raise SystemExit(f"{manifest_path} not found -- run sim/render_state_replay.py first")
    manifest = json.loads(Path(manifest_path).read_text())

    episode = manifest["source_episode_id"]
    mp4, from_ts = episode_video_location(args.dataset_root, episode, args.camera)
    print(f"episode {episode} {args.camera}: {mp4} starting at t={from_ts:.3f}s")
    if manifest.get("sign_overridden"):
        print(f"⚠️  these renders used an overridden SIGN: {manifest['sign']}")
    print(f"⚠️  {manifest.get('geometry', 'geometry unverified')}")

    os.makedirs(args.out, exist_ok=True)
    made = []
    for entry in manifest["frames"]:
        render = os.path.join(args.renders, entry[f"render_{args.camera}"])
        if not os.path.exists(render):
            print(f"  frame {entry['frame']}: no sim render at {render}, skipped")
            continue
        real_png = os.path.join(args.out, f"f{entry['frame']:05d}_real.png")
        extract_frame(mp4, from_ts + entry["timestamp_s"], real_png)
        dest = os.path.join(args.out, f"f{entry['frame']:05d}_compare.png")
        side_by_side(real_png, render, dest, f"ep{episode} frame {entry['frame']}  t={entry['timestamp_s']:.2f}s")
        made.append(dest)
        print(f"  frame {entry['frame']:>5}  -> {dest}")

    print(f"\n{len(made)} comparison image(s) in {args.out}")
    print("Judge arm CONFIGURATION, not pixel overlap -- the sim camera pose is a placeholder (gap 4).")


if __name__ == "__main__":
    main()
