#!/usr/bin/env python
"""Put several DR variants of the SAME episode side by side, next to the real recording.

This is the one picture that shows what S5 is for. Every panel replays identical action labels
from identical recorded joint angles; only the domain randomisation differs. If the panels look
the same, the DR is not doing anything; if the ARM moves differently between panels, something is
wrong, because the arm is a kinematic replay of one fixed trajectory.

    uv run python scripts/make_dr_variants_video.py \\
        --real data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60 \\
        --renders ~/isaaclab_volume/omx_sim/s5_ep0_seed42 ~/isaaclab_volume/omx_sim/s5_ep0_seed7 \\
        --out outputs/videos/ep0_dr_variants.mp4

🔴 Reads each render's fidelity flags and prints them. These clips are a pipeline and DR
demonstration, not training data -- see `docs/specs/S5_sim_replay_augmentation.md` §8.
"""
from __future__ import annotations
import argparse, json, subprocess, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--real", required=True, help="source dataset root (for the real video panel)")
ap.add_argument("--renders", nargs="+", required=True)
ap.add_argument("--camera", default="front-left")
ap.add_argument("--out", required=True)
ap.add_argument("--height", type=int, default=360)
a = ap.parse_args()

mans = [json.loads((Path(r) / "manifest.json").read_text()) for r in a.renders]
eps = {m["source_episode_id"] for m in mans}
if len(eps) != 1:
    raise SystemExit(f"renders are from different episodes {eps} -- the whole point is one episode")
n = {len(m["frames"]) for m in mans}
if len(n) != 1:
    raise SystemExit(f"renders have different frame counts {n}")

import pyarrow.parquet as pq
ds = Path(a.real)
d = pq.read_table(next((ds / "meta/episodes").rglob("*.parquet"))).to_pydict()
i = d["episode_index"].index(mans[0]["source_episode_id"])
key = f"videos/observation.images.{a.camera}"
mp4 = ds / "videos" / f"observation.images.{a.camera}" / f"chunk-{d[key+'/chunk_index'][i]:03d}" / f"file-{d[key+'/file_index'][i]:03d}.mp4"
t0 = float(d[key + "/from_timestamp"][i])

print(f"episode {mans[0]['source_episode_id']}, {len(mans[0]['frames'])} frames, {len(mans)} DR variants")
for r, m in zip(a.renders, mans):
    dr = m["dr"]
    print(f"  {Path(r).name}: seed {dr.get('seed')}, exposure {dr.get('dome_light_exposure', 0):+.2f}, "
          f"{dr.get('color_temperature_k', 0):.0f}K")
print("fidelity:", {k: v for k, v in mans[0]["fidelity"].items() if not k.endswith("_note")})

tmp = Path(tempfile.mkdtemp())
band = 26
for n_i, rec in enumerate(mans[0]["frames"]):
    panels = []
    real_p = tmp / f"r{n_i:05d}.png"
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-ss",
                    f"{t0 + rec['timestamp_s']:.4f}", "-i", str(mp4), "-frames:v", "1", str(real_p)], check=True)
    panels.append(("REAL", Image.open(real_p).convert("RGB")))
    for r, m in zip(a.renders, mans):
        img = Image.open(Path(r) / m["frames"][n_i][f"image_{a.camera}"]).convert("RGB")
        panels.append((f"DR seed {m['dr'].get('seed')}", img))
    panels = [(lab, im.resize((round(im.width * a.height / im.height), a.height))) for lab, im in panels]
    W = sum(im.width for _, im in panels)
    canvas = Image.new("RGB", (W, a.height + band), (14, 14, 16))
    dr_ = ImageDraw.Draw(canvas)
    x = 0
    for lab, im in panels:
        canvas.paste(im, (x, band))
        dr_.text((x + 6, 6), f"{lab}  f{rec['frame']}", fill=(255, 255, 255) if lab == "REAL" else (120, 220, 255))
        x += im.width
    canvas.save(tmp / f"c{n_i:05d}.png")

fps = mans[0]["dataset_fps"] / mans[0].get("stride", 1)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-framerate", f"{fps}",
                "-pattern_type", "glob", "-i", str(tmp / "c*.png"), "-c:v", "libx264",
                "-pix_fmt", "yuv420p", "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(a.out)], check=True)
print(f"\nwrote {a.out} @ {fps:g} fps (real time)")
print("If the arm differs between DR panels, that is a BUG -- they replay one fixed trajectory.")
