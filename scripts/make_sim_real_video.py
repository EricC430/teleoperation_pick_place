#!/usr/bin/env python
"""Side-by-side REAL vs SIM video from a `sim/replay_render_episode.py` output directory.

The stills version (`scripts/compare_sim_real_frames.py`) answers "is a joint sign flipped".
This answers the questions that only motion shows: does the replayed arm move the way the real one
did, does the grasp happen when the real grasp happened, does the DR look sane across a whole
episode. It was what made the 2026-09-21 vertical offset obvious at a glance -- the real gripper is
down on the cup while the replayed one hangs in the air above it.

    uv run python scripts/make_sim_real_video.py <render-dir> <out.mp4>

Output frame rate is `dataset_fps / stride`, i.e. REAL TIME: a stride-4 render plays at 3.75 fps,
which is choppy but honest. Render with `--stride 1` for a smooth 15 fps version.

🔴 Reads the fidelity flags out of the render's manifest and prints them. The sim panel is NOT a
prediction of what the real arm would do -- geometry is unaligned (S5 gap 4), the object is not
the source episode's object, and the joint zero-offsets are unmeasured (S5 §2-C). Judge motion and
timing from this, not position accuracy.
"""
import json, subprocess, sys, tempfile
from pathlib import Path
from PIL import Image, ImageDraw

render = Path(sys.argv[1]); out = Path(sys.argv[2]); cam = "front-left"
ds = Path("data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60")
m = json.loads((render/"manifest.json").read_text())
ep = m["source_episode_id"]; fps_src = m["dataset_fps"]; stride = m.get("stride",1)

import pyarrow.parquet as pq
d = pq.read_table(next((ds/"meta/episodes").rglob("*.parquet"))).to_pydict()
i = d["episode_index"].index(ep)
key = f"videos/observation.images.{cam}"
mp4 = ds/"videos"/f"observation.images.{cam}"/f"chunk-{d[key+'/chunk_index'][i]:03d}"/f"file-{d[key+'/file_index'][i]:03d}.mp4"
t0 = float(d[key+"/from_timestamp"][i])

tmp = Path(tempfile.mkdtemp())
fid = m["fidelity"]
for n, rec in enumerate(m["frames"]):
    sim_p = render/rec[f"image_{cam}"]
    if not sim_p.exists(): continue
    real_p = tmp/f"r{n:05d}.png"
    subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-ss",f"{t0+rec['timestamp_s']:.4f}",
                    "-i",str(mp4),"-frames:v","1",str(real_p)],check=True)
    R = Image.open(real_p).convert("RGB"); S = Image.open(sim_p).convert("RGB")
    h = 480
    R = R.resize((round(R.width*h/R.height),h)); S = S.resize((round(S.width*h/S.height),h))
    band = 34
    c = Image.new("RGB",(R.width+S.width,h+band),(14,14,16)); c.paste(R,(0,band)); c.paste(S,(R.width,band))
    dr = ImageDraw.Draw(c)
    dr.text((8,5), f"REAL  ep{ep} f{rec['frame']}  t={rec['timestamp_s']:.1f}s", fill=(255,255,255))
    tag = "ATTACHED" if rec.get("grasp_attached") else ""
    dist = rec.get("tcp_object_dist_m")
    extra = f"  TCP-obj {dist*100:.0f}cm" if dist else ""
    dr.text((R.width+8,5), f"SIM (kinematic replay, DR seed {m['dr'].get('seed')}){extra} {tag}", fill=(120,220,255))
    c.save(tmp/f"c{n:05d}.png")

fps_out = fps_src/stride
subprocess.run(["ffmpeg","-nostdin","-loglevel","error","-y","-framerate",f"{fps_out}",
                "-pattern_type","glob","-i",str(tmp/"c*.png"),
                "-c:v","libx264","-pix_fmt","yuv420p","-vf","scale=trunc(iw/2)*2:trunc(ih/2)*2",
                str(out)],check=True)
print(f"wrote {out}  ({len(list(tmp.glob('c*.png')))} frames @ {fps_out:g} fps = real time)")
print("fidelity:", {k:v for k,v in fid.items() if not k.endswith('_note')})
