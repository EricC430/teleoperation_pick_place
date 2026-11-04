#!/usr/bin/env python
"""Report videos built from EXISTING DR renders (no Isaac): real vs sim comparison, a sim matrix, and per-case clips.

    python scripts/make_dr_report_videos.py compare --ep 8 --sims cup_ref_v0 cup_ref_v1 cup_plastic_v0 --out a.mp4
    python scripts/make_dr_report_videos.py matrix  --ep 8 --sims cup_heavy_v1 ... (12 names) --out b.mp4
    python scripts/make_dr_report_videos.py case    --ep 19 --sims cup_ref_v0 --title "..." --out c.mp4

A sim name is <object>_v<k>, a directory under --render (dr_collect.py render step: front-left/ + wrist/ jpgs, one per
source frame at 15 fps, manifest.json). The real clip is cut from the source LeRobot v3 video with the per-episode
start times in --offsets (episode length and from_timestamp of meta/episodes). Panel labels come from each manifest, so
what is on screen (scene mode, arm skin, object kind) is what was rendered.
"""
import argparse, ast, json, re, subprocess, sys
from pathlib import Path

VOL = Path.home() / "isaaclab_volume/dr"
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("mode", choices=["compare", "matrix", "case", "mix"])
ap.add_argument("--ep", type=int, default=0, help="the episode (all modes except mix)")
ap.add_argument("--sims", nargs="+", required=True)
ap.add_argument("--render", default=str(VOL / "run_20261101/render"))
ap.add_argument("--dataset", default="data/huggingface/lerobot/ericc430/rollout_omx_b1_papercup_60_13_10rcvrycut_46far_aug_fldrop30_100k_nas30_paper_cup_20261014_151208")
ap.add_argument("--offsets", default=str(VOL / "review_20261101_bin/offsets.json"))
ap.add_argument("--title", default="")
ap.add_argument("--notes", nargs="*", default=[], help="one extra label line per sim (case/compare)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

OFF = json.loads(Path(a.offsets).read_text()); ds = Path(a.dataset)
ep_of = lambda sim: int(re.search(r"_ep(\d+)_v", sim).group(1)) if a.mode == "mix" else a.ep   # mix: sim = <object>_ep<NN>_v<k>
o = OFF[str(a.ep)] if a.mode != "mix" else None
n = max(OFF[str(ep_of(s))]["length"] for s in a.sims) if a.mode == "mix" else o["length"]   # mix: shorter episodes hold their last frame
tmp = Path(a.out).with_suffix(".txt.d"); tmp.mkdir(exist_ok=True, parents=True)


def sdir(sim):
    base, v = sim.rsplit("_v", 1)
    obj = re.sub(r"_ep\d+$", "", base)
    return Path(a.render) / f"{obj}_cl_ep{ep_of(sim):02d}_v{v}"


def label(sim):
    m = json.loads((sdir(sim) / "manifest.json").read_text())
    dr = m["dr"]; dr = ast.literal_eval(dr) if isinstance(dr, str) else dr; p = dr["params"]
    ep = f"ep{ep_of(sim):02d} " if a.mode == "mix" else ""
    return f"{ep}{m['object_id']} v{dr['variant']} | {p['mode']} | arm {p['arm_skin']} | {p['object_kind']}"


def txt(name, s):
    p = tmp / name; p.write_text(s); return str(p)


def dt(file, x, y, size=18, color="white", extra=""):
    return f"drawtext=fontfile={FONT}:textfile={file}:x={x}:y={y}:fontsize={size}:fontcolor={color}:box=1:boxcolor=black@0.6{extra}"


inputs = [] if o is None else ["-ss", f"{o['fl'][0]:.4f}", "-i", f"{ds}/videos/observation.images.front-left/chunk-000/file-000.mp4",
                               "-ss", f"{o['wr'][0]:.4f}", "-i", f"{ds}/videos/observation.images.wrist/chunk-000/file-000.mp4"]
for s in a.sims:
    inputs += ["-framerate", "15", "-i", f"{sdir(s)}/front-left/frame_%05d.jpg", "-framerate", "15", "-i", f"{sdir(s)}/wrist/frame_%05d.jpg"]
Z = "fps=15,setpts=PTS-STARTPTS,tpad=stop_mode=clone:stop_duration=30," if a.mode == "mix" else "fps=15,setpts=PTS-STARTPTS,"
B = 0 if o is None else 2   # input index of the first sim (the two real clips come before it)
cap = txt("frame.txt", "frame %{eif:t*15:d}")

if a.mode in ("compare", "case"):
    # columns: real, sim...; row 1 third-person (front-left), row 2 wrist
    k = len(a.sims) + 1; W = 480 if k <= 4 else 400; H = 270
    cw = W; parts = []
    srcs = [("[0:v]", "[1:v]")] + [(f"[{2 + 2 * i}:v]", f"[{3 + 2 * i}:v]") for i in range(len(a.sims))]
    for i, (f, w) in enumerate(srcs):
        parts.append(f"{f}{Z}scale={cw}:{H}[f{i}];{w}{Z}scale={H * 4 // 3}:{H},pad={cw}:{H}:{(cw - H * 4 // 3) // 2}:0[w{i}]")
    row1 = "".join(f"[f{i}]" for i in range(k)) + f"hstack=inputs={k}[r1]"
    row2 = "".join(f"[w{i}]" for i in range(k)) + f"hstack=inputs={k}[r2]"
    head = 64
    draws = [dt(txt("title.txt", a.title or f"ep{a.ep:02d}"), 10, 8, 22 if k > 2 else 15, "white"), dt(cap, k * cw - 130, 8, 20, "white"),
             dt(txt("real.txt", "REAL (10/14 closed loop)"), 10, head + 4, 18, "yellow")]
    for i, s in enumerate(a.sims):
        draws.append(dt(txt(f"s{i}.txt", label(s)), (i + 1) * cw + 6, head + 4, 13 if k > 3 else 15, "yellow"))
        if i < len(a.notes):
            draws.append(dt(txt(f"n{i}.txt", a.notes[i]), (i + 1) * cw + 6, head + 28, 15, "red"))
    fc = ";".join(parts) + f";{row1};{row2};[r1][r2]vstack,pad={k * cw}:{2 * H + head}:0:{head}," + ",".join(draws) + "[v]"
else:  # matrix / mix: tile = front-left + wrist side by side; 3 columns
    cols = 3; rows = (len(a.sims) + cols - 1) // cols; H = 198 if a.mode == "mix" else 216; fw = H * 16 // 9; ww = H * 4 // 3; tw = fw + ww
    parts = []; tiles = []
    for i, s in enumerate(a.sims):
        parts.append(f"[{B + 2 * i}:v]{Z}scale={fw}:{H}[f{i}];[{B + 1 + 2 * i}:v]{Z}scale={ww}:{H}[w{i}];[f{i}][w{i}]hstack[t{i}]")
    lines = []
    for r in range(rows):
        ids = range(r * cols, min(len(a.sims), (r + 1) * cols))
        lines.append("".join(f"[t{i}]" for i in ids) + f"hstack=inputs={len(ids)}[row{r}]")
    fc = ";".join(parts + lines) + ";" + "".join(f"[row{r}]" for r in range(rows)) + f"vstack=inputs={rows},"
    fc += f"pad={cols * tw}:{rows * H + 30}:0:30,"
    draws = [dt(txt("title.txt", a.title or f"ep{a.ep:02d}"), 10, 4, 20), dt(cap, cols * tw - 130, 4, 20)]
    for i, s in enumerate(a.sims):
        x, y = (i % cols) * tw + 4, 30 + (i // cols) * H + 4
        draws.append(dt(txt(f"s{i}.txt", label(s)), x, y, 14, "yellow"))
    fc += ",".join(draws) + "[v]"

cmd = ["ffmpeg", "-y", "-loglevel", "error"] + inputs + ["-filter_complex", fc, "-map", "[v]", "-frames:v", str(n), "-r", "15",
       "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "22", a.out]
subprocess.run(cmd, check=True)
print("wrote", a.out, n, "frames")
