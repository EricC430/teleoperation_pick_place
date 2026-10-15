"""Report video: real vs Isaac Sim physics replay of one 10-14 closed-loop episode, both cameras, same joint readings.
    uv run python make_report_video.py <episode dir> <sim subdir> <out.mp4>
Top: front-left real | sim. Bottom: wrist real | sim. 15 fps (real time)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

ED, SUB, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
E = json.load(open(ED / "episode.json"))
log = {r["frame"]: r for r in json.load(open(ED / SUB / "physics_log.json"))["log"]}
RID = E["dataset"]
ds = LeRobotDataset("ericc430/" + RID, root="D:/hf/lerobot/ericc430/" + RID, episodes=[E["episode"]])
W, H = 640, 360                       # each panel
FONT = cv2.FONT_HERSHEY_SIMPLEX


def panel(img, title, sub=""):
    p = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    cv2.rectangle(p, (0, 0), (W, 26), (0, 0, 0), -1)
    cv2.putText(p, title, (8, 18), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if sub:
        cv2.putText(p, sub, (8, H - 10), FONT, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    return p


tmp = Path(tempfile.mkdtemp())
n = 0
for k in range(1, len(ds)):
    sf, sw = ED / SUB / f"sim_front-left_{k:04d}.jpg", ED / SUB / f"sim_wrist_{k:04d}.jpg"
    if not sf.exists():
        continue
    item = ds[k]
    rf = (item["observation.images.front-left"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    rw = (item["observation.images.wrist"].permute(1, 2, 0).numpy() * 255).clip(0, 255).astype(np.uint8)[:, :, ::-1]
    t = k / E["fps"]
    cz = log[k]["cup_xyz"][2] * 100 if k in log else float("nan")
    top = np.hstack([panel(rf, "REAL  front-left", f"t {t:5.2f} s"), panel(cv2.imread(str(sf)), "SIM (Isaac Sim physics)  front-left", f"sim cup height {cz:4.1f} cm")])
    bot = np.hstack([panel(rw, "REAL  wrist"), panel(cv2.imread(str(sw)) if sw.exists() else np.zeros_like(rw), "SIM  wrist")])
    head = np.zeros((40, 2 * W, 3), np.uint8)
    cv2.putText(head, f"10-14 closed loop, episode {E['episode']} ({E['placement']}, real: {E['outcome']}) -- sim driven by the recorded joint readings, "
                      f"free paper cup (no attach)", (10, 26), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(str(tmp / f"f{n:05d}.png"), np.vstack([head, top, bot]))
    n += 1
OUT.parent.mkdir(parents=True, exist_ok=True)
subprocess.run([os.environ.get("FFMPEG", "D:/ffmpeg/bin/ffmpeg"), "-y", "-loglevel", "error", "-framerate", str(E["fps"]), "-i", str(tmp / "f%05d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(OUT)], check=True)
print(f"{n} frames -> {OUT}")
