#!/usr/bin/env python3
"""S10: how many frames does the real front-left video lag the joint readings? From a state-driven sim render only (no action).

For each episode: real frame k vs sim frame k-n (n = -2..4), dark-pixel (arm) masks, static dark pixels removed (chair, bin stand),
IoU averaged over the fastest-motion quartile of frames. The best n (parabola through the three best) says: the real image k shows
the same arm pose as the sim image k-n. Sim image j shows state(j - sim_lag) because the sim PD lags its target (measured separately,
carry phase ~0.75 frame for the state-driven replay), so the video lags the readings by n + sim_lag frames.

    python3 scripts/s10_image_latency.py <dataset dir> <render links dir pattern with {ep}> <ep> [<ep> ...]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_replay_ext_video as C

ds = Path(sys.argv[1]); pat = sys.argv[2]; eps = [int(x) for x in sys.argv[3:]]
SHIFTS = list(range(-2, 5)); V_MAX = 60


def dark(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2] < V_MAX


tot = {n: [] for n in SHIFTS}
for ep in eps:
    real = [cv2.resize(f, (424, 240), interpolation=cv2.INTER_AREA) for f in C.real_frames(ds, ep)]
    d = Path(pat.format(ep=f"{ep:02d}"))
    sim = {}
    for k in range(len(real)):
        im = cv2.imread(str(d / f"sim_front-left_{k:04d}.jpg"))
        if im is not None:
            sim[k] = cv2.resize(im, (424, 240), interpolation=cv2.INTER_AREA)
    mr = np.array([dark(f) for f in real]); ms = {k: dark(f) for k, f in sim.items()}
    static_r = mr.mean(0) > 0.7; static_s = np.mean([ms[k] for k in ms], 0) > 0.7
    mover = np.array([np.mean(mr[k] ^ mr[k - 1]) for k in range(1, len(real))] + [0.0]); mover[0] = 0
    ks = [k for k in range(3, len(real) - 5) if mover[k] >= np.quantile(mover, 0.75)]
    for n in SHIFTS:
        ious = []
        for k in ks:
            if k - n not in ms:
                continue
            a = mr[k] & ~static_r; b = ms[k - n] & ~static_s
            ious.append((a & b).sum() / max((a | b).sum(), 1))
        tot[n].append(np.mean(ious))
    print(f"ep{ep:02d} IoU by shift n:", " ".join(f"{n}:{tot[n][-1]:.3f}" for n in SHIFTS), f"(fastest-quartile frames {len(ks)})")
m = np.array([np.mean(tot[n]) for n in SHIFTS]); i = int(np.argmax(m))
off = 0.5 * (m[i - 1] - m[i + 1]) / (m[i - 1] - 2 * m[i] + m[i + 1]) if 0 < i < len(m) - 1 else 0.0
print("mean IoU by n:", " ".join(f"{n}:{v:.3f}" for n, v in zip(SHIFTS, m)))
print(f"best n = {SHIFTS[i] + off:.2f} frames (real image k matches sim image k-n)")
