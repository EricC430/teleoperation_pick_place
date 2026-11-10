#!/usr/bin/env python3
"""S10 §5: quantified real-vs-sim overlay. Per frame, arm (dark-pixel) mask IoU and mean boundary distance (arm-visible frames only), front-left or wrist.

Sim image k is compared with real image k (the dr_render config already applies render_lag_frames). Static dark pixels (chair, bin
stand, anything dark in >70% of frames in either video) are removed. Reported per episode and phase (thirds of the episode by frame
index: approach / grasp+carry / release), median and p10 of IoU, and median boundary distance in px at 424x240.

    python3 scripts/s10_overlay_score.py <dataset dir> <render dir pattern with {ep}, containing front-left/ or wrist/> <cam> <ep> [<ep> ...]
"""
import sys
from pathlib import Path
import cv2
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import compose_replay_ext_video as C

ds = Path(sys.argv[1]); pat = sys.argv[2]; cam = sys.argv[3]; eps = [int(x) for x in sys.argv[4:]]
C.CAM = cam; V_MAX = 60; MIN_ARM_PX = 1500; W, H = 424, 240


def dark(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[..., 2] < V_MAX


def bdist(a, b):
    """symmetric mean distance between mask contours (px)."""
    ea = cv2.morphologyEx(a.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    eb = cv2.morphologyEx(b.astype(np.uint8), cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    if not ea.any() or not eb.any():
        return np.nan
    da = cv2.distanceTransform((~ea).astype(np.uint8), cv2.DIST_L2, 3); db = cv2.distanceTransform((~eb).astype(np.uint8), cv2.DIST_L2, 3)
    return 0.5 * (da[eb].mean() + db[ea].mean())


rows = []
for ep in eps:
    real = [cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA) for f in C.real_frames(ds, ep)]
    d = Path(pat.format(ep=f"{ep:02d}")) / cam
    sim = {}
    for k in range(len(real)):
        im = cv2.imread(str(d / f"frame_{k:05d}.jpg"))
        if im is not None:
            sim[k] = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
    ks = sorted(sim); mr = np.array([dark(real[k]) for k in ks]); ms = np.array([dark(sim[k]) for k in ks])
    k5 = np.ones((7, 7), np.uint8)
    sr = cv2.dilate((mr.mean(0) > 0.3).astype(np.uint8), k5) > 0; ss = cv2.dilate((ms.mean(0) > 0.3).astype(np.uint8), k5) > 0; stat = sr | ss   # trees/chair/bin stand: dark in many frames, dilated
    iou = []; bd = []
    for i in range(len(ks)):
        a = mr[i] & ~stat; b = ms[i] & ~stat
        if max(a.sum(), b.sum()) < MIN_ARM_PX:   # arm not in view (home pose): nothing to compare
            continue
        iou.append((a & b).sum() / (a | b).sum()); bd.append(bdist(a, b))
    iou = np.array(iou); bd = np.array(bd)
    rows.append([np.median(iou), np.percentile(iou, 10), np.nanmedian(bd), np.nanpercentile(bd, 90), len(iou)])
    print(f"ep{ep:02d} {cam}: arm-visible frames {len(iou)}/{len(ks)}  IoU median {rows[-1][0]:.2f} (p10 {rows[-1][1]:.2f})  boundary distance median {rows[-1][2]:.1f}px (p90 {rows[-1][3]:.1f})")
R = np.array(rows)
print(f"mean over eps: IoU median {R[:, 0].mean():.2f}, boundary dist median {R[:, 2].mean():.1f}px (424x240 image)")
