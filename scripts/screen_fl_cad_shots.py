"""Flag front-left shots where too little of the arm is in view for the markerless CAD fit.

Rule-based pixel count, not a quality score. The static scene is the per-pixel median over every shot of the
folder (the arm sits at a different place in each, so it drops out of the median). A shot's arm pixels are the
dark pixels that differ from that median; the biggest connected blob is the arm. A shot is flagged `arm_small`
when that blob is < --min-arm px: below ~9000 px only the gripper tip peeks in at the edge, from ~13000 px the
forearm shows (looked at on the 2026-10-29 shots). It does NOT detect a hand or a cup in the frame: a try at
that (changed pixels away from the arm) mostly picked up the arm's shadow, so it was dropped.
Nothing is deleted; the CSV lists every shot with its arm_px so the threshold can be changed.

    uv run python scripts/screen_fl_cad_shots.py calibration/shots/2026-10-29_fl_cad_before
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import cv2
import numpy as np

DARK = 90      # gray level below which a pixel can be the (black) arm
DIFF = 40      # gray difference from the median background that counts as changed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--min-arm", type=int, default=10000, help="px of the largest arm blob below which a shot is arm_small")
    ap.add_argument("--out", default="", help="CSV path (default: <folder>_screen.csv next to the folder)")
    args = ap.parse_args()

    files = sorted(glob.glob(str(Path(args.folder, "*.png"))))
    if not files:
        raise SystemExit(f"no png in {args.folder}")
    grays = [cv2.GaussianBlur(cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2GRAY), (5, 5), 0) for f in files]
    median = np.median(np.stack(grays), axis=0).astype(np.uint8)

    rows = []
    for f, g in zip(files, grays):
        arm = ((cv2.absdiff(g, median) > DIFF) & (g < DARK)).astype(np.uint8)
        arm = cv2.morphologyEx(arm, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(arm)
        arm_px = int(stats[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
        rows.append({"image": Path(f).name, "arm_px": arm_px, "flag": "arm_small" if arm_px < args.min_arm else ""})

    out = Path(args.out) if args.out else Path(args.folder).with_name(Path(args.folder).name + "_screen.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} shots, {sum(bool(r['flag']) for r in rows)} arm_small (< {args.min_arm} px) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
