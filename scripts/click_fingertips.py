#!/usr/bin/env python
"""Click the closed fingertip in each S7 front-left still (keypoints for the gripper-focused arm fit, S7 §7).

Why: on 2026-10-12 the S8 plumb-line fingertip, projected with the 10-10 extrinsics, landed on the real fingertip
in the P02/P03 stills -- so the camera and S8 agree, and the error is in the arm model. A silhouette IoU is
dominated by the big links; a grasp replay needs the gripper. These clicks are the image-side fingertip data.

    uv run python scripts/click_fingertips.py --captures calibration/2026-10-10_link_tilt_captures.csv

THE POINT [2026-10-14, one definition for every still]: the MIDPOINT of the FRONT EDGE of the finger nearer the camera
(the lattice finger on the arm's left, +y, whose outer face the front-left camera sees). The fit models exactly this
point from that finger's mesh, so do not click the finger's side, the midline behind it, or anything in front of it.
If that edge is hidden (by the other finger, an object, the image edge) press s -- never click the occluder.
Example: outputs/s7s8_overlay/click_example.png.

Left panel = the whole still, right panel = 4x around the current point.
Mouse: click in the LEFT panel, then click in the RIGHT panel to refine it (as often as you like).
Keys:  Enter/Space = save and next   s = fingertip not visible (hidden, out of frame, cut by the edge), skip
       u = clear the point            q = quit (everything saved so far is kept; rerun to continue)
(matplotlib/Tk, because the project's OpenCV is headless.)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

FIELDS = ("pose_id", "approach", "image", "visible", "u", "v")
HALF = 40   # right panel shows +-40 px around the point


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--captures", default="calibration/2026-10-10_link_tilt_captures.csv")
    ap.add_argument("--camera", default="front-left")
    ap.add_argument("--out", default=None, help="default: <captures stem without _captures>_fingertip_clicks.csv")
    args = ap.parse_args()
    cap = Path(args.captures)
    out = Path(args.out) if args.out else cap.with_name(cap.stem.replace("_captures", "") + "_fingertip_clicks.csv")
    seen, todo = set(), []
    for r in csv.DictReader(cap.open(encoding="utf-8")):
        key = (r["pose_id"], r["approach"])
        if r["camera"] == args.camera and key not in seen:
            seen.add(key)
            todo.append((r["pose_id"], r["approach"], r["image"].replace("\\", "/")))
    done = set()
    if out.exists():
        done = {(r["pose_id"], r["approach"]) for r in csv.DictReader(out.open(encoding="utf-8"))}
    todo = [t for t in todo if (t[0], t[1]) not in done]
    print(f"{len(todo)} stills to do ({len(done)} already in {out})")

    fig, (ax, az) = plt.subplots(1, 2, figsize=(15, 6.2), gridspec_kw={"width_ratios": [1.77, 1]})
    st = {"i": 0, "pt": None, "img": None, "quit": False}

    def draw():
        pid, appr, _ = todo[st["i"]]
        img = st["img"]
        ax.clear(); az.clear()
        ax.imshow(img)
        ax.set_title(f"{st['i'] + 1}/{len(todo)}  {pid} {appr or 'start'}  -  click the MIDPOINT of the near finger's FRONT EDGE; Enter=save, s=hidden, u=clear, q=quit",
                     fontsize=10)
        ax.axis("off")
        if st["pt"]:
            u, v = st["pt"]
            ax.plot(u, v, "r+", ms=18, mew=2)
            x0, y0 = int(round(u)) - HALF, int(round(v)) - HALF
            pad = np.pad(img, ((HALF, HALF), (HALF, HALF), (0, 0)))
            az.imshow(pad[y0 + HALF:y0 + 3 * HALF, x0 + HALF:x0 + 3 * HALF], extent=(x0, x0 + 2 * HALF, y0 + 2 * HALF, y0),
                      interpolation="nearest")
            az.plot(u, v, "r+", ms=26, mew=2)
            az.set_title("4x: click here to refine", fontsize=10)
        else:
            az.set_title("(click the fingertip on the left)", fontsize=10)
        az.axis("off")
        fig.canvas.draw_idle()

    def load():
        st["img"] = np.asarray(Image.open(todo[st["i"]][2]).convert("RGB"))
        st["pt"] = None
        draw()

    def save(visible):
        pid, appr, path = todo[st["i"]]
        row = {"pose_id": pid, "approach": appr, "image": path, "visible": int(visible),
               "u": round(st["pt"][0], 1) if visible else "", "v": round(st["pt"][1], 1) if visible else ""}
        new = not out.exists()
        with out.open("a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            if new:
                w.writeheader()
            w.writerow(row)
        print(f"  {pid} {appr or 'start'}: " + (f"({row['u']}, {row['v']})" if visible else "not visible"))
        st["i"] += 1
        if st["i"] >= len(todo):
            plt.close(fig)
        else:
            load()

    def on_click(ev):
        if ev.inaxes in (ax, az) and ev.xdata is not None and getattr(fig.canvas.toolbar, "mode", "") == "":
            if ev.inaxes is az and st["pt"] is None:
                return
            st["pt"] = (float(ev.xdata), float(ev.ydata))
            draw()

    def on_key(ev):
        if ev.key in ("enter", " ") and st["pt"]:
            save(True)
        elif ev.key == "s":
            save(False)
        elif ev.key == "u":
            st["pt"] = None
            draw()
        elif ev.key == "q":
            plt.close(fig)

    for k in ("keymap.save", "keymap.quit", "keymap.fullscreen", "keymap.home", "keymap.back", "keymap.forward",
              "keymap.pan", "keymap.zoom", "keymap.xscale", "keymap.yscale", "keymap.grid", "keymap.grid_minor"):
        plt.rcParams[k] = []
    if not todo:
        print("nothing to do")
        return 0
    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    load()
    plt.tight_layout()
    plt.show()
    print(f"saved in {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
