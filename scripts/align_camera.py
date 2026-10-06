#!/usr/bin/env python
"""Live alignment of the third-person D455 against a training dataset's view.

Streams the D455 and overlays the reference view (median of the first frames of one training episode),
with live numbers for how far off the camera is. Move the camera until the numbers are inside the
thresholds (the header turns green: ALIGNED), then tighten the mount.

Numbers (all live minus reference, in pixels at 848x480):
  d_angle  wall/table boundary angle difference (deg)      -> camera roll
  d_y0     boundary height difference at x=0                -> camera tilt (with d_angle)
  dx, dy   phase correlation of the table wood grain just below the boundary -> pan (the boundary line alone
           cannot see pan: panning slides the scene along it). Shown as '?' when the correlation r < 0.2.
Everything is measured on the table (fixed): the boundary in the left part (x < 500) and the band below it.
The arm, clamp, router and curtain change between sessions and are ignored.
Reference = median of frame 0 over several episodes, which removes the object (it sits elsewhere each episode).
Clear the table while aligning.
2026-10-05: uvc_60 ep 0-49 and ep 50-59 differ by ~17 px pan / 0.5 deg roll (the camera moved between them);
the default episodes are from the 50-episode majority.

Keys: 1 blend 50/50 | 2 edges (red = reference, green = live, yellow = both) | 3 live + boundary lines |
      4 difference | s save snapshot to outputs/camera_align/ | q quit

    uv run python scripts/align_camera.py                                   # paper-cup / bottle training view
    uv run python scripts/align_camera.py --dataset-root .cache/lerobot/omx_pick_place_pilot_60_alcan_fixed --episodes 1 11 21 31 41
    uv run python scripts/align_camera.py --dry-run --live-from <dataset root> --live-episode 0   # no camera
"""

import argparse
import time
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

VIDEO_KEY = "videos/observation.images.front-left"
ROI_X = 500  # left part used for all numbers
TOL_ANGLE, TOL_PX = 0.5, 3.0
MIN_R = 0.2  # below this the wood-grain correlation is not trusted (seen on single noisy frames)


def episode_frames(root: Path, episode: int, n: int = 5) -> np.ndarray:
    """Median of the first n front-left frames of an episode (BGR)."""
    eps = pd.concat(pd.read_parquet(p) for p in sorted((root / "meta/episodes").rglob("*.parquet")))
    row = eps[eps.episode_index == episode]
    if row.empty:
        raise SystemExit(f"episode {episode} not found in {root}")
    r = row.iloc[0]
    path = root / f"{VIDEO_KEY}/chunk-{int(r[f'{VIDEO_KEY}/chunk_index']):03d}/file-{int(r[f'{VIDEO_KEY}/file_index']):03d}.mp4"
    t0, out = r[f"{VIDEO_KEY}/from_timestamp"], []
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        for fr in c.decode(s):
            if float(fr.pts * s.time_base) >= t0 - 1e-6:
                out.append(fr.to_ndarray(format="bgr24"))
                if len(out) == n:
                    break
    return np.median(np.stack(out), 0).astype(np.uint8)


def table_edge(img: np.ndarray):
    """(angle_deg, y_at_x0, y_at_x400) of the longest near-horizontal line in the left part, or None."""
    e = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 40, 120)
    e[:, ROI_X:] = 0
    e[:120] = 0
    e[400:] = 0
    lines = cv2.HoughLinesP(e, 1, np.pi / 720, 60, minLineLength=150, maxLineGap=20)
    best = None
    for x1, y1, x2, y2 in lines[:, 0] if lines is not None else []:
        a = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -30 < a < 0 and (best is None or np.hypot(x2 - x1, y2 - y1) > best[0]):
            best = (np.hypot(x2 - x1, y2 - y1), x1, y1, x2, y2, a)
    if best is None:
        return None
    _, x1, y1, x2, y2, a = best
    y_at = lambda x: y1 + (y2 - y1) * (x - x1) / (x2 - x1)
    return a, y_at(0), y_at(400)


def _grad(img: np.ndarray) -> np.ndarray:
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (5, 5), 0).astype(np.float32)
    m = cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))
    return m


def table_band(img: np.ndarray, ref_edge) -> np.ndarray:
    """Wood grain just below the reference boundary (x 40..440, 110 px high)."""
    top = int(max(ref_edge[1], ref_edge[2])) + 15 if ref_edge else 260
    return _grad(img)[top:top + 110, 40:440]


def shift(ref: np.ndarray, live: np.ndarray, ref_edge) -> tuple[float, float, float]:
    """(dx, dy, r): live table texture is displaced by (dx, dy) px relative to the reference."""
    a, b = table_band(ref, ref_edge), table_band(live, ref_edge)
    (dx, dy), resp = cv2.phaseCorrelate(a, b, cv2.createHanningWindow(a.shape[::-1], cv2.CV_32F))
    return dx, dy, resp


def measure(ref: np.ndarray, ref_edge, live: np.ndarray) -> dict:
    out = {"edge": table_edge(live)}
    out["dx"], out["dy"], out["resp"] = shift(ref, live, ref_edge)
    if ref_edge and out["edge"]:
        out["d_angle"] = out["edge"][0] - ref_edge[0]
        out["d_y0"] = out["edge"][1] - ref_edge[1]
        out["d_y400"] = out["edge"][2] - ref_edge[2]
    return out


def aligned(m: dict) -> bool:
    return (
        "d_angle" in m
        and abs(m["d_angle"]) <= TOL_ANGLE
        and abs(m["d_y0"]) <= TOL_PX
        and abs(m["d_y400"]) <= TOL_PX
        and m["resp"] >= MIN_R
        and abs(m["dx"]) <= TOL_PX
        and abs(m["dy"]) <= TOL_PX
    )


def describe(m: dict) -> str:
    edge = (
        f"d_angle {m['d_angle']:+.2f}deg  d_y0 {m['d_y0']:+.1f}px  d_y400 {m['d_y400']:+.1f}px"
        if "d_angle" in m
        else "boundary line not found"
    )
    grain = f"dx {m['dx']:+.1f} dy {m['dy']:+.1f}px" if m["resp"] >= MIN_R else "dx ? dy ?"
    return f"{edge}  |  table grain {grain} (r {m['resp']:.2f})"


def render(ref, ref_edge, live, m, mode: int) -> np.ndarray:
    if mode == 1:
        img = cv2.addWeighted(ref, 0.5, live, 0.5, 0)
    elif mode == 2:
        img = np.zeros_like(live)
        img[..., 2] = cv2.Canny(cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY), 60, 150)
        img[..., 1] = cv2.Canny(cv2.cvtColor(live, cv2.COLOR_BGR2GRAY), 60, 150)
    elif mode == 4:
        img = cv2.absdiff(ref, live)
    else:
        img = live.copy()
    for edge, colour in ((ref_edge, (0, 0, 255)), (m.get("edge"), (0, 255, 0))):
        if edge:
            cv2.line(img, (0, int(edge[1])), (400, int(edge[2])), colour, 1)
    cv2.line(img, (ROI_X, 0), (ROI_X, img.shape[0]), (128, 128, 128), 1)
    ok = aligned(m)
    cv2.rectangle(img, (0, 0), (img.shape[1], 52), (0, 120, 0) if ok else (0, 0, 0), -1)
    cv2.putText(img, ("ALIGNED  " if ok else "") + f"mode {mode}  (1 blend 2 edges 3 lines 4 diff, s save, q quit)",
                (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    cv2.putText(img, describe(m), (8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return img


def run_live(args, ref, ref_edge) -> None:
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(args.serial)
    cfg.enable_stream(rs.stream.color, ref.shape[1], ref.shape[0], rs.format.bgr8, args.fps)
    sensor = pipe.start(cfg).get_device().first_color_sensor()
    if args.exposure is not None:  # same pinned values as the recording YAML, so the overlay looks alike
        sensor.set_option(rs.option.enable_auto_exposure, 0)
        sensor.set_option(rs.option.exposure, float(args.exposure))
        sensor.set_option(rs.option.gain, float(args.gain))
    # opencv-python-headless has no imshow: show frames with tkinter, as scripts/tune_uvc_exposure.py does
    import tkinter as tk

    from PIL import Image, ImageTk

    out_dir = Path("outputs/camera_align")
    state = {"mode": 1, "hist": [], "live": None, "m": None}
    root = tk.Tk()
    root.title("align_camera  (1 blend 2 edges 3 lines 4 diff | s save | q quit)")
    img_label = tk.Label(root)
    img_label.pack()

    def close() -> None:
        pipe.stop()
        root.destroy()

    def on_key(event) -> None:
        k = event.char
        if k == "q":
            close()
        elif k in "1234" and k:
            state["mode"] = int(k)
        elif k == "s" and state["live"] is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(str(out_dir / f"live_{stamp}.png"), state["live"])
            cv2.imwrite(str(out_dir / f"overlay_{stamp}.png"), render(ref, ref_edge, state["live"], state["m"], 2))
            print(f"saved {out_dir}/*_{stamp}.png  |  {describe(state['m'])}", flush=True)

    def tick() -> None:
        frames = pipe.poll_for_frames()
        if frames:
            live = np.asanyarray(frames.get_color_frame().get_data()).copy()
            m = measure(ref, ref_edge, live)
            state["hist"] = (state["hist"] + [m])[-8:]  # average over the last 8 frames (~0.5 s) to steady them
            for key in ("d_angle", "d_y0", "d_y400", "dx", "dy", "resp"):
                vals = [h[key] for h in state["hist"] if key in h]
                if vals and key in m:
                    m[key] = float(np.mean(vals))
            state["live"], state["m"] = live, m
            shown = cv2.cvtColor(render(ref, ref_edge, live, m, state["mode"]), cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(shown))
            img_label.configure(image=photo)
            img_label.image = photo  # keep a reference, or tkinter drops the image
        root.after(10, tick)

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(10, tick)
    root.mainloop()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=Path, default=Path(".cache/lerobot/omx_pick_place_pilot_uvc_60"),
                    help="training dataset whose view to match (default: paper-cup uvc_60 = camera pose 'A')")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 10, 20, 30, 40],
                    help="reference = median of frame 0 over these episodes (default: uvc_60 majority pose)")
    ap.add_argument("--serial", default="262822305610")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--exposure", type=float, default=400, help="pin like the YAML; pass a negative value for AUTO")
    ap.add_argument("--gain", type=float, default=70)
    ap.add_argument("--dry-run", action="store_true", help="no camera: compare against --live-from instead")
    ap.add_argument("--live-from", type=Path, help="dry run: dataset whose episode stands in for the live view")
    ap.add_argument("--live-episode", type=int, default=0)
    args = ap.parse_args()
    if args.exposure is not None and args.exposure < 0:
        args.exposure = None

    ref = np.median(np.stack([episode_frames(args.dataset_root, e) for e in args.episodes]), 0).astype(np.uint8)
    ref_edge = table_edge(ref)
    print(f"reference: {args.dataset_root.name} median of ep{args.episodes}  boundary "
          + ("not found" if ref_edge is None else f"angle {ref_edge[0]:.2f}deg y0 {ref_edge[1]:.1f} y400 {ref_edge[2]:.1f}"))
    if not args.dry_run:
        run_live(args, ref, ref_edge)
        return

    # self-check of the shift sign: move the reference by a known amount
    known = cv2.warpAffine(ref, np.float32([[1, 0, 10], [0, 1, 5]]), ref.shape[1::-1], borderMode=cv2.BORDER_REFLECT)
    dx, dy, _ = shift(ref, known, ref_edge)
    print(f"self-check: reference moved by (+10, +5) px -> shift ({dx:+.1f}, {dy:+.1f})")
    if args.live_from:
        live = episode_frames(args.live_from, args.live_episode)
        m = measure(ref, ref_edge, live)
        print(f"vs {args.live_from.name} ep{args.live_episode}: {describe(m)}  ->  {'ALIGNED' if aligned(m) else 'not aligned'}")
        out = Path("outputs/camera_align")
        out.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / f"dryrun_{args.live_from.name}_ep{args.live_episode}.png"),
                    np.vstack([render(ref, ref_edge, live, m, i) for i in (1, 2)]))


if __name__ == "__main__":
    main()
