#!/usr/bin/env python
"""Rim aim from the wrist image -- the program proposes, a person accepts or corrects.

Why (2026-10-18, docs/meeting/2026-10-16_closed_loop.md §4-2 step 3): fully automatic rim detection was right on
4/10 episodes (hue crossing + local texture direction); the rest failed on the dark / low-saturation table or on
wood grain. So the program draws its best guess and the labeler fixes it with two clicks.

Per episode, two frames (Boyu 10-17, "both"):
  tip_rim       the TIP@RIM frame (export_episode_clips.tip_at_rim), fixed
  last_visible  the last frame before the first close in which the rim is still visible; the labeler steps to it
Per frame, the rim near the jaw line is a straight line through two points on the rim. From it:
  t_px       where that line crosses the jaw line, in px along the line from the jaw-gap centre G (+ = towards the
             right jaw). This is d_center, still in px: the px -> cm conversion needs step 2.
  angle_deg  angle between that line and the jaw line, 0-90; 90 = rim perpendicular to the jaws (Boyu's criterion).
             In-image angle, not yet corrected for the oblique view (step 2).
Jaw line: through G = (322, 292), direction (1, 0.28) -- step 1, closed-loop 10-16 sessions only.

Keys (window title shows the current step):
  left click x2   rim line (a third click starts over)       right click / u   clear the clicks
  Enter / space   accept: the clicked line, else the program's (yellow) line
  n               no rim near the jaw line in this frame (nothing to measure)
  x               unsure whether the rim is visible / where it is (Boyu, c5 10-18)
  a / d           previous / next frame      A / D   5 frames   (last_visible only; tip_rim is fixed -- Boyu moved
                  it on c2 / c12 / c33 in the first pass, thinking step 1 was also "last visible")
  s               skip this episode           q / Esc   quit (rows already accepted are saved)

    uv run python scripts/label_rim_aim.py <rollout dataset name>                  # all episodes
    uv run python scripts/label_rim_aim.py <rollout dataset name> --pid c2 c5
    uv run python scripts/label_rim_aim.py <rollout dataset name> --auto-only      # no window, tip_rim only
Output: eval/rim_aim/<dataset name>.csv, one row per accepted frame, written immediately; episodes / steps already
in the file are skipped on the next run (delete a row to redo it). source = auto (accepted as drawn), manual
(clicked), none (n), unsure (x), auto_unreviewed (--auto-only); auto / (auto + manual) is the hit rate of the detector.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import av
import cv2
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "sim")); sys.path.insert(0, str(REPO))
import touch_test as TT  # noqa: E402
import fit_paper_to_pan as F  # noqa: E402
import export_episode_clips as X  # noqa: E402

G = np.array([322.0, 292.0])
U = np.array([1.0, 0.28]) / np.hypot(1.0, 0.28)
T_MIN, T_MAX = -322, 240  # beyond +240 px the right jaw's tissue pad looks like cup
SCALE = 1.5
MIN_CLICK_PX = 30  # two clicks 8 px apart with ~1.5 px click error -> angle off by ~20 deg (Boyu's first pass, 10-18)
COLS = ["placement_id", "episode_index", "outcome", "step", "frame", "tip_rim_frame", "close_frame", "source",
        "x1", "y1", "x2", "y2", "t_px", "angle_deg", "auto_x1", "auto_y1", "auto_x2", "auto_y2"]


# --- the program's guess -------------------------------------------------------------------------------------

def hue_crossing(fr: np.ndarray) -> float | None:
    """Cup (bluish) / table (orange) boundary along the jaw line nearest G, as t in px; None if no boundary."""
    hsv = cv2.cvtColor(cv2.GaussianBlur(fr, (7, 7), 0), cv2.COLOR_BGR2HSV)
    ts = np.arange(T_MIN, T_MAX + 1, 2.0)
    P = (G[None] + ts[:, None] * U[None]).round().astype(int)
    ok = (P[:, 0] >= 0) & (P[:, 0] < fr.shape[1]) & (P[:, 1] >= 0) & (P[:, 1] < fr.shape[0])
    P, ts = P[ok], ts[ok]
    h, s = hsv[P[:, 1], P[:, 0], 0].astype(int), hsv[P[:, 1], P[:, 0], 1].astype(int)
    cls = np.zeros(len(P), int)
    cls[(h >= 80) & (h <= 130) & (s >= 5)] = 1
    cls[(h >= 5) & (h <= 45) & (s >= 8)] = 2
    sm = cls.copy()
    for k in range(len(cls)):  # majority of the known labels within +-4 samples
        w = cls[max(0, k - 4):k + 5]
        w = w[w > 0]
        sm[k] = np.bincount(w).argmax() if w.size else 0
    runs: list[list] = []
    for k in range(len(sm)):
        if sm[k] == 0:
            continue
        if runs and runs[-1][0] == sm[k] and ts[k] - runs[-1][2] <= 6:
            runs[-1][2] = ts[k]
        else:
            runs.append([sm[k], ts[k], ts[k]])
    cross = [(a[2] + b[1]) / 2 for a, b in zip(runs, runs[1:]) if a[0] != b[0] and b[1] - a[2] <= 80]
    return min(cross, key=abs) if cross else None


def texture(fr: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    hp = g - cv2.GaussianBlur(g, (0, 0), 3)
    return np.sqrt(cv2.GaussianBlur(hp * hp, (0, 0), 4))


def local_direction(tex: np.ndarray, q: np.ndarray, rad: int = 35) -> np.ndarray:
    """Main direction of the brightest 15 % fine-detail pixels within rad px of q (the rim is a bright ridge)."""
    y0, y1 = int(max(0, q[1] - rad)), int(min(tex.shape[0], q[1] + rad))
    x0, x1 = int(max(0, q[0] - rad)), int(min(tex.shape[1], q[0] + rad))
    sub = tex[y0:y1, x0:x1]
    ys, xs = np.nonzero(sub >= np.percentile(sub, 85))
    w = sub[ys, xs]
    pts = np.c_[xs + x0, ys + y0].astype(float)
    mu = (pts * w[:, None]).sum(0) / w.sum()
    C = ((pts - mu).T * w) @ (pts - mu) / w.sum()
    return np.linalg.eigh(C)[1][:, 1]


def propose(fr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    t = hue_crossing(fr)
    if t is None:
        return None
    q = G + t * U
    d = local_direction(texture(fr), q)
    return q - 40 * d, q + 40 * d


# --- measurement ---------------------------------------------------------------------------------------------

def measure(p1: np.ndarray, p2: np.ndarray) -> tuple[float, float] | None:
    """(t_px, angle_deg) of the line p1-p2 against the jaw line; t is nan if they are parallel."""
    d = p2 - p1
    if np.hypot(*d) < 5:
        return None
    d = d / np.hypot(*d)
    angle = float(np.degrees(np.arccos(min(1.0, abs(d @ U)))))
    A = np.c_[U, -d]  # G + t U = p1 + s d
    t = float(np.linalg.solve(A, p1 - G)[0]) if abs(np.linalg.det(A)) > 1e-6 else float("nan")
    return t, angle


def draw(fr: np.ndarray, auto, clicks: list[np.ndarray]) -> np.ndarray:
    im = fr.copy()
    cv2.line(im, tuple((G + T_MIN * U).astype(int)), tuple((G + T_MAX * U).astype(int)), (255, 255, 255), 1)
    cv2.drawMarker(im, tuple(G.astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    lines = []
    if auto is not None:
        lines.append((auto, (0, 255, 255)))
    if len(clicks) == 2:
        lines.append(((clicks[0], clicks[1]), (255, 255, 0)))
    for (p1, p2), col in lines:
        d = (p2 - p1) / max(np.hypot(*(p2 - p1)), 1e-6)
        cv2.line(im, tuple((p1 - 400 * d).astype(int)), tuple((p2 + 400 * d).astype(int)), col, 1)  # extended
        cv2.line(im, tuple(p1.astype(int)), tuple(p2.astype(int)), col, 3)
        m = measure(p1, p2)
        if m and np.isfinite(m[0]):
            cv2.circle(im, tuple((G + m[0] * U).astype(int)), 5, (0, 0, 255), -1)
    for c in clicks:
        cv2.circle(im, tuple(c.astype(int)), 4, (255, 255, 0), -1)
    return im


# --- episodes and output -------------------------------------------------------------------------------------

def wrist_frames(root: Path, row: pd.Series, fps: int, lo: int, hi: int) -> dict[int, np.ndarray]:
    """Episode frames lo..hi (episode-relative) of the wrist video; seeks instead of decoding from the file start."""
    pre = "videos/observation.images.wrist/"
    path = root / f"{pre}chunk-{int(row[pre + 'chunk_index']):03d}/file-{int(row[pre + 'file_index']):03d}.mp4"
    k0 = int(round(row[pre + "from_timestamp"] * fps))
    out = {}
    with av.open(str(path)) as c:
        s = c.streams.video[0]
        c.seek(int((k0 + lo) / fps / s.time_base), stream=s, backward=True, any_frame=False)
        for fr in c.decode(s):
            k = int(round(fr.time * fps)) - k0
            if k > hi:
                break
            if k >= lo:
                out[k] = fr.to_ndarray(format="bgr24")
    return out


def load_episode(root: Path, epm: pd.DataFrame, e: int, st: np.ndarray, fit: TT.FitModel, fps: int):
    close = F.grasp_frame(st, "first")
    rim = X.tip_at_rim(fit, st, close)
    if rim is None:
        return None
    lo, hi = max(0, rim - 15), min(len(st) - 1, (close if close is not None else rim + 60) + 5)
    return rim, close, wrist_frames(root, epm.loc[e], fps, lo, hi)


def row(meta: dict, e: int, step: str, frame: int, rim: int, close, source: str, line=None, auto=None) -> dict:
    r = {c: "" for c in COLS}
    r.update(placement_id=meta.get("placement_id", "").strip(), episode_index=e, outcome=meta.get("outcome", ""),
             step=step, frame=frame, tip_rim_frame=rim, close_frame="" if close is None else close, source=source)
    if line is not None:
        (r["x1"], r["y1"]), (r["x2"], r["y2"]) = np.round(line[0], 1), np.round(line[1], 1)
        m = measure(*line)
        if m:
            r["t_px"], r["angle_deg"] = round(m[0], 1), round(m[1], 1)
    if auto is not None:
        (r["auto_x1"], r["auto_y1"]), (r["auto_x2"], r["auto_y2"]) = np.round(auto[0], 1), np.round(auto[1], 1)
    return r


class Writer:
    def __init__(self, path: Path):
        self.path = path
        self.done = set()
        if path.exists():
            self.done = {(int(r["episode_index"]), r["step"]) for r in csv.DictReader(path.open(encoding="utf-8-sig"))}
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", newline="", encoding="utf-8-sig") as f:
                csv.DictWriter(f, COLS).writeheader()

    def add(self, r: dict) -> None:
        with self.path.open("a", newline="", encoding="utf-8-sig") as f:
            csv.DictWriter(f, COLS).writerow(r)
        self.done.add((int(r["episode_index"]), r["step"]))


def run_gui(jobs, writer: Writer, load) -> None:
    import tkinter as tk
    from PIL import Image, ImageTk

    win = tk.Tk()
    canvas = tk.Canvas(win, width=int(640 * SCALE), height=int(480 * SCALE))
    canvas.pack()
    status = tk.Label(win, justify="left", anchor="w", font=("Microsoft JhengHei", 11))
    status.pack(fill="x")
    tk.Label(win, justify="left", anchor="w", fg="#555", font=("Microsoft JhengHei", 10),
             text="左鍵點兩點＝杯緣（沿杯緣相隔 ≥ 30 px，第三下重來）  右鍵/u＝清除  Enter/空白＝接受（有點選用點選，否則用黃線）\n"
                  "n＝這幀兩爪連線附近沒有杯緣  x＝看不出來／不確定  a/d＝前/後一幀（只限步驟 2）  A/D＝5 幀  s＝跳過這集  q/Esc＝結束").pack(fill="x")
    S = {"job": -1, "steps": [], "k": 0, "clicks": [], "auto": None}

    def next_step():
        while S["steps"] == []:
            S["job"] += 1
            if S["job"] >= len(jobs):
                win.destroy()
                return
            e = jobs[S["job"]][0]
            S["steps"] = [s for s in ("tip_rim", "last_visible") if (e, s) not in writer.done]
            if S["steps"]:
                win.title(f"ep{e} 讀取中…")
                win.update()
                S["ep"] = load(e)
                if S["ep"] is None:
                    print(f"ep{e}: no TIP@RIM (tip never drops through the rim height before the first close) -- skipped")
                    S["steps"] = []
        S["k"] = S["ep"][0]  # both steps start at TIP@RIM
        S["clicks"] = []
        show()

    def show():
        (e, meta), (rim, close, frames) = jobs[S["job"]], S["ep"]
        k = S["k"]
        S["auto"] = propose(frames[k])
        im = cv2.resize(draw(frames[k], S["auto"], S["clicks"]), None, fx=SCALE, fy=SCALE)
        S["img"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=S["img"])
        step = S["steps"][0]
        what = ("1/2  TIP@RIM 那一幀（固定，不能移幀）" if step == "tip_rim"
                else "2/2  閉合前最後一幀還看得到杯緣（用 a/d 移到那一幀）")
        line = (S["clicks"][0], S["clicks"][1]) if len(S["clicks"]) == 2 else S["auto"]
        m = measure(*line) if line is not None else None
        ms = f"t {m[0]:+.0f} px  夾角 {m[1]:.0f}°" if m else "（沒有線）"
        win.title(f"{meta.get('placement_id', '')} ep{e}  {meta.get('outcome', '')}  —  {step}")
        status.config(text=f"第 {S['job'] + 1}/{len(jobs)} 集  {meta.get('placement_id', '')} ep{e}  "
                           f"[{meta.get('outcome', '')} / {meta.get('mechanism', '')}]   步驟：{what}\n"
                           f"幀 {k}（TIP@RIM {rim}，CLOSE {close}）   {'點選' if len(S['clicks']) == 2 else '黃線'}：{ms}")

    def accept(source: str):
        (e, meta), (rim, close, _) = jobs[S["job"]], S["ep"]
        line = None
        if source not in ("none", "unsure"):
            if len(S["clicks"]) == 2:
                gap = float(np.hypot(*(S["clicks"][1] - S["clicks"][0])))
                if gap < MIN_CLICK_PX:
                    status.config(text=status.cget("text") + f"\n→ 兩點只差 {gap:.0f} px，角度會不準：請沿杯緣點遠一點（≥ {MIN_CLICK_PX} px，連線兩側各一點）")
                    return
                line, source = (S["clicks"][0], S["clicks"][1]), "manual"
            elif S["auto"] is not None:
                line, source = S["auto"], "auto"
            else:
                status.config(text=status.cget("text") + "\n→ 沒有黃線：請點兩下，或按 n")
                return
        writer.add(row(meta, e, S["steps"].pop(0), S["k"], rim, close, source, line, S["auto"]))
        if S["steps"]:
            S["clicks"] = []
            show()
        else:
            next_step()

    def key(ev):
        frames = S["ep"][2]
        step = {"a": -1, "d": 1, "A": -5, "D": 5, "Left": -1, "Right": 1}.get(ev.keysym)
        if step and S["steps"][0] == "tip_rim":
            status.config(text=status.cget("text") + "\n→ 步驟 1 固定在 TIP@RIM 這一幀，不能移；移幀是步驟 2 的事")
        elif step:
            S["k"] = min(max(S["k"] + step, min(frames)), max(frames))
            S["clicks"] = []
            show()
        elif ev.keysym in ("Return", "space"):
            accept("auto")
        elif ev.keysym == "n":
            accept("none")
        elif ev.keysym == "x":
            accept("unsure")
        elif ev.keysym == "u":
            S["clicks"] = []
            show()
        elif ev.keysym == "s":
            S["steps"] = []
            next_step()
        elif ev.keysym in ("q", "Escape"):
            win.destroy()

    def click(ev):
        if len(S["clicks"]) == 2:
            S["clicks"] = []
        S["clicks"].append(np.array([ev.x / SCALE, ev.y / SCALE]))
        show()

    def clear(_):
        S["clicks"] = []
        show()

    win.bind("<Key>", key)
    canvas.bind("<Button-1>", click)
    canvas.bind("<Button-3>", clear)
    next_step()
    win.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="rollout dataset name (folder under --data)")
    ap.add_argument("--data", default="D:/hf/lerobot/ericc430")
    ap.add_argument("--pid", nargs="*", default=[], help="placement ids, e.g. c2 c5")
    ap.add_argument("--episodes", nargs="*", type=int, default=[])
    ap.add_argument("--auto-only", action="store_true", help="no window: write the program's tip_rim guess only")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.data) / a.dataset
    fps = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))["fps"]
    meta = {int(r["episode_index"]): r for r in csv.DictReader((REPO / "episode_meta" / f"{a.dataset}.csv").open(encoding="utf-8-sig"))}
    eps = sorted(set(a.episodes) | {e for e, r in meta.items() if r["placement_id"].strip() in a.pid}) if (a.pid or a.episodes) else sorted(meta)
    writer = Writer(Path(a.out) if a.out else REPO / "eval/rim_aim" / f"{a.dataset}.csv")
    epm = pd.concat(pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))).set_index("episode_index")
    F.DATA = Path(a.data)
    states = F.load_states(a.dataset)
    fit = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")

    def load(e):
        return load_episode(root, epm, e, states[e], fit, fps)

    if a.auto_only:
        for e in eps:
            if (e, "tip_rim") in writer.done:
                continue
            ep = load(e)
            if ep is None:
                print(f"ep{e}: no TIP@RIM (tip never drops through the rim height before the first close) -- skipped")
                continue
            rim, close, frames = ep
            auto = propose(frames[rim])
            writer.add(row(meta.get(e, {}), e, "tip_rim", rim, rim, close, "auto_unreviewed", auto, auto))
            m = measure(*auto) if auto else None
            print(f"{meta.get(e, {}).get('placement_id', '')} ep{e} f{rim}: " + (f"t {m[0]:+.0f} px  angle {m[1]:.0f}" if m else "no guess"))
    else:
        jobs = [(e, meta.get(e, {})) for e in eps if not all((e, s) in writer.done for s in ("tip_rim", "last_visible"))]
        if jobs:
            run_gui(jobs, writer, load)
    print(f"\n{writer.path}: {len(writer.done)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
