#!/usr/bin/env python
"""Rim aim from the wrist image: the labeler clicks the graspable point, the program measures the jaw gap's offset.

Why (2026-10-18, docs/meeting/2026-10-16_closed_loop.md §4-2 step 3):
  - fully automatic rim detection was right on too few episodes (dark / low-saturation table, wood grain);
  - measuring at "the rim nearest the jaw line" scores a rim parallel to the jaws as well as any other, although
    there is nothing to grasp there (Boyu, 10-18). So the target is the graspable point instead.
Graspable point (Boyu 10-18): the rim point whose tangent is perpendicular to the jaw line -- where the jaw line would
cross the rim if it went through the cup centre. Of the two such points, the one nearest the jaw-gap centre G in that
frame (Boyu 10-18: which wall the model "meant" cannot be told from one image, so no guessing). When the jaw line is
parallel to the rim, that point is far to the side of the cup and p is large -- which is the measurement.

Per episode, one frame (Boyu 10-18, "only that frame"): last_visible = the last frame before the first close in
which the rim is still visible. The window opens at TIP@RIM (export_episode_clips.tip_at_rim) and the labeler steps
forward to it; it cannot go past CLOSE. (Until 10-18 the TIP@RIM frame was labelled as well; dropped because the tip
still moves 0.6-4.9 cm after it -- docs/meeting/2026-10-16_closed_loop.md §4-2.)
Per frame, one click on the graspable point P. Offset P - G, with G = jaw-gap centre (322, 292) and the jaw line
direction U = (1, 0.28) (step 1, closed-loop 10-16 sessions only):
  t_px  along the jaw line, + = towards the right jaw
  p_px  perpendicular to it, + = downwards in the image
Both still in px (cm needs step 2). Which one is "reach" depends on wrist roll: the jaws close along U, so for a
normal grasp p is roughly radial (reach) and t sideways; for a turned grasp the other way round [inference]. Hence
wrist_roll is written next to them.

Keys:
  left click      graspable point (click again to move it)    right click / u   clear
  Enter / space   accept the click
  n               no graspable point in view in this frame
  m               cup already pushed / moved -- not measured (Boyu 10-18, c6)
  x               unsure (Boyu, c5 10-18)
A magenta line through the cursor, perpendicular to the jaw line, follows the mouse: where it lies along the rim is
the graspable point. (Boyu's third pass clicked the jaw-line / rim crossing instead -- the help text was cut off.)
  a / d           previous / next frame      A / D   5 frames
  s               skip this episode           q / Esc   quit (rows already accepted are saved)

    uv run python scripts/label_rim_aim.py <rollout dataset name>                  # all episodes
    uv run python scripts/label_rim_aim.py <rollout dataset name> --pid c2 c5
Output: eval/rim_aim/<dataset name>.csv, one row per accepted frame, written immediately; episodes / steps already
in the file are skipped on the next run (delete a row to redo it). source = click, none (n), moved (m), unsure (x).
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
N = np.array([-U[1], U[0]])  # perpendicular, pointing down in the image
SCALE = 1.5
STEPS = ("last_visible",)  # "tip_rim" dropped 10-18
COLS = ["placement_id", "episode_index", "outcome", "step", "frame", "tip_rim_frame", "close_frame", "wrist_roll",
        "source", "x", "y", "t_px", "p_px"]


def measure(P: np.ndarray) -> tuple[float, float]:
    v = P - G
    return float(v @ U), float(v @ N)


def draw(fr: np.ndarray, click: np.ndarray | None) -> np.ndarray:
    im = fr.copy()
    cv2.line(im, tuple((G - 400 * U).astype(int)), tuple((G + 400 * U).astype(int)), (255, 255, 255), 1)
    cv2.drawMarker(im, tuple(G.astype(int)), (0, 255, 0), cv2.MARKER_CROSS, 22, 2)
    if click is not None:
        t, p = measure(click)
        foot = G + t * U
        cv2.line(im, tuple(G.astype(int)), tuple(foot.astype(int)), (0, 165, 255), 2)       # along the jaw line
        cv2.line(im, tuple(foot.astype(int)), tuple(click.astype(int)), (255, 0, 255), 2)   # perpendicular
        cv2.circle(im, tuple(click.astype(int)), 5, (0, 255, 255), -1)
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
    return rim, close, wrist_frames(root, epm.loc[e], fps, lo, hi), st


def row(meta: dict, e: int, step: str, frame: int, rim: int, close, roll: float, source: str, P=None) -> dict:
    r = {c: "" for c in COLS}
    r.update(placement_id=meta.get("placement_id", "").strip(), episode_index=e, outcome=meta.get("outcome", ""),
             step=step, frame=frame, tip_rim_frame=rim, close_frame="" if close is None else close,
             wrist_roll=round(float(roll), 1), source=source)
    if P is not None:
        r["x"], r["y"] = round(float(P[0]), 1), round(float(P[1]), 1)
        r["t_px"], r["p_px"] = (round(v, 1) for v in measure(P))
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
    panel_w = 380
    scale = min(SCALE, (win.winfo_screenheight() - 200) / 480, (win.winfo_screenwidth() - panel_w - 40) / 640)
    canvas = tk.Canvas(win, width=int(640 * scale), height=int(480 * scale), cursor="crosshair")
    canvas.grid(row=0, column=0, sticky="nw")
    side = tk.Frame(win, width=panel_w)
    side.grid(row=0, column=1, sticky="nw", padx=10, pady=6)
    status = tk.Label(side, justify="left", anchor="w", wraplength=panel_w, font=("Microsoft JhengHei", 11))
    status.pack(fill="x", anchor="w")
    tk.Label(side, justify="left", anchor="w", wraplength=panel_w, fg="#333", font=("Microsoft JhengHei", 10),
             text="要點的是「可以夾的地方」，不是白線和杯緣的交點：\n"
                  "• 杯緣上，切線和白線（兩爪連線）垂直的那一點。\n"
                  "• 紫色細線會跟著滑鼠，方向和白線垂直：把它移到杯緣上，紫線剛好和杯緣重合（貼著杯緣走）的地方就是。\n"
                  "• 這種點杯子兩側各有一個：取離綠十字（兩爪空隙中心）最近的那一個。\n"
                  "• 白線和杯緣平行時，那一點會在杯子側邊、離白線很遠，照樣點；畫面裡看不到就按 n。\n\n"
                  "左鍵　點（再點一次可移動）\n"
                  "右鍵 / u　清除\n"
                  "Enter / 空白　接受\n"
                  "n　這幀看不到可以夾的地方\n"
                  "m　杯子已經被推動（不量）\n"
                  "x　不確定\n"
                  "a / d　前 / 後一幀；A / D　5 幀\n"
                  "s　跳過這集　　q / Esc　結束（已接受的都存了）").pack(fill="x", anchor="w", pady=(10, 0))
    S = {"job": -1, "steps": [], "k": 0, "click": None}

    def next_step():
        while S["steps"] == []:
            S["job"] += 1
            if S["job"] >= len(jobs):
                win.destroy()
                return
            e = jobs[S["job"]][0]
            S["steps"] = [s for s in STEPS if (e, s) not in writer.done]
            if S["steps"]:
                win.title(f"ep{e} 讀取中…")
                win.update()
                S["ep"] = load(e)
                if S["ep"] is None:
                    print(f"ep{e}: no TIP@RIM (tip never drops through the rim height before the first close) -- skipped")
                    S["steps"] = []
        S["k"] = S["ep"][0]  # both steps start at TIP@RIM
        S["click"] = None
        show()

    def show(note: str = ""):
        (e, meta), (rim, close, frames, st) = jobs[S["job"]], S["ep"]
        k = S["k"]
        im = cv2.resize(draw(frames[k], S["click"]), None, fx=scale, fy=scale)
        S["img"] = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=S["img"])
        step = S["steps"][0]
        what = ("1/2  TIP@RIM 那一幀（固定，不能移幀）" if step == "tip_rim"
                else "閉合前最後一幀還看得到杯緣：從 TIP@RIM 開始，用 d 往後移到那一幀（最多到 CLOSE）")
        ms = "t {:+.0f} px（沿白線）  p {:+.0f} px（垂直白線）".format(*measure(S["click"])) if S["click"] is not None else "（還沒點）"
        win.title(f"{meta.get('placement_id', '')} ep{e}  {meta.get('outcome', '')}  —  {step}")
        status.config(text=f"第 {S['job'] + 1}/{len(jobs)} 集  {meta.get('placement_id', '')} ep{e}  "
                           f"[{meta.get('outcome', '')} / {meta.get('mechanism', '')}]   步驟：{what}\n"
                           f"幀 {k}（TIP@RIM {rim}，CLOSE {close}）  roll {st[k, 4]:.0f}   {ms}"
                           + (f"\n→ {note}" if note else ""))

    def accept(source: str):
        (e, meta), (rim, close, _, st) = jobs[S["job"]], S["ep"]
        if source == "click" and S["click"] is None:
            show("還沒點：請點可以夾的地方，或按 n / x")
            return
        P = S["click"] if source == "click" else None
        writer.add(row(meta, e, S["steps"].pop(0), S["k"], rim, close, st[S["k"], 4], source, P))
        if S["steps"]:
            S["click"] = None
            show()
        else:
            next_step()

    def key(ev):
        frames = S["ep"][2]
        step = {"a": -1, "d": 1, "A": -5, "D": 5, "Left": -1, "Right": 1}.get(ev.keysym)
        if step and S["steps"][0] == "tip_rim":
            show("步驟 1 固定在 TIP@RIM 這一幀，不能移；移幀是步驟 2 的事")
        elif step:
            last = S["ep"][1] if S["ep"][1] is not None else max(frames)  # "before the first close": stop at CLOSE
            S["k"] = min(max(S["k"] + step, min(frames)), last)
            S["click"] = None
            show()
        elif ev.keysym in ("Return", "space"):
            accept("click")
        elif ev.keysym == "n":
            accept("none")
        elif ev.keysym == "x":
            accept("unsure")
        elif ev.keysym == "m":
            accept("moved")
        elif ev.keysym == "u":
            S["click"] = None
            show()
        elif ev.keysym == "s":
            S["steps"] = []
            next_step()
        elif ev.keysym in ("q", "Escape"):
            win.destroy()

    def click(ev):
        S["click"] = np.array([ev.x / scale, ev.y / scale])
        show()

    def guide(ev):  # line through the cursor, perpendicular to the jaw line: lay it along the rim to find the point
        canvas.delete("guide")
        c, h = np.array([ev.x, ev.y]), 90 * scale * N
        canvas.create_line(*(c - h), *(c + h), fill="#ff40ff", width=1, tags="guide")

    def clear(_):
        S["click"] = None
        show()

    win.bind("<Key>", key)
    canvas.bind("<Button-1>", click)
    canvas.bind("<Button-3>", clear)
    canvas.bind("<Motion>", guide)
    next_step()
    win.mainloop()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="rollout dataset name (folder under --data)")
    ap.add_argument("--data", default="D:/hf/lerobot/ericc430")
    ap.add_argument("--pid", nargs="*", default=[], help="placement ids, e.g. c2 c5")
    ap.add_argument("--episodes", nargs="*", type=int, default=[])
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

    jobs = [(e, meta.get(e, {})) for e in eps if not all((e, s) in writer.done for s in STEPS)]
    if jobs:
        run_gui(jobs, writer, load)
    print(f"\n{writer.path}: {len(writer.done)} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
