#!/usr/bin/env python
"""One mp4 per episode: wrist (left) | front-left (right), with the episode and its timeline written on the frame.

Why (2026-10-17): the dataset videos hold every episode back to back, so it is hard to tell which episode is playing.
The labels are drawn only on these exported copies -- never on the recorded frames, which are what the policy sees
and what training / h7 replays read.

Drawn on each frame:
  top     : run, placement, episode, outcome, mechanism (episode_meta CSV)
  bottom  : frame / seconds, gripper and wrist_roll readings, and two events from the joint data:
            TIP@RIM = jaw-midline fingertip drops through the cup rim height (9.3 cm) on its last descent before
                      the first close (touch-pinned arm model, calibration/2026-10-14_link_marker_fit.json; height
                      accuracy at 9 cm not verified -- prototype, docs/worklog/2026-10-16_closed_loop.md)
            CLOSE   = first sustained gripper close (fit_paper_to_pan.grasp_frame, same as rim_distance.py)
            The banner turns orange from TIP@RIM to CLOSE.

    uv run python scripts/export_episode_clips.py <rollout dataset name> --pid c3 c12 c22 c33
    uv run python scripts/export_episode_clips.py <rollout dataset name> --episodes 2 11 --slow 2
Output: outputs/clips/<dataset name>/<pid>_ep<N>.mp4 (h264, plays in the Windows player / VLC).
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
import omx_constants as K  # noqa: E402
import touch_test as TT  # noqa: E402
import fit_paper_to_pan as F  # noqa: E402

RIM_Z_CM = 9.3
TIP = np.array([K.GRIPPER_TIP_X_M, K.GRIPPER_MIDLINE_Y_M, 0.0, 1.0])
H = 360  # output height of each camera tile
CAMS = ("wrist", "front-left")  # left, right


def tip_at_rim(fit: TT.FitModel, st: np.ndarray, close: int | None, rim_cm: float = RIM_Z_CM) -> int | None:
    if rim_cm <= 0:
        return None
    end = close if close is not None else len(st) - 1
    z = np.array([(fit.link5(s) @ TIP)[2] * 100 for s in st[:end + 1]])
    down = [k for k in range(1, len(z)) if z[k - 1] > rim_cm >= z[k]]
    return down[-1] if down else None


def episode_frames(root: Path, cam: str, row: pd.Series, fps: int):
    pre = f"videos/observation.images.{cam}/"
    path = root / f"{pre}chunk-{int(row[pre + 'chunk_index']):03d}/file-{int(row[pre + 'file_index']):03d}.mp4"
    k0, n = int(round(row[pre + "from_timestamp"] * fps)), int(row["length"])
    with av.open(str(path)) as c:
        for i, fr in enumerate(c.decode(video=0)):
            if i >= k0 + n:
                break
            if i >= k0:
                yield fr.to_ndarray(format="bgr24")


def text(img, s, org, color=(255, 255, 255), scale=0.6):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="rollout dataset name (folder under --data)")
    ap.add_argument("--data", default="D:/hf/lerobot/ericc430")
    ap.add_argument("--pid", nargs="*", default=[], help="placement ids, e.g. c3 c12")
    ap.add_argument("--episodes", nargs="*", type=int, default=[])
    ap.add_argument("--slow", type=float, default=1.0, help="2 = half speed")
    ap.add_argument("--rim-cm", type=float, default=RIM_Z_CM,
                    help="TIP@RIM height (paper-cup rim by default); 0 = no TIP@RIM, e.g. for the bottle")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    root = Path(a.data) / a.dataset
    fps = json.loads((root / "meta/info.json").read_text(encoding="utf-8"))["fps"]
    meta = {int(r["episode_index"]): r for r in csv.DictReader((REPO / "episode_meta" / f"{a.dataset}.csv").open(encoding="utf-8-sig"))}
    eps = sorted(set(a.episodes) | {e for e, r in meta.items() if r["placement_id"].strip() in a.pid})
    if not eps:
        raise SystemExit("no episode selected (check --pid / --episodes against the episode_meta CSV)")
    epmeta = pd.concat(pd.read_parquet(f) for f in sorted((root / "meta/episodes").rglob("*.parquet"))).set_index("episode_index")
    F.DATA = Path(a.data)
    states = F.load_states(a.dataset)
    fit = TT.FitModel(REPO / "calibration/2026-10-14_link_marker_fit.json")
    out_dir = Path(a.out) if a.out else REPO / "outputs/clips" / a.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    short = a.dataset.replace("rollout_omx_b1_papercup_", "")[:70]

    for e in eps:
        r, st = meta.get(e, {}), states[e]
        pid = (r.get("placement_id") or f"ep{e}").strip()
        close = F.grasp_frame(st, "first")
        rim = tip_at_rim(fit, st, close, a.rim_cm)
        head = f"{pid}  ep{e}  {r.get('outcome', '')}  [{r.get('mechanism', '')}]"
        dst = out_dir / f"{pid}_ep{e}.mp4"
        with av.open(str(dst), "w") as oc:
            vs = None
            for k, (wr, fl) in enumerate(zip(*(episode_frames(root, c, epmeta.loc[e], fps) for c in CAMS))):
                tiles = [cv2.resize(x, (int(x.shape[1] * H / x.shape[0]), H)) for x in (wr, fl)]
                img = np.hstack(tiles)
                img = np.vstack([np.zeros((34, img.shape[1], 3), np.uint8), img, np.zeros((34, img.shape[1], 3), np.uint8)])
                if rim is not None and close is not None and rim <= k <= close:
                    img[-34:] = (0, 120, 230)
                text(img, short, (8, 14), (180, 180, 180), 0.4)
                text(img, head, (8, 30), (255, 255, 255), 0.55)
                text(img, "wrist", (8, 54)); text(img, "front-left", (tiles[0].shape[1] + 8, 54))
                ev = "  ".join(s for s, f in (("TIP@RIM", rim), ("CLOSE", close)) if f is not None and k >= f)
                text(img, f"f {k:3d}  {k / fps:5.2f} s   grip {st[k, 5]:5.1f}  roll {st[k, 4]:5.1f}   "
                          f"TIP@RIM f{rim}  CLOSE f{close}   {ev}", (8, img.shape[0] - 11))
                if vs is None:
                    vs = oc.add_stream("libx264", rate=max(1, int(round(fps / a.slow))))
                    vs.width, vs.height, vs.pix_fmt = img.shape[1] // 2 * 2, img.shape[0] // 2 * 2, "yuv420p"
                frame = av.VideoFrame.from_ndarray(img[:vs.height, :vs.width], format="bgr24")
                for p in vs.encode(frame):
                    oc.mux(p)
            for p in vs.encode():
                oc.mux(p)
        print(f"{dst}  ({len(st)} frames, TIP@RIM f{rim}, CLOSE f{close})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
