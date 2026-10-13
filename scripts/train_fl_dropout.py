#!/usr/bin/env python
"""lerobot-train with front-left dropout in the approach phase only (2026-10-13).

Same arguments as `lerobot-train`, plus the flags below. In the last --fl-window frames before an episode's FIRST
gripper close (action gripper < --close-below), the front-left image is replaced by black with probability --fl-p.
Everything else -- other frames, the wrist image, the image_transforms -- is untouched. Why and how to judge it:
docs/handoff/2026-10-13_fl_dropout.md.

    ./scripts/run_container.sh python scripts/train_fl_dropout.py --fl-p 0.3 --fl-window 30 \
        --config_path configs/train_omx_cup_uvc60_normal_tightcut_far_A1_aug_fldrop.yaml

- No lerobot source is modified (docs/environment.md): train() is lerobot's own; only the train dataset returned by
  make_train_eval_datasets is switched to a subclass whose __getitem__ masks the frame. The subclass lives at module
  level so the spawn dataloader workers can unpickle it.
- The mask is applied after image_transforms, on the uint8 frame (0 = black); lerobot divides by 255 and normalizes
  later, exactly as for a real frame.
- train_config.json does not know these flags: they are written to <output_dir>/fl_dropout.json and put in job_name.
  A --resume run must be given the same flags again.
- Episodes with no gripper close get no dropout.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

FL = "observation.images.front-left"
GRIPPER = 5  # action index (omx_follower: shoulder_pan .. wrist_roll, gripper)


class FLDropDataset(LeRobotDataset):
    """LeRobotDataset whose front-left frame is blacked out at random in the approach window."""

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        if isinstance(idx, slice):
            return item
        close = self._fl_close.get(int(item["episode_index"]), -1)
        f = int(item["frame_index"])
        if close >= 0 and close - self._fl_window <= f < close and torch.rand(()) < self._fl_p:
            item[self._fl_key] = torch.zeros_like(item[self._fl_key])
        return item


def first_close(ds: LeRobotDataset, close_below: float) -> dict[int, int]:
    """Episode index -> frame_index of the first frame whose action gripper is below close_below (-1: never)."""
    cols = ds.select_columns(["episode_index", "frame_index", "action"]).with_format("numpy")[:]
    ep, fr, grip = np.asarray(cols["episode_index"]), np.asarray(cols["frame_index"]), np.stack(cols["action"])[:, GRIPPER]
    out = {}
    for e in np.unique(ep):
        m = ep == e
        order = np.argsort(fr[m])
        closed = np.flatnonzero(grip[m][order] < close_below)
        out[int(e)] = int(fr[m][order][closed[0]]) if len(closed) else -1
    return out


def enable(ds: LeRobotDataset, p: float, window: int, close_below: float, key: str = FL) -> dict:
    if key not in ds.meta.camera_keys:
        raise SystemExit(f"{key} is not a camera of {ds.repo_id}: {ds.meta.camera_keys}")
    close = first_close(ds, close_below)
    ds.__class__ = FLDropDataset
    ds._fl_close, ds._fl_window, ds._fl_p, ds._fl_key = close, window, p, key
    valid = [c for c in close.values() if c >= 0]
    in_window = sum(min(window, c) for c in valid)
    return {
        "camera": key, "p": p, "window_frames": window, "close_below": close_below,
        "episodes": len(close), "episodes_without_close": [e for e, c in close.items() if c < 0],
        "frames": ds.num_frames, "frames_in_window": in_window,
        "expected_masked_fraction": round(p * in_window / ds.num_frames, 4),
        "first_close_frame": close,
    }


def parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--fl-p", type=float, required=True, help="probability of blacking out front-left in the window")
    p.add_argument("--fl-window", type=int, required=True, help="frames before the first gripper close")
    p.add_argument("--close-below", type=float, default=54.0,
                   help="action gripper below this = closed (open ~59-60, closed ~48 in the paper-cup data)")
    return p.parse_known_args(argv)


def main() -> None:
    args, rest = parse_args(sys.argv[1:])
    if not 0.0 < args.fl_p <= 1.0 or args.fl_window <= 0:
        raise SystemExit("--fl-p must be in (0, 1] and --fl-window > 0")
    sys.argv = [sys.argv[0], *rest]  # lerobot's parser must not see our flags

    import lerobot.scripts.lerobot_train as lt

    orig = lt.make_train_eval_datasets

    def make_train_eval_datasets(cfg):
        train_ds, eval_ds = orig(cfg)
        info = enable(train_ds, args.fl_p, args.fl_window, args.close_below)
        logging.info("front-left approach dropout: %s", {k: v for k, v in info.items() if k != "first_close_frame"})
        out = Path(cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fl_dropout.json").write_text(json.dumps(info, indent=1))
        return train_ds, eval_ds

    lt.make_train_eval_datasets = make_train_eval_datasets
    lt.main()


if __name__ == "__main__":
    main()
