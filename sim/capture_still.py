#!/usr/bin/env python
"""Save one still from a robot camera at EXACTLY the resolution its intrinsics were measured at.

`lerobot-find-cameras` saves stills at whatever size the device defaults to, which can differ from the
YAML's width/height. Intrinsics scale with resolution, so feeding such a still to
calib_extrinsics_aruco.py gives numbers that look fine and are wrong. This opens the camera from the same
YAML block calib_intrinsics_*.py reads (serial / index / backend / fourcc / size), discards warm-up frames so
auto-exposure settles, and writes one PNG. No GUI needed (works on an opencv-python-headless install).

    uv run python sim/capture_still.py --camera wrist                # configs/teleoperate_omx.yaml
    uv run python sim/capture_still.py --camera front-left           # configs/record_omx.yaml

Opens the camera and releases it on exit, so run it between (not during) other scripts that use the same
camera. 🔴 Not run against real hardware where it was written -- report back anything that breaks.
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path

import cv2
import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = {"wrist": "teleoperate_omx.yaml", "front-left": "record_omx.yaml"}
_BACKENDS = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "any": cv2.CAP_ANY, "v4l2": cv2.CAP_V4L2}


def load_block(config_path: Path, camera: str) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cams = (raw.get("robot") or {}).get("cameras") or {}
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not in {config_path} (has: {sorted(cams)})")
    return cams[camera]


def grab_realsense(block: dict, warmup: int) -> np.ndarray:
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(str(block["serial_number_or_name"]))
    cfg.enable_stream(rs.stream.color, block["width"], block["height"], rs.format.bgr8, block.get("fps", 15))
    pipe.start(cfg)
    try:
        for _ in range(warmup):
            pipe.wait_for_frames()
        return np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).copy()
    finally:
        pipe.stop()


def grab_uvc(block: dict, index: int | None, backend: str | None, warmup: int) -> np.ndarray:
    idx = index if index is not None else block["index_or_path"]
    cap = cv2.VideoCapture(idx, _BACKENDS[(backend or block.get("backend", "any")).lower()])
    if block.get("fourcc"):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*block["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, block["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, block["height"])
    if not cap.isOpened():
        raise SystemExit(f"could not open camera index {idx} -- run `uv run lerobot-find-cameras opencv` and pass --index")
    try:
        frame = None
        for _ in range(warmup):
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("camera read failed -- check --index / --backend")
        return frame
    finally:
        cap.release()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--camera", required=True, choices=sorted(_DEFAULT_CONFIG))
    ap.add_argument("--config", type=Path, default=None, help="default: teleoperate_omx.yaml for wrist, record_omx.yaml for front-left")
    ap.add_argument("--index", type=int, default=None, help="UVC only: override the config's index_or_path")
    ap.add_argument("--backend", default=None, choices=list(_BACKENDS), help="UVC only: override the config's backend")
    ap.add_argument("--warmup", type=int, default=30, help="frames discarded so auto-exposure settles")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print what would be opened and exit; camera NOT opened")
    args = ap.parse_args(argv)

    config = args.config or (_REPO / "configs" / _DEFAULT_CONFIG[args.camera])
    block = load_block(config, args.camera)
    want = (block["width"], block["height"])
    is_rs = block.get("type") in ("intelrealsense", "intelrealsense_pinned")
    print(f"camera '{args.camera}' from {config}: {'RealSense ' + str(block['serial_number_or_name']) if is_rs else 'UVC index ' + str(args.index if args.index is not None else block['index_or_path'])} {want[0]}x{want[1]}")
    if args.dry_run:
        print("dry run: camera NOT opened.")
        return 0

    frame = grab_realsense(block, args.warmup) if is_rs else grab_uvc(block, args.index, args.backend, args.warmup)
    got = (frame.shape[1], frame.shape[0])
    if got != want:
        raise SystemExit(f"got {got[0]}x{got[1]}, expected {want[0]}x{want[1]} -- not saved: intrinsics for {want[0]}x{want[1]} do not apply to this size")

    out = args.out or (_REPO / "calibration" / "shots" / f"{dt.datetime.now():%Y-%m-%d_%H%M%S}_{args.camera}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blown = float((gray >= 250).mean())
    print(f"wrote {out}  ({got[0]}x{got[1]}, mean brightness {gray.mean():.0f}/255, {blown * 100:.1f}% of pixels >= 250)")
    if blown > 0.2:
        print("⚠️ more than 20% of the frame is blown out -- markers on a bright surface may not be detected; dim the light or move the camera")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
