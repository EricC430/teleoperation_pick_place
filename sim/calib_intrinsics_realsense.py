#!/usr/bin/env python
"""T1 (S4 §5-5) for the D455: read real intrinsics off the device, not off a datasheet FOV.

Resolution / serial number are read from a lerobot camera YAML's `front-left` block, not
hardcoded here, so this always calibrates at the SAME settings the camera actually records at
(intrinsics scale with resolution -- calibrating at the wrong one gives numbers that look
plausible and are wrong, the same class of silent error this repo has been bitten by before:
docs/CLAUDE.md's mimic-gearing and joint-direction incidents).

Also quantifies how far the D455's own (Brown-Conrady) distortion would displace a pixel from
where an IDEAL pinhole camera would put it -- Isaac Sim's camera has no distortion model, so this
is the residual error floor T3's reprojection-error acceptance number has to absorb. Uses
pyrealsense2's own rs2_deproject/rs2_project functions (the SDK's own model), not an assumed
OpenCV equivalent, to avoid a second, possibly-mismatched distortion model.

🔴 Not executed against real hardware in the environment this was written in (no pyrealsense2 /
no camera attached). The pyrealsense2 calls below follow its documented, stable API and match the
style already used in scripts/freeze_realsense_exposure.py, but this has NOT been run end-to-end
-- run it on the lab machine and report back errors so this note can be removed. `[AI推論，未實測]`
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]


def load_camera_block(config_path: Path, camera: str) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cams = (raw.get("robot") or {}).get("cameras") or {}
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not in {config_path} (has: {sorted(cams)})")
    block = cams[camera]
    if block.get("type") not in ("intelrealsense", "intelrealsense_pinned"):
        raise SystemExit(f"camera '{camera}' is type '{block.get('type')}', not a RealSense camera -- use calib_intrinsics_checkerboard.py instead")
    return block


def read_intrinsics(serial: str, width: int, height: int, fps: int):
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(str(serial))
    cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
    profile = pipe.start(cfg)
    try:
        for _ in range(10):  # let AE/AWB settle; we only need the stream profile, not exposure
            pipe.wait_for_frames()
        intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        return intr
    finally:
        pipe.stop()


def distortion_displacement_px(intr, n_grid: int = 9) -> np.ndarray:
    """Per-sample-point |distorted_pixel - ideal_pinhole_pixel|, evaluated on an n_grid x n_grid
    grid spanning the sensor. Uses the SDK's own (de)projection so the comparison is against
    exactly the model the D455 reports, not an assumed one."""
    import pyrealsense2 as rs

    ideal = rs.intrinsics()
    ideal.width, ideal.height = intr.width, intr.height
    ideal.fx, ideal.fy, ideal.ppx, ideal.ppy = intr.fx, intr.fy, intr.ppx, intr.ppy
    ideal.model = rs.distortion.none
    ideal.coeffs = [0.0] * 5

    disps = []
    for v in np.linspace(0, intr.height - 1, n_grid):
        for u in np.linspace(0, intr.width - 1, n_grid):
            point = rs.rs2_deproject_pixel_to_point(intr, [float(u), float(v)], 1.0)
            ideal_px = rs.rs2_project_point_to_pixel(ideal, point)
            disps.append(float(np.hypot(ideal_px[0] - u, ideal_px[1] - v)))
    return np.array(disps)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "record_omx.yaml")
    ap.add_argument("--camera", default="front-left")
    ap.add_argument("--max-accept-px", type=float, default=5.0, help="distortion displacement above this -> WARN, do not silently ignore it in T3")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; camera NOT opened")
    args = ap.parse_args(argv)

    block = load_camera_block(args.config, args.camera)
    serial, width, height, fps = block["serial_number_or_name"], block["width"], block["height"], block.get("fps", 15)
    print(f"camera '{args.camera}' from {args.config}: serial={serial} {width}x{height}@{fps}")

    if args.dry_run:
        print("dry run: camera NOT opened.")
        return 0

    intr = read_intrinsics(serial, width, height, fps)
    print(f"fx={intr.fx:.3f} fy={intr.fy:.3f} ppx={intr.ppx:.3f} ppy={intr.ppy:.3f} model={intr.model} coeffs={list(intr.coeffs)}")

    disp = distortion_displacement_px(intr)
    p50, p95, pmax = float(np.percentile(disp, 50)), float(np.percentile(disp, 95)), float(disp.max())
    verdict = "OK -- negligible, T3 can compare against the ideal pinhole directly" if pmax <= args.max_accept_px else "WARN -- undistort real T2/T3 images before comparing to the sim's ideal-pinhole render"
    print(f"distortion-induced pixel displacement across the frame: p50={p50:.2f} p95={p95:.2f} max={pmax:.2f} px -> {verdict}")

    result = {
        "camera": args.camera,
        "serial": serial,
        "width": width,
        "height": height,
        "fps": fps,
        "fx": intr.fx,
        "fy": intr.fy,
        "cx": intr.ppx,
        "cy": intr.ppy,
        "distortion_model": str(intr.model),
        "distortion_coeffs": list(intr.coeffs),
        "distortion_px_p50": p50,
        "distortion_px_p95": p95,
        "distortion_px_max": pmax,
        "measured_at": dt.date.today().isoformat(),
        "source_config": str(args.config),
    }
    out = args.out or (_REPO / "calibration" / f"{dt.date.today().isoformat()}_camera_intrinsics_{args.camera}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
