#!/usr/bin/env python
"""Read the exposure / gain / white balance that the D455's AUTO mode picks, so they can be frozen in the YAML.

Method (lerobot's own RealSenseCameraConfig docstring: disabling auto-exposure "freezes exposure at its current
value"): let auto-exposure + auto white balance settle on the real scene, switch both off, read the options back.
The numbers are in the same units lerobot writes, so they paste straight into the `front-left:` block.

Self-check: mean brightness just before and just after the freeze must match; if it does not, the freeze did not
hold on this sensor and the numbers are not trustworthy (exit code 2).

The camera is opened with lerobot's RealSenseCamera (same serial / resolution / fps as recording). On exit both
auto modes are switched back ON unless --keep: lerobot leaves an omitted option UNCHANGED, so a config without
values would otherwise silently inherit the frozen ones.
"""

import argparse
import dataclasses
import io
import statistics
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]

BRIGHTNESS_TOLERANCE = 0.10  # frozen vs auto mean brightness may differ by at most 10 %
STABLE_SPREAD = 0.10  # repeated exposure readings must stay within 10 % of their median


def load_camera_config(config_path: Path, camera: str):
    """The camera block of a lerobot YAML as a RealSenseCameraConfig, with any manual values stripped."""
    import draccus
    from lerobot.cameras.configs import CameraConfig
    from lerobot.cameras.realsense import RealSenseCameraConfig  # noqa: F401  (registers `type: intelrealsense`)

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cams = (raw.get("robot") or {}).get("cameras") or {}
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not in {config_path} (has: {sorted(cams)})")
    block = dict(cams[camera])
    if block.get("type") != "intelrealsense":
        raise SystemExit(f"camera '{camera}' is type '{block.get('type')}', not an intelrealsense camera")
    cfg = draccus.load(CameraConfig, io.StringIO(yaml.safe_dump(block)))
    # Start from pure AUTO: never let connect() apply manual values from the config.
    return dataclasses.replace(cfg, exposure=None, gain=None, white_balance=None)


def brightness_check(auto_mean: float, frozen_mean: float) -> tuple[bool, float]:
    """(ok, relative difference) between the auto and frozen mean brightness."""
    diff = abs(frozen_mean - auto_mean) / max(auto_mean, 1.0)
    return diff <= BRIGHTNESS_TOLERANCE, diff


def summarize(readings: list[dict]) -> tuple[dict, bool]:
    """Median of each option across repeats, and whether exposure stayed within STABLE_SPREAD of its median."""
    med = {k: int(round(statistics.median(r[k] for r in readings))) for k in ("exposure", "gain", "white_balance")}
    exps = [r["exposure"] for r in readings]
    stable = all(abs(e - med["exposure"]) <= STABLE_SPREAD * max(abs(med["exposure"]), 1) for e in exps)
    return med, stable


def yaml_snippet(values: dict, camera: str) -> str:
    return (
        f"# `{camera}:` block of configs/record_omx.yaml AND configs/teleoperate_omx.yaml\n"
        f"      exposure: {values['exposure']}\n"
        f"      gain: {values['gain']}\n"
        f"      white_balance: {values['white_balance']}"
    )


def _mean_brightness(cam, seconds: float) -> float:
    vals, t0 = [], time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        vals.append(float(cam.async_read(timeout_ms=1000).mean()))
    return float(np.mean(vals))


def run_live(cfg, args) -> int:
    import pyrealsense2 as rs
    from lerobot.cameras.realsense import RealSenseCamera

    cam = RealSenseCamera(cfg)
    cam.connect()
    sensor = cam._get_color_sensor()
    auto_on = lambda on: (  # noqa: E731
        sensor.set_option(rs.option.enable_auto_exposure, 1 if on else 0),
        sensor.set_option(rs.option.enable_auto_white_balance, 1 if on else 0),
    )
    rng = sensor.get_option_range(rs.option.exposure)
    print(f"exposure option: range {rng.min:g}..{rng.max:g} (step {rng.step:g}, default {rng.default:g})")
    print(f"  description: {sensor.get_option_description(rs.option.exposure)}")
    print(f"  one frame at {cfg.fps} fps = {1000 / cfg.fps:.1f} ms — check the exposure unit before judging blur")

    readings: list[dict] = []
    try:
        auto_on(True)
        input(
            "\nSet the scene as for recording (lights, object in place, arm at the approach pose), then press Enter..."
        )
        for i in range(args.repeats):
            auto_on(True)
            time.sleep(args.settle_s)  # let auto-exposure / auto-WB converge
            auto_mean = _mean_brightness(cam, 1.5)
            auto_on(False)  # freeze
            time.sleep(0.5)
            frozen_mean = _mean_brightness(cam, 1.5)
            r = {
                "exposure": sensor.get_option(rs.option.exposure),
                "gain": sensor.get_option(rs.option.gain),
                "white_balance": sensor.get_option(rs.option.white_balance),
            }
            ok, diff = brightness_check(auto_mean, frozen_mean)
            r["ok"] = ok
            readings.append(r)
            print(
                f"[{i + 1}/{args.repeats}] exposure={r['exposure']:g} gain={r['gain']:g} "
                f"white_balance={r['white_balance']:g} | brightness auto {auto_mean:.1f} -> frozen {frozen_mean:.1f} "
                f"({diff * 100:.1f} %) {'OK' if ok else 'MISMATCH'}"
            )
    finally:
        if not args.keep:
            auto_on(True)
            print("auto-exposure / auto white balance restored to ON")
        cam.disconnect()

    values, stable = summarize(readings)
    all_ok = all(r["ok"] for r in readings)
    print("\nsuggested values (median):\n" + yaml_snippet(values, args.camera))
    if not stable:
        print("\nWARNING: exposure changed between repeats — the scene or light was not steady; redo it.")
    if not all_ok:
        print("\nWARNING: brightness changed at the freeze — the freeze did not hold; do NOT trust these values.")
        return 2
    return 0 if stable else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "record_omx.yaml")
    ap.add_argument("--camera", default="front-left")
    ap.add_argument("--repeats", type=int, default=3, help="auto -> freeze -> read cycles (default 3)")
    ap.add_argument("--settle-s", type=float, default=5.0, help="seconds of auto mode before each freeze")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="leave the camera frozen on exit (default: auto back ON; lerobot leaves omitted options unchanged, "
        "so a config without values would silently inherit the frozen ones)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; the camera is NOT opened")
    args = ap.parse_args(argv)

    cfg = load_camera_config(args.config, args.camera)
    print(
        f"camera '{args.camera}' from {args.config}: serial={cfg.serial_number_or_name} "
        f"{cfg.width}x{cfg.height}@{cfg.fps}; {args.repeats} x (auto {args.settle_s:g} s -> freeze -> read)"
    )
    if args.dry_run:
        print("dry run: camera NOT opened.")
        return 0
    return run_live(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
