#!/usr/bin/env python
"""Find fixed exposure / gain / white balance that reproduce what the D455's AUTO mode shows, for the YAML.

Why not simply "freeze": on this D455, switching auto off jumps to the sensor defaults (exposure 156 / gain 64 /
WB 4600, brightness -37 %), and per-frame metadata is not available on this laptop (D022 §2026-09-13). So the
values are MATCHED against an AUTO reference instead:

  1. AUTO reference: mean brightness, B/R and G/R of the scene exactly as it will be recorded.
  2. Exposure: scan at the default gain. Exposure is quantised in coarse steps here (power_line_frequency = Auto,
     anti-flicker), so take the brightest scanned step that stays at or below AUTO.
  3. Gain: scan at that exposure and interpolate to AUTO's brightness (gain is fine-grained, 0..128).
  4. White balance: stays AUTO by default - fixed WB left a green cast on BOTH cameras under our lamp (D022);
     `--wb match` sweeps temperatures and keeps the one whose B/R is closest to AUTO's.
  5. Verify the pinned values against AUTO on brightness, B/R AND G/R (B/R alone missed a green cast on the wrist
     camera, 2026-09-13), then measure AUTO again to catch a scene change during the run.

Proven by hand on 2026-09-13: exposure 400 / gain 70 / 3500 K -> brightness 0.0 %, B/R 4.2 % from AUTO.

The camera is opened with pyrealsense2 at the config's serial / resolution / fps. AUTO is switched back on at exit
unless --keep: lerobot leaves omitted RealSense options unchanged, so a config without values would inherit them.
"""

import argparse
import dataclasses
import io
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]

EXPOSURE_SCAN = (20, 40, 80, 160, 320, 640, 1000)
GAIN_SCAN = (16, 32, 48, 64, 80, 96, 112, 128)
DEFAULT_GAIN = 64
WB_SWEEP_MIN, WB_SWEEP_MAX = 2800, 6500
TOLERANCE = 0.10  # pinned vs AUTO on brightness, B/R and G/R; also the allowed AUTO drift during the run


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
    if block.get("type") not in ("intelrealsense", "intelrealsense_pinned"):
        raise SystemExit(f"camera '{camera}' is type '{block.get('type')}', not an intelrealsense camera")
    import lerobot_camera_uvc  # noqa: F401  (registers `type: intelrealsense_pinned`)
    cfg = draccus.load(CameraConfig, io.StringIO(yaml.safe_dump(block)))
    return dataclasses.replace(cfg, exposure=None, gain=None, white_balance=None)


def pick_exposure(scan: list[tuple[int, float]], target: float) -> int:
    """Brightest scanned exposure whose brightness (at the default gain) is at or below the target."""
    below = [e for e, m in scan if m <= target]
    return max(below) if below else min(e for e, _ in scan)


def interp_gain(scan: list[tuple[int, float]], target: float) -> int:
    """Gain whose brightness hits the target, by linear interpolation between scanned points (clamped)."""
    pts = sorted(scan)
    if target <= pts[0][1]:
        return pts[0][0]
    for (g1, m1), (g2, m2) in zip(pts, pts[1:]):
        if m1 <= target <= m2:
            return int(round(g1 + (g2 - g1) * (target - m1) / max(m2 - m1, 1e-6)))
    return pts[-1][0]


def pick_wb(sweep: list[tuple[int, float]], target_br: float) -> int:
    """Colour temperature whose measured B/R is closest to AUTO white balance's B/R."""
    return min(sweep, key=lambda tb: abs(tb[1] - target_br))[0]


def match_check(auto: tuple[float, float, float], fixed: tuple[float, float, float]) -> tuple[bool, list[float]]:
    """(ok, [brightness, B/R, G/R relative changes]) of the pinned values against the AUTO reference."""
    diffs = [abs(f - a) / max(abs(a), 1e-6) for a, f in zip(auto, fixed)]
    return all(d <= TOLERANCE for d in diffs), diffs


def drift(m_before: float, m_after: float) -> float:
    return abs(m_after - m_before) / max(m_before, 1.0)


def yaml_snippet(values: dict, camera: str) -> str:
    wb = "null" if values["white_balance"] is None else values["white_balance"]
    return (
        f"# `{camera}:` block of configs/record_omx.yaml AND configs/teleoperate_omx.yaml\n"
        f"      exposure: {values['exposure']}\n"
        f"      gain: {values['gain']}\n"
        f"      white_balance: {wb}"
    )


def run_live(cfg, args) -> int:
    import pyrealsense2 as rs

    pipe, rs_cfg = rs.pipeline(), rs.config()
    rs_cfg.enable_device(str(cfg.serial_number_or_name))
    rs_cfg.enable_stream(rs.stream.color, cfg.width, cfg.height, rs.format.rgb8, cfg.fps)
    sensor = pipe.start(rs_cfg).get_device().first_color_sensor()
    O = rs.option

    def measure(n: int = 12, settle_s: float = 1.2) -> tuple[float, float, float]:
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < settle_s:  # a changed option lands a few frames later
            pipe.wait_for_frames()
        acc = [np.asanyarray(pipe.wait_for_frames().get_color_frame().get_data()).reshape(-1, 3).mean(0) for _ in range(n)]
        r, g, b = np.mean(acc, 0)  # RGB8
        return float(np.mean([r, g, b])), float(b / max(r, 1e-6)), float(g / max(r, 1e-6))

    def auto() -> tuple[float, float, float]:
        sensor.set_option(O.enable_auto_exposure, 1)
        sensor.set_option(O.enable_auto_white_balance, 1)
        return measure(30, settle_s=args.settle_s)

    def manual(e: int, g: int, wb: int | None) -> tuple[float, float, float]:
        sensor.set_option(O.enable_auto_exposure, 0)
        if wb is None:  # white balance stays AUTO while exposure / gain are scanned
            sensor.set_option(O.enable_auto_white_balance, 1)
        else:
            sensor.set_option(O.enable_auto_white_balance, 0)
            sensor.set_option(O.white_balance, float(wb))
        sensor.set_option(O.exposure, float(e))
        sensor.set_option(O.gain, float(g))
        return measure()

    try:
        input("\nSet the scene as for recording (lights, object in place, arm at the approach pose), then press Enter...")
        print("keep the scene still, ~60 s", flush=True)
        ref = auto()
        m_auto, br_auto = ref[0], ref[1]
        print(f"[1] AUTO reference: brightness {m_auto:.1f}  B/R {br_auto:.2f}", flush=True)

        wb0 = None if args.wb == "auto" else 4600
        e_scan = [(e, manual(e, DEFAULT_GAIN, wb0)[0]) for e in EXPOSURE_SCAN]
        e = pick_exposure(e_scan, m_auto)
        print("[2] exposure scan @ gain 64: " + "  ".join(f"{x}:{m:.0f}" for x, m in e_scan) + f"  -> {e}", flush=True)

        g_scan = [(g, manual(e, g, wb0)[0]) for g in GAIN_SCAN]
        g = interp_gain(g_scan, m_auto)
        print(f"[3] gain scan @ exposure {e}: " + "  ".join(f"{x}:{m:.0f}" for x, m in g_scan) + f"  -> {g}", flush=True)

        if args.wb == "auto":
            wb = None
            print("[4] white balance stays AUTO (fixed WB left a green cast under our lamp, D022)", flush=True)
        else:
            sweep = [(t, manual(e, g, t)[1]) for t in range(WB_SWEEP_MIN, WB_SWEEP_MAX + 1, 300)]
            coarse = pick_wb(sweep, br_auto)
            sweep += [(t, manual(e, g, t)[1]) for t in range(coarse - 200, coarse + 201, 100)]
            wb = pick_wb(sweep, br_auto)
            print(f"[4] white balance -> {wb} K", flush=True)

        fixed = manual(e, g, wb)
        ok, (dm, dbr, dgr) = match_check(ref, fixed)
        m_after = auto()[0]
        d = drift(m_auto, m_after)
        print(
            f"[5] pinned exposure={e} gain={g} white_balance={wb}: brightness {m_auto:.1f} -> {fixed[0]:.1f} "
            f"({dm * 100:.1f} %), B/R {br_auto:.2f} -> {fixed[1]:.2f} ({dbr * 100:.1f} %), "
            f"G/R {ref[2]:.2f} -> {fixed[2]:.2f} ({dgr * 100:.1f} %) -> {'OK' if ok else 'MISMATCH'}"
        )
        print(f"    AUTO after the run: {m_after:.1f} (scene drift {d * 100:.1f} %) -> {'steady' if d <= TOLERANCE else 'SCENE CHANGED'}")
    finally:
        if not args.keep:
            sensor.set_option(O.enable_auto_exposure, 1)
            sensor.set_option(O.enable_auto_white_balance, 1)
            print("auto-exposure / auto white balance restored to ON")
        pipe.stop()

    print("\nvalues:\n" + yaml_snippet(dict(exposure=e, gain=g, white_balance=wb), args.camera))
    if not ok:
        print("\nWARNING: pinned image differs from AUTO by more than 10 % - do NOT use these values.")
        return 2
    if d > TOLERANCE:
        print("\nWARNING: the scene / light changed during the run - redo it with the scene still.")
        return 1
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "record_omx.yaml")
    ap.add_argument("--camera", default="front-left")
    ap.add_argument(
        "--wb",
        choices=["auto", "match"],
        default="auto",
        help="white balance: keep AUTO (default; fixed WB looked green on both cameras, D022) or match a fixed value",
    )
    ap.add_argument("--settle-s", type=float, default=4.0, help="seconds of AUTO before each AUTO measurement")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="leave the camera on the pinned manual values at exit (default: AUTO back on; lerobot leaves omitted "
        "options unchanged, so a config without values would silently inherit them)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; the camera is NOT opened")
    args = ap.parse_args(argv)

    cfg = load_camera_config(args.config, args.camera)
    print(
        f"camera '{args.camera}' from {args.config}: serial={cfg.serial_number_or_name} "
        f"{cfg.width}x{cfg.height}@{cfg.fps}; AUTO reference -> exposure scan -> gain -> WB -> verify"
    )
    if args.dry_run:
        print("dry run: camera NOT opened.")
        return 0
    return run_live(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
