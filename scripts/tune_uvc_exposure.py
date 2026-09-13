#!/usr/bin/env python
"""Live exposure / gain / white-balance tuner for the wrist UVC camera (`type: opencv_uvc`, D022 §2026-09-13).

Why a separate tool: the plugin applies exposure only at connect, so tuning through lerobot-teleoperate costs
a reconnect per value, and quitting teleop drops the arm (disable_torque_on_disconnect). Here the arm stays
torque-off and is posed BY HAND while this window shows the camera live.

The camera is driven through the plugin's own code (OpenCVUVCCamera._configure_capture_settings /
_apply_uvc_controls) and every value is validated by OpenCVUVCCameraConfig, so what you see is what
lerobot-record will apply. Reading and setting happen on one thread: no cap.set() races a reader.

Recommended (D022 §2026-09-13 option 2, [柏宇決定]): pose the grasp close-up, hold still, press `f` - it pins the
exposure / gain AUTO picks there (frozen on the device) and checks the result against AUTO. White balance stays
AUTO: manual WB leaves a green cast on this camera that no temperature removes (D022). Manual keys stay for when
`f` fails: DSHOW exposure steps double the brightness, so hand-matching is coarse.
"""

import argparse
import dataclasses
import io
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parents[1]

MANUAL_START_EXPOSURE, MANUAL_START_GAIN, MANUAL_START_WB = -6, 0, 4600
GAIN_STEP, WB_STEP = 5, 200
CLIP_LEVEL, CRUSH_LEVEL = 250, 10
FREEZE_SETTLE_S = 3.0  # seconds of AUTO before freezing
FREEZE_TOLERANCE = 0.10  # pinned result may differ from AUTO by at most 10 % in brightness and in B/R

KEY_HELP = (
    "keys:  a  auto <-> manual exposure    e / E  exposure -1 / +1 (log2 s: each step halves / doubles)\n"
    "       g / G  gain -5 / +5            b  auto <-> manual white balance    w / W  white balance -200 / +200 K\n"
    "       s  snapshot + print YAML       q  quit (restores AUTO unless --keep; prints the final YAML)\n"
    "       f  FREEZE the exposure / gain AUTO picks right now (WB stays AUTO; D022) - hold the arm still ~10 s"
)


@dataclass(frozen=True)
class Controls:
    exposure: int | None = None
    gain: int | None = None
    white_balance: int | None = None


def exposure_ms(exposure: int) -> float:
    return 2.0**exposure * 1e3


def describe(c: Controls) -> str:
    exp = "auto" if c.exposure is None else f"{c.exposure} ({exposure_ms(c.exposure):.1f} ms)"
    gain = "auto" if c.gain is None else str(c.gain)
    wb = "auto" if c.white_balance is None else f"{c.white_balance} K"
    return f"exposure={exp}  gain={gain}  white_balance={wb}"


def step(c: Controls, key: str) -> Controls | None:
    """Controls after a key press, or None if `key` is not a control key. Validation happens elsewhere."""
    if len(key) != 1:
        return None
    manual = c.exposure is not None
    if key == "a":
        if manual:
            return dataclasses.replace(c, exposure=None, gain=None)
        return dataclasses.replace(c, exposure=MANUAL_START_EXPOSURE, gain=MANUAL_START_GAIN)
    if key == "b":
        return dataclasses.replace(c, white_balance=None if c.white_balance is not None else MANUAL_START_WB)
    if key in "eE" and manual:
        return dataclasses.replace(c, exposure=c.exposure + (1 if key == "E" else -1))
    if key in "gG" and manual:
        return dataclasses.replace(c, gain=max(0, c.gain + (GAIN_STEP if key == "G" else -GAIN_STEP)))
    if key in "wW" and c.white_balance is not None:
        return dataclasses.replace(c, white_balance=c.white_balance + (WB_STEP if key == "W" else -WB_STEP))
    return None


def with_controls(cfg, c: Controls):
    """(config, None) if the plugin's validation accepts `c`, else (None, reason)."""
    try:
        return dataclasses.replace(cfg, **dataclasses.asdict(c)), None
    except ValueError as e:
        return None, str(e)


def frame_stats(frame: np.ndarray) -> dict:
    """Brightness stats on a HxWx3 uint8 frame. A pixel counts as clipped/crushed by its brightest channel."""
    peak = frame.max(axis=2)
    h, w = peak.shape
    centre = peak[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
    return {
        "mean": float(frame.mean()),
        "clipped_pct": float((peak >= CLIP_LEVEL).mean() * 100),
        "centre_clipped_pct": float((centre >= CLIP_LEVEL).mean() * 100),
        "crushed_pct": float((peak <= CRUSH_LEVEL).mean() * 100),
        "b_over_r": float(frame[..., 0].mean() / max(float(frame[..., 2].mean()), 1e-6)),  # BGR frame
    }


def freeze_check(m_auto: float, br_auto: float, m_fix: float, br_fix: float) -> tuple[bool, float, float]:
    """(ok, brightness change, B/R change) of the pinned values against the AUTO reference."""
    dm = abs(m_fix - m_auto) / max(m_auto, 1.0)
    dbr = abs(br_fix - br_auto) / max(br_auto, 1e-6)
    return dm <= FREEZE_TOLERANCE and dbr <= FREEZE_TOLERANCE, dm, dbr


def yaml_snippet(c: Controls, camera: str) -> str:
    def v(x):
        return "null" if x is None else str(x)

    return (
        f"# `{camera}:` block of configs/record_omx.yaml AND configs/teleoperate_omx.yaml\n"
        f"      type: opencv_uvc\n"
        f"      exposure: {v(c.exposure)}\n"
        f"      gain: {v(c.gain)}\n"
        f"      white_balance: {v(c.white_balance)}"
    )


def load_camera_config(config_path: Path, camera: str):
    """The camera block of a lerobot YAML as an OpenCVUVCCameraConfig (a plain `type: opencv` block is lifted)."""
    import draccus
    import lerobot_camera_uvc  # noqa: F401  (registers `type: opencv_uvc`)
    from lerobot.cameras.configs import CameraConfig

    raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    cams = (raw.get("robot") or {}).get("cameras") or {}
    if camera not in cams:
        raise SystemExit(f"camera '{camera}' not in {config_path} (has: {sorted(cams)})")
    block = dict(cams[camera])
    if block.get("type") not in ("opencv", "opencv_uvc"):
        raise SystemExit(f"camera '{camera}' is type '{block.get('type')}', not an OpenCV / UVC camera")
    block["type"] = "opencv_uvc"
    return draccus.load(CameraConfig, io.StringIO(yaml.safe_dump(block)))


def run_live(cfg, start: Controls, args) -> int:
    import tkinter as tk

    import cv2
    from lerobot_camera_uvc import OpenCVUVCCamera
    from PIL import Image, ImageTk

    cam = OpenCVUVCCamera(cfg)
    # Open the way OpenCVCamera.connect() does, but WITHOUT its background read thread.
    cv2.setNumThreads(1)
    cam.videocapture = cv2.VideoCapture(cfg.index_or_path, cfg.backend)
    if not cam.videocapture.isOpened():
        raise SystemExit(
            f"could not open camera index {cfg.index_or_path}: run `lerobot-find-cameras opencv` and check the index"
        )
    cam._configure_capture_settings()  # size / fps / fourcc, then the plugin applies the start controls

    args.out.mkdir(parents=True, exist_ok=True)
    state = {"c": start, "msg": "", "msg_t": 0.0, "frame": None}
    stamps: deque[float] = deque(maxlen=90)

    root = tk.Tk()
    root.title(f"tune {args.camera} (index {cfg.index_or_path})")
    img_label = tk.Label(root)
    img_label.pack()
    info = tk.Label(root, font=("Consolas", 11), justify="left", anchor="w")
    info.pack(fill="x", padx=6)
    tk.Label(root, text=KEY_HELP, font=("Consolas", 9), justify="left", anchor="w", fg="#555").pack(fill="x", padx=6)

    def flash(msg: str) -> None:
        state["msg"], state["msg_t"] = msg, time.perf_counter()
        print(msg, flush=True)

    def apply(new: Controls) -> None:
        cfg_new, err = with_controls(cam.config, new)
        if err:
            flash(f"rejected: {err}")
            return
        old = cam.config
        cam.config = cfg_new
        try:
            cam._apply_uvc_controls()
            state["c"] = new
            flash(f"applied: {describe(new)}")
        except RuntimeError as e:
            cam.config = old
            try:
                cam._apply_uvc_controls()
            except RuntimeError:
                pass
            flash(f"camera refused it, reverted: {e}")

    def snapshot() -> None:
        if state["frame"] is None:
            return
        c = state["c"]
        name = f"{args.camera}_e{c.exposure}_g{c.gain}_wb{c.white_balance}_{time.strftime('%H%M%S')}.png"
        cv2.imwrite(str(args.out / name), state["frame"])
        flash(f"saved {args.out / name}\n{yaml_snippet(c, args.camera)}")

    def close() -> None:
        if not args.keep:
            try:
                cam.config = dataclasses.replace(cam.config, exposure=None, gain=None, white_balance=None)
                cam._apply_uvc_controls()
                print("camera restored to AUTO (plain `type: opencv` configs would otherwise inherit the values)")
            except Exception as e:  # noqa: BLE001 - report it, but still release the camera
                print(f"WARNING: could not restore auto: {e}")
        cam.videocapture.release()
        print("\nfinal values:\n" + yaml_snippet(state["c"], args.camera), flush=True)
        root.destroy()

    def measure(secs: float, skip: int = 8) -> tuple[float, float]:
        for _ in range(skip):
            cam.videocapture.read()
        acc, t0 = [], time.perf_counter()
        while time.perf_counter() - t0 < secs:
            ok, fr = cam.videocapture.read()
            if ok:
                acc.append(fr.reshape(-1, 3).mean(0))
        b, _g, r = np.mean(acc, 0)
        return float(np.mean(acc)), float(b / max(r, 1e-6))

    def freeze_auto() -> None:
        """Option 2 (D022 §2026-09-13): pin the exposure / gain AUTO picks in the current pose. Probed on this camera
        2026-09-13: switching auto-exposure off without writing a value holds AUTO's exposure, and the read-back is
        then real. White balance is left on AUTO (see below)."""
        vc = cam.videocapture
        print("freeze: hold the arm still, ~10 s (the window pauses) ...", flush=True)
        cam.config = dataclasses.replace(cam.config, exposure=None, gain=None, white_balance=None)
        cam._apply_uvc_controls()
        time.sleep(FREEZE_SETTLE_S)
        m_auto, br_auto = measure(2.0)
        print(f"freeze: AUTO reference mean {m_auto:.1f}  B/R {br_auto:.2f}", flush=True)
        vc.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # manual, but no value written -> holds AUTO's choice
        time.sleep(0.8)
        e, g = int(round(vc.get(cv2.CAP_PROP_EXPOSURE))), int(round(vc.get(cv2.CAP_PROP_GAIN)))
        print(f"freeze: exposure {e} ({exposure_ms(e):.1f} ms), gain {g}; white balance stays AUTO", flush=True)
        # White balance stays AUTO. This camera's manual WB is a single colour-temperature axis and leaves a green
        # cast under our lamp that no temperature removes (2026-09-13 sweep 2800-6400 K: G/R >= 0.93 everywhere vs
        # AUTO 0.92 at B/R 0.95; per-channel WB controls unsupported). Exposure/gain fixed + WB AUTO matched full
        # AUTO (B/R 0.94, G/R 0.92). Manual WB stays available on the `b` / `w` keys.
        new = Controls(e, g, None)
        cfg_new, err = with_controls(cam.config, new)
        if err:
            flash(f"freeze: AUTO picked values the plugin rejects ({err}); use the manual keys")
            cam._apply_uvc_controls()  # cam.config is still all-None -> back to AUTO
            return
        cam.config = cfg_new
        cam._apply_uvc_controls()  # written exactly the way lerobot will write them on connect
        state["c"] = new
        m_fix, br_fix = measure(1.5)
        ok, dm, dbr = freeze_check(m_auto, br_auto, m_fix, br_fix)
        warn = ""
        if 2.0**e >= 1.0 / 30:
            warn = f"\nWARNING: {exposure_ms(e):.0f} ms >= one frame at 30 fps -> teleoperate_omx.yaml (fps 30) will reject it"
        flash(
            f"freeze: {describe(new)} | mean {m_auto:.1f} -> {m_fix:.1f} ({dm * 100:.1f} %), "
            f"B/R {br_auto:.2f} -> {br_fix:.2f} ({dbr * 100:.1f} %) -> {'OK' if ok else 'MISMATCH - do not use'}{warn}\n"
            f"{yaml_snippet(new, args.camera)}"
        )

    def on_key(event) -> None:
        k = event.char
        if k in ("q", "\x1b"):
            close()
        elif k == "s":
            snapshot()
        elif k == "f":
            freeze_auto()
        else:
            new = step(state["c"], k)
            if new is not None:
                apply(new)

    def tick() -> None:
        ok, frame = cam.videocapture.read()
        if ok:
            stamps.append(time.perf_counter())
            state["frame"] = frame
            st = frame_stats(frame)
            span = stamps[-1] - stamps[0]
            fps = (len(stamps) - 1) / span if span > 0 else 0.0
            photo = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            img_label.configure(image=photo)
            img_label.image = photo
            msg = state["msg"] if time.perf_counter() - state["msg_t"] < 4 else ""
            info.configure(
                text=f"{describe(state['c'])}\n"
                f"mean {st['mean']:5.1f}   B/R {st['b_over_r']:4.2f}   clipped {st['clipped_pct']:5.2f} % (centre {st['centre_clipped_pct']:5.2f} %)"
                f"   crushed {st['crushed_pct']:5.2f} %   camera {fps:4.1f} fps (record needs >= {cfg.fps})\n{msg}"
            )
        root.after(1, tick)

    root.bind("<Key>", on_key)
    root.protocol("WM_DELETE_WINDOW", close)
    root.after(1, tick)
    root.mainloop()
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], epilog=KEY_HELP, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", type=Path, default=_REPO / "configs" / "record_omx.yaml")
    ap.add_argument("--camera", default="wrist")
    ap.add_argument("--out", type=Path, default=_REPO / "outputs" / "exposure_tuning", help="snapshot folder")
    ap.add_argument(
        "--keep",
        action="store_true",
        help="leave the last values in the camera on exit (default: restore AUTO; UVC settings persist in the "
        "device, and a plain `type: opencv` config would silently inherit them)",
    )
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; the camera is NOT opened")
    args = ap.parse_args(argv)

    cfg = load_camera_config(args.config, args.camera)
    start = Controls(cfg.exposure, cfg.gain, cfg.white_balance)
    print(
        f"camera '{args.camera}' from {args.config}: index={cfg.index_or_path} backend={cfg.backend.name} "
        f"{cfg.width}x{cfg.height}@{cfg.fps} fourcc={cfg.fourcc}"
    )
    print(f"start: {describe(start)}")
    if args.dry_run:
        print("dry run: camera NOT opened.")
        print(KEY_HELP)
        return 0
    return run_live(cfg, start, args)


if __name__ == "__main__":
    raise SystemExit(main())
