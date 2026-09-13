"""Unit checks for the `opencv_uvc` camera plugin (plugins/lerobot_camera_uvc, D022 §2026-09-13).

No hardware: the capture object is faked. What the real Innomaker does with these calls was measured on
2026-09-13 and is recorded in D022; this file holds the plugin's logic to that contract.
"""

import importlib.metadata
import io
from pathlib import Path

import cv2
import draccus
import pytest

from lerobot.cameras.configs import CameraConfig, Cv2Backends
from lerobot.cameras.opencv import OpenCVCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_camera_uvc import OpenCVUVCCamera, OpenCVUVCCameraConfig

_REPO = Path(__file__).resolve().parents[1]


class FakeCapture:
    """Records set() calls; get() echoes the stored value unless `clamp` overrides it (a rejected value)."""

    def __init__(self, clamp=None):
        self.props, self.calls, self.clamp = {}, [], clamp or {}

    def set(self, prop, value):
        self.calls.append((prop, value))
        self.props[prop] = self.clamp.get(prop, value)
        return True

    def get(self, prop):
        return self.props.get(prop, -1.0)


def _cfg(**kw):
    base = dict(index_or_path=3, backend=Cv2Backends.DSHOW, fps=30, width=640, height=480, fourcc="YUY2")
    return OpenCVUVCCameraConfig(**{**base, **kw})


def _applied(cfg, clamp=None):
    cam = OpenCVUVCCamera(cfg)
    cam.videocapture = FakeCapture(clamp)
    cam._apply_uvc_controls()
    return cam.videocapture.calls


def test_manual_values_are_applied_in_order():
    assert _applied(_cfg(exposure=-6, gain=10, white_balance=4600)) == [
        (cv2.CAP_PROP_AUTO_EXPOSURE, 0.25),
        (cv2.CAP_PROP_EXPOSURE, -6.0),
        (cv2.CAP_PROP_GAIN, 10.0),
        (cv2.CAP_PROP_AUTO_WB, 0.0),
        (cv2.CAP_PROP_WB_TEMPERATURE, 4600.0),
    ]


def test_none_forces_auto_instead_of_inheriting_stale_manual_values():
    assert _applied(_cfg()) == [(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75), (cv2.CAP_PROP_AUTO_WB, 1.0)]


def test_value_the_camera_rejects_fails_loudly():
    with pytest.raises(RuntimeError, match="exposure"):
        _applied(_cfg(exposure=-6, gain=10), clamp={cv2.CAP_PROP_EXPOSURE: -5})


@pytest.mark.parametrize(
    "kw, msg",
    [
        (dict(exposure=-6), "gain"),
        (dict(gain=10), "gain"),
        (dict(exposure=-4, gain=0), "frame"),  # 62.5 ms at 30 fps: measured drop to 17 fps
        (dict(exposure=0, gain=0), r"\[-13, -1\]"),
        (dict(backend=Cv2Backends.MSMF), "DSHOW"),
    ],
)
def test_config_rejects(kw, msg):
    with pytest.raises(ValueError, match=msg):
        _cfg(**kw)


def test_exposure_just_under_one_frame_is_allowed():
    assert _cfg(exposure=-5, gain=0).exposure == -5  # 31.25 ms < 33.3 ms at 30 fps


def test_controls_are_applied_after_size_fps_fourcc(monkeypatch):
    # connect() calls _configure_capture_settings() before _start_read_thread(); the plugin hooks in there.
    order = []
    monkeypatch.setattr(OpenCVCamera, "_configure_capture_settings", lambda self: order.append("size/fps/fourcc"))
    monkeypatch.setattr(OpenCVUVCCamera, "_apply_uvc_controls", lambda self: order.append("uvc"))
    OpenCVUVCCamera(_cfg())._configure_capture_settings()
    assert order == ["size/fps/fourcc", "uvc"]


def test_yaml_type_resolves_through_lerobots_camera_factory():
    yml = (
        "type: opencv_uvc\nindex_or_path: 3\nbackend: DSHOW\nfps: 15\nwidth: 640\nheight: 480\n"
        "fourcc: YUY2\nexposure: -6\ngain: 0\nwhite_balance: 4600\n"
    )
    cfg = draccus.load(CameraConfig, io.StringIO(yml))
    assert isinstance(cfg, OpenCVUVCCameraConfig) and cfg.exposure == -6
    cam = make_cameras_from_configs({"wrist": cfg})["wrist"]  # constructs only; nothing is opened
    assert type(cam) is OpenCVUVCCamera


@pytest.mark.parametrize("config_name", ["record_omx.yaml", "teleoperate_omx.yaml"])
def test_live_omx_configs_use_the_plugin_for_the_wrist(config_name):
    # Switched to `type: opencv_uvc` on 2026-09-13 (field_manual §5-(0) step A); both files must stay in step.
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

    cls = RecordConfig if config_name.startswith("record") else TeleoperateConfig
    cfg = draccus.parse(cls, config_path=str(_REPO / "configs" / config_name), args=[])
    wrist = cfg.robot.cameras["wrist"]
    assert isinstance(wrist, OpenCVUVCCameraConfig)
    assert (wrist.index_or_path, wrist.backend, wrist.width, wrist.height) == (3, Cv2Backends.DSHOW, 640, 480)


def test_plugin_is_discoverable_by_lerobot():
    names = {d.metadata.get("Name") for d in importlib.metadata.distributions()}
    assert "lerobot_camera_uvc" in names, (
        "plugin not installed in this venv -> lerobot will not see `type: opencv_uvc`. "
        "Run: uv pip install -e plugins/lerobot_camera_uvc  (docs/environment.md)"
    )
