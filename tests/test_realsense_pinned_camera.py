"""Unit checks for `type: intelrealsense_pinned` (plugins/lerobot_camera_uvc, D022 §2026-09-13).

No hardware: the colour sensor is faked and we record which options get set. Why the type exists was measured on
the real D455 on 2026-09-13: lerobot's `intelrealsense` leaves a None option unchanged, and the D455 keeps its
auto-white-balance state across processes, so `white_balance: null` silently inherited a green manual 3500 K.
"""

import io
from pathlib import Path

import draccus
import pyrealsense2 as rs
import pytest

from lerobot.cameras.configs import CameraConfig
from lerobot.cameras.realsense import RealSenseCamera
from lerobot.cameras.utils import make_cameras_from_configs
from lerobot_camera_uvc import RealSensePinnedCamera, RealSensePinnedCameraConfig

_REPO = Path(__file__).resolve().parents[1]
O = rs.option


class FakeSensor:
    def __init__(self):
        self.calls = []

    def supports(self, option):
        return True

    def set_option(self, option, value):
        self.calls.append((option, float(value)))


def _cfg(**kw):
    base = dict(serial_number_or_name="262822305610", width=848, height=480, fps=15)
    return RealSensePinnedCameraConfig(**{**base, **kw})


def _applied(cfg, monkeypatch):
    cam = RealSensePinnedCamera(cfg)
    sensor = FakeSensor()
    monkeypatch.setattr(cam, "_get_color_sensor", lambda: sensor)
    cam._configure_sensor_options()
    return sensor.calls


def test_fixed_exposure_with_white_balance_forced_back_to_auto(monkeypatch):
    calls = _applied(_cfg(exposure=400, gain=70), monkeypatch)
    assert (O.exposure, 400.0) in calls and (O.gain, 70.0) in calls
    assert calls[-1] == (O.enable_auto_white_balance, 1.0)  # set explicitly, not left as found
    assert not any(option == O.white_balance for option, _ in calls)


def test_all_none_forces_both_auto_modes(monkeypatch):
    assert _applied(_cfg(), monkeypatch) == [(O.enable_auto_exposure, 1.0), (O.enable_auto_white_balance, 1.0)]


def test_all_set_is_fully_manual(monkeypatch):
    calls = _applied(_cfg(exposure=400, gain=70, white_balance=3500), monkeypatch)
    assert (O.white_balance, 3500.0) in calls
    assert (O.enable_auto_exposure, 1.0) not in calls and (O.enable_auto_white_balance, 1.0) not in calls


@pytest.mark.parametrize("kw", [dict(exposure=400), dict(gain=70)])
def test_exposure_and_gain_go_together(kw):
    with pytest.raises(ValueError, match="together"):
        _cfg(**kw)


def test_yaml_type_resolves_through_lerobots_camera_factory():
    yml = (
        "type: intelrealsense_pinned\nserial_number_or_name: '262822305610'\nwidth: 848\nheight: 480\nfps: 15\n"
        "exposure: 400\ngain: 70\nwhite_balance: null\n"
    )
    cfg = draccus.load(CameraConfig, io.StringIO(yml))
    assert isinstance(cfg, RealSensePinnedCameraConfig) and cfg.white_balance is None
    cam = make_cameras_from_configs({"front-left": cfg})["front-left"]  # constructs only; nothing is opened
    assert type(cam) is RealSensePinnedCamera and isinstance(cam, RealSenseCamera)


@pytest.mark.parametrize("config_name", ["record_omx.yaml", "teleoperate_omx.yaml"])
def test_live_omx_configs_pin_the_d455(config_name):
    from lerobot.scripts.lerobot_record import RecordConfig
    from lerobot.scripts.lerobot_teleoperate import TeleoperateConfig

    cls = RecordConfig if config_name.startswith("record") else TeleoperateConfig
    cam = draccus.parse(cls, config_path=str(_REPO / "configs" / config_name), args=[]).robot.cameras["front-left"]
    assert isinstance(cam, RealSensePinnedCameraConfig)
    assert cam.white_balance is None  # AUTO, set explicitly on connect
    assert (cam.exposure is None) == (cam.gain is None)
