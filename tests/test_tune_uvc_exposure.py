"""Checks for scripts/tune_uvc_exposure.py that need no camera and no display.

The live window only runs at the lab; here we hold the pure logic (keys, validation, stats, YAML) and the two
things that must be safe from a keyboard: --help and --dry-run.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "tune_uvc_exposure.py"
_RECORD = _REPO / "configs" / "record_omx.yaml"

_spec = importlib.util.spec_from_file_location("tune_uvc_exposure", _SCRIPT)
tune = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tune)
C = tune.Controls


def _run(*args):
    return subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, cwd=_REPO, timeout=120)


def test_help_exits_zero_and_lists_keys():
    r = _run("--help")
    assert r.returncode == 0
    assert "--dry-run" in r.stdout and "--keep" in r.stdout and "exposure" in r.stdout


def test_dry_run_reads_the_record_config_and_opens_nothing():
    r = _run("--dry-run")
    assert r.returncode == 0, r.stderr
    assert "index=3" in r.stdout and "DSHOW" in r.stdout
    assert "NOT opened" in r.stdout


def test_a_toggles_manual_exposure_with_a_start_gain():
    assert tune.step(C(), "a") == C(exposure=-6, gain=0)
    assert tune.step(C(exposure=-7, gain=10), "a") == C()


@pytest.mark.parametrize(
    "start, key, expected",
    [
        (C(exposure=-6, gain=0), "e", C(exposure=-7, gain=0)),
        (C(exposure=-6, gain=0), "E", C(exposure=-5, gain=0)),
        (C(exposure=-6, gain=0), "g", C(exposure=-6, gain=0)),  # gain never below 0
        (C(exposure=-6, gain=5), "G", C(exposure=-6, gain=10)),
        (C(white_balance=4600), "w", C(white_balance=4400)),
        (C(), "b", C(white_balance=4600)),
    ],
)
def test_keys(start, key, expected):
    assert tune.step(start, key) == expected


@pytest.mark.parametrize("start, key", [(C(), "e"), (C(), "g"), (C(), "w"), (C(), "x"), (C(), "")])
def test_keys_that_do_nothing_in_the_current_mode(start, key):
    assert tune.step(start, key) is None


def test_validation_is_the_plugins():
    cfg = tune.load_camera_config(_RECORD, "wrist")  # fps 15 -> one frame = 66.7 ms
    ok, err = tune.with_controls(cfg, C(exposure=-6, gain=0))
    assert err is None and ok.exposure == -6
    bad, err = tune.with_controls(cfg, C(exposure=-3, gain=0))  # 125 ms > one frame
    assert bad is None and "frame" in err
    bad, err = tune.with_controls(cfg, C(exposure=-6))  # exposure without gain
    assert bad is None and "gain" in err


def test_load_lifts_plain_opencv_block_to_uvc():
    from lerobot_camera_uvc import OpenCVUVCCameraConfig

    cfg = tune.load_camera_config(_RECORD, "wrist")
    assert isinstance(cfg, OpenCVUVCCameraConfig)
    assert (cfg.index_or_path, cfg.backend.name, cfg.width, cfg.height) == (3, "DSHOW", 640, 480)


def test_load_refuses_a_realsense_block():
    with pytest.raises(SystemExit, match="not an OpenCV"):
        tune.load_camera_config(_RECORD, "front-left")


def test_frame_stats():
    white = np.full((480, 640, 3), 255, np.uint8)
    black = np.zeros((480, 640, 3), np.uint8)
    assert tune.frame_stats(white)["clipped_pct"] == 100.0
    assert tune.frame_stats(black)["crushed_pct"] == 100.0
    half = black.copy()
    half[120:360, 160:480] = 255  # exactly the centre quarter
    st = tune.frame_stats(half)
    assert st["centre_clipped_pct"] == 100.0 and st["clipped_pct"] == 25.0


def test_frame_stats_colour_balance():
    f = np.zeros((10, 10, 3), np.uint8)
    f[..., 0], f[..., 2] = 100, 50  # BGR: blue 100, red 50
    assert tune.frame_stats(f)["b_over_r"] == pytest.approx(2.0)


@pytest.mark.parametrize("m_fix, br_fix, ok", [(115.2, 0.97, True), (90.0, 0.99, False), (113.0, 0.75, False)])
def test_freeze_check(m_fix, br_fix, ok):
    # AUTO reference and the pinned result from the 2026-09-13 probe: 113.2 / 0.99 -> 115.2 / 0.97
    assert tune.freeze_check(113.2, 0.99, m_fix, br_fix)[0] is ok


def test_help_lists_the_freeze_key():
    assert "f  FREEZE" in tune.KEY_HELP


def test_yaml_snippet_is_pasteable():
    s = tune.yaml_snippet(C(exposure=-6, gain=10), "wrist")
    assert "type: opencv_uvc" in s and "exposure: -6" in s and "gain: 10" in s and "white_balance: null" in s
