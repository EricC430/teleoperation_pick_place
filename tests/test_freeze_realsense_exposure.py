"""Checks for scripts/freeze_realsense_exposure.py that need no camera.

The AUTO-matching run only happens at the lab; here we hold the pure logic against the numbers measured on the
real D455 on 2026-09-13 (D022), plus the two keyboard-safe paths: --help and --dry-run.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "freeze_realsense_exposure.py"
_RECORD = _REPO / "configs" / "record_omx.yaml"

_spec = importlib.util.spec_from_file_location("freeze_realsense_exposure", _SCRIPT)
freeze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(freeze)

# 2026-09-13, D455 at 848x480@15, WB 3500 K: exposure scan at gain 64, and gain scan at exposure 400
EXPOSURE_SCAN = [(20, 11.7), (40, 22.3), (80, 43.9), (160, 68.4), (320, 104.0), (640, 155.5), (1000, 214.3)]
GAIN_SCAN = [(16, 57.0), (32, 73.1), (48, 89.2), (64, 105.9), (80, 129.5), (96, 153.7), (112, 177.8), (128, 199.6)]


def _run(*args):
    return subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, cwd=_REPO, timeout=120)


def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    assert "--dry-run" in r.stdout and "--keep" in r.stdout and "--wb" in r.stdout


def test_dry_run_reads_the_d455_block_and_opens_nothing():
    r = _run("--dry-run")
    assert r.returncode == 0, r.stderr
    assert "262822305610" in r.stdout and "848x480@15" in r.stdout
    assert "NOT opened" in r.stdout


def test_load_strips_manual_values_so_the_run_starts_from_auto():
    cfg = freeze.load_camera_config(_RECORD, "front-left")  # the live YAML now carries values
    assert (cfg.exposure, cfg.gain, cfg.white_balance) == (None, None, None)


def test_load_refuses_the_uvc_wrist_block():
    with pytest.raises(SystemExit, match="not an intelrealsense"):
        freeze.load_camera_config(_RECORD, "wrist")


def test_pick_exposure_takes_the_brightest_step_not_above_auto():
    assert freeze.pick_exposure(EXPOSURE_SCAN, 117.7) == 320  # 104 <= 117.7 < 155.5
    assert freeze.pick_exposure(EXPOSURE_SCAN, 5.0) == 20  # nothing below -> darkest step


def test_interp_gain_reproduces_the_hand_found_value():
    assert freeze.interp_gain(GAIN_SCAN, 114.5) == 70  # found by hand on 2026-09-13 -> 0.0 % off AUTO
    assert freeze.interp_gain(GAIN_SCAN, 10.0) == 16 and freeze.interp_gain(GAIN_SCAN, 250.0) == 128  # clamped


def test_pick_wb_takes_the_closest_colour_balance():
    assert freeze.pick_wb([(3200, 0.97), (3500, 0.93), (3800, 0.88)], 0.93) == 3500


@pytest.mark.parametrize(
    "auto, fixed, ok",
    [
        ((110.9, 0.94, 1.00), (119.4, 0.89, 1.05), True),  # D455 YAML values vs AUTO, 2026-09-13
        ((114.5, 0.94, 1.00), (155.2, 0.97, 1.00), False),  # the failed first attempt (+31.8 % brightness)
        ((129.3, 0.95, 0.92), (131.7, 0.94, 1.07), False),  # wrist-style green cast: B/R fine, G/R +16 %
    ],
)
def test_match_check_covers_brightness_and_both_colour_ratios(auto, fixed, ok):
    assert freeze.match_check(auto, fixed)[0] is ok


def test_drift_flags_a_changed_scene():
    assert freeze.drift(114.5, 126.9) > freeze.TOLERANCE  # the 10.9 % drift seen on 2026-09-13
    assert freeze.drift(117.7, 114.0) <= freeze.TOLERANCE


def test_yaml_snippet_writes_null_for_auto_white_balance():
    assert "white_balance: null" in freeze.yaml_snippet(dict(exposure=400, gain=70, white_balance=None), "front-left")


def test_yaml_snippet_is_pasteable():
    s = freeze.yaml_snippet(dict(exposure=400, gain=70, white_balance=3500), "front-left")
    assert "exposure: 400" in s and "gain: 70" in s and "white_balance: 3500" in s
