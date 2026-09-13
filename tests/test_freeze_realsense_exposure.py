"""Checks for scripts/freeze_realsense_exposure.py that need no camera.

The auto -> freeze -> read cycle only runs at the lab; here we hold the pure logic (config loading, the
brightness self-check, stability across repeats, YAML) and the two keyboard-safe paths: --help and --dry-run.
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


def _run(*args):
    return subprocess.run([sys.executable, str(_SCRIPT), *args], capture_output=True, text=True, cwd=_REPO, timeout=120)


def test_help_exits_zero():
    r = _run("--help")
    assert r.returncode == 0
    assert "--dry-run" in r.stdout and "--keep" in r.stdout and "--repeats" in r.stdout


def test_dry_run_reads_the_d455_block_and_opens_nothing():
    r = _run("--dry-run")
    assert r.returncode == 0, r.stderr
    assert "262822305610" in r.stdout and "848x480@15" in r.stdout
    assert "NOT opened" in r.stdout


def test_load_strips_manual_values_so_connect_starts_in_auto(tmp_path):
    text = _RECORD.read_text(encoding="utf-8").replace(
        "      use_depth: false\n", "      use_depth: false\n      exposure: 100\n      gain: 16\n", 1
    )
    cfg_file = tmp_path / "with_values.yaml"
    cfg_file.write_text(text, encoding="utf-8")
    cfg = freeze.load_camera_config(cfg_file, "front-left")
    assert (cfg.exposure, cfg.gain, cfg.white_balance) == (None, None, None)


def test_load_refuses_the_uvc_wrist_block():
    with pytest.raises(SystemExit, match="not an intelrealsense"):
        freeze.load_camera_config(_RECORD, "wrist")


@pytest.mark.parametrize("auto, frozen, ok", [(120, 118, True), (120, 132, True), (120, 90, False), (0.5, 0.5, True)])
def test_brightness_check(auto, frozen, ok):
    assert freeze.brightness_check(auto, frozen)[0] is ok


def test_summarize_takes_the_median_and_flags_unstable_exposure():
    steady = [dict(exposure=156, gain=64, white_balance=4600)] * 2 + [dict(exposure=160, gain=64, white_balance=4650)]
    values, stable = freeze.summarize(steady)
    assert values == dict(exposure=156, gain=64, white_balance=4600) and stable
    jumpy = [dict(exposure=e, gain=64, white_balance=4600) for e in (100, 156, 300)]
    assert freeze.summarize(jumpy)[1] is False


def test_yaml_snippet_is_pasteable():
    s = freeze.yaml_snippet(dict(exposure=156, gain=64, white_balance=4600), "front-left")
    assert "exposure: 156" in s and "gain: 64" in s and "white_balance: 4600" in s
