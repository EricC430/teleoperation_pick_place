"""Acceptance checks for the CLI entry point (spec 10).

The interactive loop can only run at the lab; here we hold the line on the two
things that must be safe from a keyboard: --help and --dry-run.
"""

import subprocess
import sys
from pathlib import Path

import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO / "scripts" / "reach_logger.py"


def _run(*args):
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *args],
        capture_output=True,
        text=True,
        cwd=_REPO,
        timeout=60,
    )


def test_help_exits_zero_and_lists_the_keys():
    r = _run("--help")
    assert r.returncode == 0
    assert "--dry-run" in r.stdout
    assert "--fk-fallback" in r.stdout


def test_dry_run_prints_plan_and_touches_nothing(tmp_path):
    out = tmp_path / "reach_log_x.csv"
    r = _run("--dry-run", "--out", str(out), "--mode", "follower-only")
    assert r.returncode == 0
    assert "NOT contacted" in r.stdout
    assert "follower-only" in r.stdout
    assert str(out) in r.stdout
    assert not out.exists()  # dry run writes nothing


def test_bad_mode_is_rejected():
    r = _run("--mode", "wiggle", "--dry-run")
    assert r.returncode != 0
    assert "wiggle" in r.stderr or "invalid choice" in r.stderr


@pytest.mark.parametrize("flag", ["--config", "--urdf", "--out"])
def test_dry_run_echoes_each_path_flag(flag):
    r = _run("--dry-run", flag, "SENTINEL_VALUE")
    assert r.returncode == 0
    assert "SENTINEL_VALUE" in r.stdout
