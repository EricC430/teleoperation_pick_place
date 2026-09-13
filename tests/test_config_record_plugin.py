"""Checks for the config-record plugin (plugins/lerobot_robot_config_record).

No hardware and no robot command: the hook body is called directly, and the plugin-discovery path is exercised by
calling lerobot's own register_third_party_plugins() in a subprocess with a faked lerobot-record argv.
"""

import importlib.metadata
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from lerobot_robot_config_record.recorder import entrypoint_name, record_from_argv, write_record

_REPO = Path(__file__).resolve().parents[1]
_TELEOP = _REPO / "configs" / "teleoperate_omx.yaml"
_FAKE_RECORD = r"D:\x\.venv\Scripts\lerobot-record"

_CFG = "teleop:\n  type: omx_leader\n  id: L1\nrobot:\n  type: omx_follower\n  id: {rid}\n  port: COM8\n"


def _cfg(tmp_path, rid="2026-09-13_omx_follower", name="teleop.yaml", extra=""):
    p = tmp_path / name
    p.write_text(_CFG.format(rid=rid) + extra, encoding="utf-8")
    return p


def _save(cfg, rec, *overrides):
    return write_record(cfg, ["--config_path", str(cfg), *overrides], "lerobot-teleoperate", rec, now="T", git="G")


@pytest.mark.parametrize(
    "argv0, name",
    [
        (r"D:\x\.venv\Scripts\lerobot-record", "lerobot-record"),
        (r"D:\x\.venv\Scripts\lerobot-record.exe", "lerobot-record"),
        (r"D:\x\.venv\Scripts\lerobot-teleoperate.exe\__main__.py", "lerobot-teleoperate"),
        ("/usr/bin/pytest", None),
        (r"D:\x\.venv\Scripts\python.exe", None),
    ],
)
def test_entrypoint_name(argv0, name):
    assert entrypoint_name(argv0) == name


def test_first_run_of_an_id_saves_a_copy_like_calibration(tmp_path):
    cfg = _cfg(tmp_path)
    path, status = _save(cfg, tmp_path / "rec")
    assert status == "saved"
    assert path.parent == tmp_path / "rec" / "2026-09-13_omx_follower" and path.name.startswith("teleop__")
    body = path.read_text(encoding="utf-8")
    assert body.endswith(cfg.read_text(encoding="utf-8"))  # the YAML itself, untouched
    assert "# git:           G" in body and "# entrypoint:    lerobot-teleoperate" in body


def test_same_id_same_content_writes_nothing_new(tmp_path):
    cfg = _cfg(tmp_path)
    first, _ = _save(cfg, tmp_path / "rec")
    again, status = _save(cfg, tmp_path / "rec")
    assert status == "unchanged" and again == first
    assert len(list((tmp_path / "rec" / "2026-09-13_omx_follower").iterdir())) == 1


def test_same_id_changed_content_keeps_both_and_flags_it(tmp_path):
    cfg = _cfg(tmp_path)
    _save(cfg, tmp_path / "rec")
    cfg.write_text(cfg.read_text(encoding="utf-8") + "fps: 15\n", encoding="utf-8")
    _, status = _save(cfg, tmp_path / "rec")
    assert status == "changed"
    assert len(list((tmp_path / "rec" / "2026-09-13_omx_follower").iterdir())) == 2


def test_new_id_gets_its_own_folder(tmp_path):
    _save(_cfg(tmp_path), tmp_path / "rec")
    _, status = _save(_cfg(tmp_path, rid="2026-09-14_omx_follower"), tmp_path / "rec")
    assert status == "saved"
    assert sorted(p.name for p in (tmp_path / "rec").iterdir()) == ["2026-09-13_omx_follower", "2026-09-14_omx_follower"]


def test_cli_overrides_count(tmp_path):
    cfg = _cfg(tmp_path)
    path, status = _save(cfg, tmp_path / "rec", "--robot.id=OVERRIDE")  # id from the CLI wins, like lerobot
    assert status == "saved" and path.parent.name == "OVERRIDE"
    _, status = _save(cfg, tmp_path / "rec", "--robot.id=OVERRIDE", "--robot.port=COM6")  # other override -> new
    assert status == "changed"


def test_config_without_ids_is_skipped(tmp_path):
    p = tmp_path / "train.yaml"
    p.write_text("policy:\n  type: act\n", encoding="utf-8")
    assert _save(p, tmp_path / "rec") == (None, "no-id")
    assert not (tmp_path / "rec").exists()


def test_hook_ignores_non_lerobot_commands_help_and_missing_files(tmp_path):
    cfg = _cfg(tmp_path)
    rec = tmp_path / "rec"
    assert record_from_argv(["/usr/bin/pytest", "--config_path", str(cfg)], rec) is None
    assert record_from_argv([_FAKE_RECORD, "--config_path", str(cfg), "--help"], rec) is None
    assert record_from_argv([_FAKE_RECORD, "--config_path", str(tmp_path / "nope.yaml")], rec) is None
    assert record_from_argv([_FAKE_RECORD], rec) is None
    assert not rec.exists()
    assert "saved" in record_from_argv([_FAKE_RECORD, f"--config_path={cfg}"], rec)


def test_command_recognised_from_main_file_when_argv0_is_not(tmp_path):
    # __main__.__file__ exactly as it appeared in a real lerobot-teleoperate traceback on this laptop (2026-09-13)
    main_file = r"D:\teleoperation_pick_place\.venv\Scripts\lerobot-teleoperate.exe\__main__.py"
    cfg = _cfg(tmp_path)
    msg = record_from_argv(["python", "--config_path", str(cfg)], tmp_path / "rec", main_file=main_file)
    assert msg is not None and "saved" in msg
    assert record_from_argv(["python", "--config_path", str(cfg)], tmp_path / "rec2", main_file="C:/x/pytest") is None


def test_real_teleop_config_is_keyed_by_its_robot_id(tmp_path):
    rid = yaml.safe_load(_TELEOP.read_text(encoding="utf-8"))["robot"]["id"]
    path, status = _save(_TELEOP, tmp_path / "rec")
    assert status == "saved" and path.parent.name == rid


def test_lerobot_plugin_discovery_triggers_the_hook(tmp_path):
    cfg = _cfg(tmp_path)
    code = (
        "import sys; "
        f"sys.argv = [{_FAKE_RECORD!r}, '--config_path', {str(cfg)!r}]; "
        "from lerobot.utils.import_utils import register_third_party_plugins; "
        "register_third_party_plugins()"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    assert "[config-record] saved" in r.stderr
    assert (tmp_path / "config_records" / "2026-09-13_omx_follower").is_dir()


def test_plugin_is_installed():
    names = {d.metadata.get("Name") for d in importlib.metadata.distributions()}
    assert "lerobot_robot_config_record" in names, (
        "not installed -> lerobot will not run the hook. Run: uv pip install -e plugins/lerobot_robot_config_record"
    )
