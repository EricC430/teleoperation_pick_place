"""Copy the YAML a lerobot command runs with to config_records/<robot id>/, the way calibration/<id>.json works.

Why: lerobot stores no camera exposure, port or config in the dataset (checked 2026-09-13: meta/info.json only has
video-encoding info), and configs/*.yaml are edited in place. Without this, "which settings recorded these
episodes" is recoverable only from git, and only if someone committed before recording.

Layout: config_records/<robot id>/<config stem>__<digest>.yaml, where digest = sha256 of the YAML text plus the
CLI overrides. Same id + same content -> nothing new. Same id + different content -> a second file and a warning.
"""

import hashlib
import subprocess
import sys
import time
from pathlib import Path

import yaml

RECORD_DIR = Path("config_records")  # relative to the working directory, like `calibration_dir: ./calibration`
_SKIP_FLAGS = {"-h", "--help"}


def entrypoint_name(argv0: str) -> str | None:
    """'lerobot-record' for .../Scripts/lerobot-record, ...lerobot-record.exe or ...lerobot-record.exe/__main__.py."""
    p = Path(argv0)
    candidates = [p.parent.name, p.name] if p.name == "__main__.py" else [p.name]
    for part in candidates:
        name = part[:-4] if part.lower().endswith(".exe") else part
        if name.startswith("lerobot-"):
            return name
    return None


def effective_ids(cfg: dict, args: list[str]) -> dict[str, str]:
    """robot / teleop ids, with a CLI `--robot.id=...` overriding the YAML just as lerobot's parser does."""
    from lerobot.configs.parser import parse_arg

    ids = {}
    for section in ("robot", "teleop"):
        block = cfg.get(section)
        yml = block.get("id") if isinstance(block, dict) else None
        value = parse_arg(f"{section}.id", args) or yml
        if value:
            ids[section] = str(value)
    return ids


def cli_overrides(args: list[str]) -> list[str]:
    from lerobot.configs.parser import filter_arg

    return filter_arg("config_path", list(args))


def digest_of(text: str, overrides: list[str]) -> str:
    return hashlib.sha256((text + "\0" + "\n".join(overrides)).encode("utf-8")).hexdigest()[:8]


def git_state(config_path: Path) -> str:
    try:
        run = lambda *c: subprocess.run(c, capture_output=True, text=True, timeout=5, check=True).stdout.strip()  # noqa: E731
        sha = run("git", "rev-parse", "--short", "HEAD")
        dirty = run("git", "status", "--porcelain", "--", str(config_path))
        return f"{sha} ({'config file has UNCOMMITTED changes' if dirty else 'config file committed'})"
    except Exception:  # noqa: BLE001
        return "unknown (not a git checkout, or git missing)"


def write_record(config_path, args, entrypoint, record_dir=RECORD_DIR, now=None, git=None):
    """Copy `config_path` into record_dir/<id>/. Returns (path or None, status).

    status: "saved"     first record of this file under this id
            "unchanged" an identical record (same YAML text + same CLI overrides) already exists
            "changed"   this id already holds a DIFFERENT version of this file; a new one was written
            "no-id"     the YAML has neither robot.id nor teleop.id - nothing to key on, nothing written
    """
    config_path = Path(config_path)
    text = config_path.read_text(encoding="utf-8")
    cfg = yaml.safe_load(text) or {}
    ids = effective_ids(cfg, list(args))
    key = ids.get("robot") or ids.get("teleop")
    if not key:
        return None, "no-id"
    overrides = cli_overrides(args)
    digest = digest_of(text, overrides)
    folder = Path(record_dir) / key
    target = folder / f"{config_path.stem}__{digest}.yaml"
    if target.exists():
        return target, "unchanged"
    earlier = list(folder.glob(f"{config_path.stem}__*.yaml")) if folder.exists() else []
    folder.mkdir(parents=True, exist_ok=True)
    header = [
        "# config record - written automatically by lerobot_robot_config_record; do not edit",
        f"# source:        {config_path.as_posix()}",
        f"# saved_at:      {now or time.strftime('%Y-%m-%dT%H:%M:%S')}",
        f"# entrypoint:    {entrypoint}",
        f"# cli overrides: {' '.join(overrides) or '(none)'}",
        f"# ids:           {', '.join(f'{k}={v}' for k, v in ids.items())}",
        f"# git:           {git if git is not None else git_state(config_path)}",
        f"# digest:        {digest}  (sha256[:8] of the YAML below + the CLI overrides)",
        "",
    ]
    target.write_text("\n".join(header) + text, encoding="utf-8")
    return target, ("changed" if earlier else "saved")


def record_from_argv(argv=None, record_dir=RECORD_DIR, main_file=None) -> str | None:
    """Body of the import hook. Returns the message it printed, or None when nothing applies.

    The command is recognised from argv[0] or, as a fallback, from __main__.__file__ - a real run on this laptop
    showed it as `.venv\\Scripts\\lerobot-teleoperate.exe\\__main__.py` (traceback, 2026-09-13), while the exact
    argv[0] form of the .exe launcher was never observed directly.
    """
    real = argv is None
    argv = list(sys.argv if real else argv)
    if main_file is None and real:
        main_file = getattr(sys.modules.get("__main__"), "__file__", None) or ""
    entry = (entrypoint_name(argv[0]) if argv else None) or (entrypoint_name(main_file) if main_file else None)
    args = argv[1:]
    if entry is None or _SKIP_FLAGS & set(args):
        return None
    from lerobot.configs.parser import parse_arg

    config_path = parse_arg("config_path", args)
    if not config_path or not Path(config_path).is_file():
        return None
    path, status = write_record(config_path, args, entry, record_dir)
    if status == "saved":
        msg = f"[config-record] saved {path.as_posix()}"
    elif status == "unchanged":
        msg = f"[config-record] unchanged, already in {path.as_posix()}"
    elif status == "changed":
        msg = (
            f"[config-record] WARNING: id '{path.parent.name}' already has a DIFFERENT {Path(config_path).name}; "
            f"saved this version as {path.as_posix()}. Same id + changed settings: if a scene constant "
            "(camera exposure / position / count) changed, that is a new campaign (D004) - use a new id."
        )
    else:
        return None
    print(msg, file=sys.stderr, flush=True)
    return msg
