"""Follower arm COM port, read from a robot config yaml so the scripts need no --port.

The port lives in one place: robot.port of DEFAULT_CONFIG. Change the COM number there (or point --config at
another yaml); --port still overrides both.
"""
from __future__ import annotations

from pathlib import Path

import yaml

DEFAULT_CONFIG = "configs/teleoperate_omx_com5.yaml"
REPO = Path(__file__).resolve().parent.parent


def follower_port(config: str = DEFAULT_CONFIG) -> str:
    path = Path(config)
    if not path.is_absolute():
        path = REPO / path
    if not path.exists():
        raise SystemExit(f"config not found: {path}")
    port = (yaml.safe_load(path.read_text(encoding="utf-8")).get("robot") or {}).get("port")
    if not port:
        raise SystemExit(f"{path} has no robot.port")
    return str(port)
