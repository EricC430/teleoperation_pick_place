"""Hardware boundary for the reach logger (docs/specs/S1_reach_logger.md 4).

Two run modes:
  - "teleop":        drive the follower from the real leader; step() servos one cycle
  - "follower-only": torque off the follower, the operator hand-poses it; step() is a no-op

read_ticks() returns raw Present_Position for all six motors (normalize=False, per
spec 3). The real OMX classes are imported lazily so this module - and every
test - loads without a serial port present.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

MOTORS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
MODES = ("teleop", "follower-only")


class RobotIOError(RuntimeError):
    """Raised for a hardware problem we can explain (e.g. a motor missing)."""


class DryRunError(RuntimeError):
    """Raised if a --dry-run session tries to touch hardware."""


def require_motors(ticks: dict[str, float]) -> dict[str, float]:
    missing = [m for m in MOTORS if m not in ticks]
    if missing:
        raise RobotIOError(
            "arm did not report these motors: "
            + ", ".join(missing)
            + " — is it powered and on the right COM port?"
        )
    return ticks


@runtime_checkable
class ReachRobot(Protocol):
    def __enter__(self) -> "ReachRobot": ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...
    def read_ticks(self) -> dict[str, float]: ...
    def step(self) -> None: ...


class FakeReachRobot:
    """Replays a scripted list of tick readings; advanced by step()."""

    def __init__(self, tick_frames: Sequence[dict[str, float]]) -> None:
        if not tick_frames:
            raise ValueError("need at least one tick frame")
        self._frames = list(tick_frames)
        self._i = 0
        self.steps = 0
        self.connected = False

    def __enter__(self) -> "FakeReachRobot":
        self.connected = True
        return self

    def __exit__(self, *exc: object) -> None:
        self.connected = False

    def read_ticks(self) -> dict[str, float]:
        return require_motors(dict(self._frames[self._i]))

    def step(self) -> None:
        self.steps += 1
        if self._i < len(self._frames) - 1:
            self._i += 1


class DryRunRobot:
    """Stands in for a robot during --dry-run; any hardware call raises."""

    def __enter__(self) -> "DryRunRobot":
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read_ticks(self) -> dict[str, float]:
        raise DryRunError("read_ticks() called during --dry-run")

    def step(self) -> None:
        raise DryRunError("step() called during --dry-run")


class OmxReachRobot:
    """The real path. Thin wrapper over lerobot's OmxFollower (+ OmxLeader)."""

    def __init__(self, config_path: str, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.config_path = config_path
        self.mode = mode
        self._follower = None
        self._leader = None

    def __enter__(self) -> "OmxReachRobot":
        follower, leader = _load_omx(self.config_path, self.mode)
        follower.connect()
        if leader is not None:
            leader.connect()
        elif self.mode == "follower-only":
            follower.bus.disable_torque()
        self._follower, self._leader = follower, leader
        return self

    def __exit__(self, *exc: object) -> None:
        for dev in (self._leader, self._follower):
            if dev is not None:
                dev.disconnect()
        self._follower = self._leader = None

    def read_ticks(self) -> dict[str, float]:
        if self._follower is None:
            raise RobotIOError("robot is not connected")
        raw = self._follower.bus.sync_read("Present_Position", normalize=False)
        return require_motors(dict(raw))

    def step(self) -> None:
        if self.mode == "follower-only":
            return
        if self._follower is None or self._leader is None:
            raise RobotIOError("teleop step needs both leader and follower connected")
        self._follower.send_action(self._leader.get_action())


def build_robot(*, config_path: str, mode: str, dry_run: bool) -> ReachRobot:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if dry_run:
        return DryRunRobot()
    return OmxReachRobot(config_path=config_path, mode=mode)


def _load_omx(config_path: str, mode: str):  # pragma: no cover - needs lerobot + a port
    """Build (follower, leader-or-None) from the teleoperate_omx.yaml blocks."""
    import yaml

    from lerobot.robots.omx_follower import OmxFollower, OmxFollowerConfig

    cfg = yaml.safe_load(open(config_path, encoding="utf-8"))
    rc = cfg["robot"]
    follower = OmxFollower(
        OmxFollowerConfig(port=rc["port"], id=rc["id"], calibration_dir=rc["calibration_dir"])
    )
    leader = None
    if mode == "teleop":
        from lerobot.teleoperators.omx_leader import OmxLeader, OmxLeaderConfig

        tc = cfg["teleop"]
        leader = OmxLeader(
            OmxLeaderConfig(port=tc["port"], id=tc["id"], calibration_dir=tc["calibration_dir"])
        )
    return follower, leader


def frames_from_single(ticks: dict[str, float]) -> Iterable[dict[str, float]]:
    """Convenience for callers that want a constant FakeReachRobot."""
    return [ticks]
