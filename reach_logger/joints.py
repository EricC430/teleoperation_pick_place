"""Raw Dynamixel readings -> FK joint-angle vector, in radians.

The reach logger reads ``Present_Position`` with ``normalize=False`` (see
docs/specs/S1_reach_logger.md 3): raw ticks, 4096 per revolution, and in
EXTENDED_POSITION mode the count can be negative or exceed one turn. ``.pos``
must not be used because it clamps.

Per-joint ``offset_rad`` and ``sign`` come from the on-site 5-pose validation
(spec 5): ``offset_rad`` is the raw angle read when the joint is at the URDF
zero, ``sign`` is +1 or -1 depending on how the servo is mounted.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

_TICKS_PER_REV = 4096

# The five arm joints, in the order fk.ee_transform expects. The gripper motor is
# read for the record but never enters FK (link5 is the end-effector's parent).
FK_MOTOR_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)


def ticks_to_rad(ticks: float) -> float:
    """Raw encoder ticks -> angle in radians, wrapped to (-pi, pi]."""
    rad = ticks * (2.0 * math.pi / _TICKS_PER_REV)
    wrapped = math.remainder(rad, 2.0 * math.pi)
    # math.remainder maps exact -pi to -pi; normalise to (-pi, pi].
    if wrapped <= -math.pi:
        wrapped += 2.0 * math.pi
    return wrapped


@dataclass(frozen=True)
class JointCalibration:
    offset_rad: list[float] = field(default_factory=lambda: [0.0] * len(FK_MOTOR_ORDER))
    sign: list[int] = field(default_factory=lambda: [1] * len(FK_MOTOR_ORDER))

    def __post_init__(self) -> None:
        n = len(FK_MOTOR_ORDER)
        if len(self.offset_rad) != n or len(self.sign) != n:
            raise ValueError(f"calibration must have {n} entries per field")


def identity_calibration() -> JointCalibration:
    """No offset, no sign flip - FK runs but azimuth is only relative."""
    return JointCalibration()


def joint_vector(readings: dict[str, float], calib: JointCalibration) -> list[float]:
    """Map a ``{motor: ticks}`` dict to the 5-element radian vector fk expects."""
    return [
        sign * (ticks_to_rad(readings[motor]) - offset)
        for motor, offset, sign in zip(FK_MOTOR_ORDER, calib.offset_rad, calib.sign)
    ]
