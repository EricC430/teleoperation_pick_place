"""Forward kinematics for the omx_f arm, hand-coded from assets/omx_f/omx_f.urdf.

placo/cmeel do not build on Windows (D026), so instead of lerobot's RobotKinematics
we walk the URDF chain directly. Every joint in omx_f.urdf has rpy="0 0 0", so each
link is: translate by the joint origin, then rotate by the joint angle about its axis.

Joint order (URDF joint1..joint5 == motors shoulder_pan..wrist_roll):

    #  origin xyz (m)              axis
    1  (-0.01125, 0, 0.034)       z
    2  (0, 0, 0.0635)             y
    3  (0.0415, 0, 0.11315)       y
    4  (0.162, 0, 0)              y
    5  (0.0287, 0, 0)             x
    ee (0.09193, -0.0016, 0)      fixed
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

# (origin_xyz, axis) per joint, in URDF order. rpy is 0 for all, so it is omitted.
_CHAIN: tuple[tuple[tuple[float, float, float], str], ...] = (
    ((-0.01125, 0.0, 0.034), "z"),
    ((0.0, 0.0, 0.0635), "y"),
    ((0.0415, 0.0, 0.11315), "y"),
    ((0.162, 0.0, 0.0), "y"),
    ((0.0287, 0.0, 0.0), "x"),
)
_EE_ORIGIN = (0.09193, -0.0016, 0.0)

# XY intersection of the joint1 (shoulder_pan) rotation axis with the base frame.
_PAN_AXIS_XY = (-0.01125, 0.0)

N_JOINTS = len(_CHAIN)


def _translation(xyz: Sequence[float]) -> np.ndarray:
    t = np.eye(4)
    t[:3, 3] = xyz
    return t


def _rotation(axis: str, angle_rad: float) -> np.ndarray:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    r = np.eye(4)
    if axis == "x":
        r[1, 1], r[1, 2], r[2, 1], r[2, 2] = c, -s, s, c
    elif axis == "y":
        r[0, 0], r[0, 2], r[2, 0], r[2, 2] = c, s, -s, c
    elif axis == "z":
        r[0, 0], r[0, 1], r[1, 0], r[1, 1] = c, -s, s, c
    else:  # pragma: no cover - guarded by _CHAIN
        raise ValueError(f"unknown axis {axis!r}")
    return r


def ee_transform(joint_rad: Sequence[float]) -> np.ndarray:
    """4x4 homogeneous transform of end_effector_link in the arm base frame."""
    if len(joint_rad) != N_JOINTS:
        raise ValueError(f"expected {N_JOINTS} joint angles, got {len(joint_rad)}")
    t = np.eye(4)
    for (origin, axis), q in zip(_CHAIN, joint_rad):
        t = t @ _translation(origin) @ _rotation(axis, q)
    return t @ _translation(_EE_ORIGIN)


def ee_position_m(joint_rad: Sequence[float]) -> tuple[float, float, float]:
    x, y, z = ee_transform(joint_rad)[:3, 3]
    return float(x), float(y), float(z)


def reach_cm(joint_rad: Sequence[float]) -> float:
    """Planar distance (cm) from the shoulder_pan axis to the end effector."""
    x, y, _ = ee_position_m(joint_rad)
    dx = x - _PAN_AXIS_XY[0]
    dy = y - _PAN_AXIS_XY[1]
    return math.hypot(dx, dy) * 100.0


def azimuth_base_deg(joint_rad: Sequence[float]) -> float:
    """Azimuth (deg, wrapped to (-180, 180]) of the end effector about the pan axis."""
    x, y, _ = ee_position_m(joint_rad)
    return math.degrees(math.atan2(y - _PAN_AXIS_XY[1], x - _PAN_AXIS_XY[0]))
