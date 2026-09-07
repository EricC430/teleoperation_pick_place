"""Closed-form inverse kinematics for the OMX 5-DOF arm.

The arm has 5 joints and a grasp pose has 6 DOF, so a general pose is NOT reachable
and any honest IK has to say which part it gave up. This one always hits the
**position** exactly when the position is reachable, and reports whatever approach
error is left over — the right priority for grasping, where being at the object
matters more than the exact angle of arrival.

What can and cannot be matched, structurally:

  - position     : exact (3 DOF, consumed by joint1 + the planar 3-link sub-chain)
  - approach roll: exact and free (joint5 spins about the approach axis; for a
                   parallel jaw that is just which way the fingers point)
  - approach dir : only its in-plane component. The plane is pinned by the target
                   position, so the out-of-plane part of a requested approach is
                   unreachable, and comes back as `approach_error_rad`.

Every solution names its branch. There are four: the arm can swing to the target
directly (`front`) or fold back through its own base axis (`back`), each with the
elbow up or down. **`back` solutions are geometrically valid but probably not
usable** — omx_f.urdf's joint limits are placeholders (+/-2pi, D026), so nothing
here rules them out yet. Callers that only want realistic motions should filter on
`branch.startswith("front")` until real limits are known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from omx_ik import kinematics as kin
from omx_ik.workspace import PanWindow
from reach_logger.fk import ee_transform

_ELBOWS = ((True, "elbow_up"), (False, "elbow_down"))


@dataclass(frozen=True)
class IKSolution:
    joints: np.ndarray  # (5,) radians, in fk.ee_transform order
    position_error_m: float
    approach_error_rad: float
    branch: str

    @property
    def approach_error_deg(self) -> float:
        return math.degrees(self.approach_error_rad)

    @property
    def folds_through_base(self) -> bool:
        return self.branch.startswith("back")

    def within(self, position_tol_m: float = 1e-3, approach_tol_deg: float = 10.0) -> bool:
        return self.position_error_m <= position_tol_m and self.approach_error_deg <= approach_tol_deg


def _plane_branches(position: np.ndarray, roll: float) -> list[tuple[float, float, str]]:
    """The (azimuth, radial, name) options for which vertical plane to work in.

    The EE sits `lateral_offset` off the plane, so the plane is rotated slightly from
    the target's own bearing — solved exactly, not iterated. Two options because the
    target can be in front of the base axis or behind it."""
    dx = position[0] - kin.PAN_AXIS_XY[0]
    dy = position[1] - kin.PAN_AXIS_XY[1]
    horizontal = math.hypot(dx, dy)
    lateral = kin.lateral_offset(roll)
    if horizontal < max(kin.AXIS_SINGULARITY_RADIUS_M, abs(lateral)):
        return []  # see kinematics.AXIS_SINGULARITY_RADIUS_M
    bearing = math.atan2(dy, dx)
    swing = math.asin(lateral / horizontal)
    radial = math.sqrt(horizontal**2 - lateral**2)
    return [
        (bearing - swing, radial, "front"),
        (bearing - math.pi + swing, -radial, "back"),
    ]


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    """atan2 form, not acos: acos(1-eps) bottoms out around 1e-8 rad for unit vectors,
    which is coarser than the errors this solver is supposed to resolve."""
    return math.atan2(float(np.linalg.norm(np.cross(a, b))), float(np.dot(a, b)))


def _score(joints: np.ndarray, position: np.ndarray, approach: np.ndarray) -> tuple[float, float]:
    pose = ee_transform(joints)
    position_error = float(np.linalg.norm(pose[:3, 3] - position))
    return position_error, _angle_between(pose[:3, 0], approach)


def _better(a: IKSolution | None, b: IKSolution | None) -> IKSolution | None:
    if a is None:
        return b
    if b is None:
        return a
    key = lambda s: (s.approach_error_rad, s.position_error_m, s.folds_through_base)  # noqa: E731
    return a if key(a) <= key(b) else b


def _solve_in_plane(
    position: np.ndarray,
    approach: np.ndarray,
    pitch: float,
    azimuth: float,
    radial: float,
    roll: float,
    plane: str,
    pan_window: PanWindow | None = None,
) -> IKSolution | None:
    if pan_window is not None and not pan_window.contains(azimuth):
        return None
    best: IKSolution | None = None
    for elbow_up, elbow in _ELBOWS:
        planar = kin.solve_planar(radial, position[2], pitch, roll=roll, elbow_up=elbow_up)
        if planar is None:
            continue
        joints = np.array([azimuth, *planar, roll])
        position_error, approach_error = _score(joints, position, approach)
        best = _better(best, IKSolution(joints, position_error, approach_error, f"{plane}+{elbow}"))
    return best


def solve(
    position: np.ndarray,
    approach: np.ndarray,
    roll: float = 0.0,
    pan_window: PanWindow | None = None,
) -> IKSolution | None:
    """Reach `position` pointing as close to `approach` as the arm's plane allows.

    Returns None when the position cannot be held at the pitch `approach` implies —
    `solve_closest` falls back to the nearest holdable pitch instead."""
    position = np.asarray(position, dtype=float)
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)

    best: IKSolution | None = None
    for azimuth, radial, plane in _plane_branches(position, roll):
        pitch = kin.pitch_for_approach(azimuth, approach)
        best = _better(
            best,
            _solve_in_plane(position, approach, pitch, azimuth, radial, roll, plane, pan_window),
        )
    return best


def roll_for_rotation(rotation: np.ndarray, azimuth: float, pitch: float) -> float:
    """joint5 that lands the gripper's finger axis where `rotation` asks for it.

    EE orientation is Rz(t1) Ry(pitch) Rx(t5), so stripping the first two leaves a
    pure Rx to read the angle off. Always solvable — roll never causes infeasibility,
    it only ever has one right value."""
    c1, s1 = math.cos(azimuth), math.sin(azimuth)
    cb, sb = math.cos(pitch), math.sin(pitch)
    rz_t = np.array([[c1, s1, 0.0], [-s1, c1, 0.0], [0.0, 0.0, 1.0]])
    ry_t = np.array([[cb, 0.0, -sb], [0.0, 1.0, 0.0], [sb, 0.0, cb]])
    residual = ry_t @ rz_t @ rotation
    return math.atan2(residual[2, 1], residual[1, 1])


def solve_pose(
    position: np.ndarray,
    rotation: np.ndarray,
    refine: int = 12,
    pan_window: PanWindow | None = None,
    allow_fold_through_base: bool = True,
) -> IKSolution | None:
    """Full 6-DoF target: `rotation`'s first column is the approach axis, the rest is
    the finger orientation. Roll is derived, not passed in.

    Roll feeds back into the plane choice through the EE's 1.6 mm lateral offset, so
    each branch is re-solved a couple of times; it converges at that scale."""
    position = np.asarray(position, dtype=float)
    rotation = np.asarray(rotation, dtype=float)
    approach = rotation[:, 0]

    # Excluding the fold-through branch has to happen HERE, not by discarding the
    # answer afterwards: this returns a single best solution, so a marginally-better
    # `back` result would otherwise mask a perfectly good `front` one for the same pose.
    plane_indices = (0,) if not allow_fold_through_base else (0, 1)
    best: IKSolution | None = None
    for plane_index in plane_indices:
        roll = 0.0
        for _ in range(max(1, refine)):
            branches = _plane_branches(position, roll)
            if plane_index >= len(branches):
                break
            azimuth, _radial, _plane = branches[plane_index]
            updated = roll_for_rotation(rotation, azimuth, kin.pitch_for_approach(azimuth, approach))
            # compare mod 2pi: roll_for_rotation returns (-pi, pi], so a roll sitting
            # near the wrap point never looks converged under a plain subtraction
            settled = abs(math.remainder(updated - roll, 2.0 * math.pi)) < 1e-14
            roll = updated
            if settled:
                break
        branches = _plane_branches(position, roll)
        if plane_index >= len(branches):
            continue
        azimuth, radial, plane = branches[plane_index]
        pitch = kin.pitch_for_approach(azimuth, approach)
        best = _better(
            best,
            _solve_in_plane(position, approach, pitch, azimuth, radial, roll, plane, pan_window),
        )
    return best


def solve_closest(
    position: np.ndarray,
    approach: np.ndarray,
    roll: float = 0.0,
    pitch_steps: int = 1440,
    pan_window: PanWindow | None = None,
) -> IKSolution | None:
    """Like `solve`, but when the implied pitch cannot be held at this position it
    searches the reachable pitch arc for the one whose approach comes closest.

    Returns None only when the position is unreachable at every pitch, in every plane."""
    position = np.asarray(position, dtype=float)
    approach = np.asarray(approach, dtype=float)
    approach = approach / np.linalg.norm(approach)

    grid = np.linspace(-math.pi, math.pi, pitch_steps, endpoint=False)
    best: IKSolution | None = None
    for azimuth, radial, plane in _plane_branches(position, roll):
        pitch = kin.pitch_for_approach(azimuth, approach)
        exact = _solve_in_plane(
            position, approach, pitch, azimuth, radial, roll, plane, pan_window
        )
        if exact is not None:
            best = _better(best, exact)
            continue
        mask = kin.reachable_pitches(radial, position[2], roll, grid)
        if not mask.any():
            continue
        candidates = grid[mask]
        dots = np.array([kin.approach_direction(azimuth, p) for p in candidates]) @ approach
        nearest = float(candidates[int(np.argmax(dots))])
        best = _better(
            best,
            _solve_in_plane(position, approach, nearest, azimuth, radial, roll, plane, pan_window),
        )
    return best
