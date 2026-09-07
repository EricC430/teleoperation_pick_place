"""Planar reduction of the OMX chain, used by the IK solver.

`reach_logger/fk.py` remains the source of truth for the forward model (it is the
one validated against `urchin`, D026). What lives here is a *derived* planar
representation of the same chain, which is what makes a closed-form IK possible:

    joint1 (z)        picks a vertical plane
    joint2/3/4 (y)    a 3-link planar arm inside that plane
    joint5 (x)        rolls about the end-effector's own pointing axis

Because it is derived rather than shared, it can silently drift from fk.py. That is
what `tests/test_omx_ik.py` guards: every constant and formula here is asserted
against `fk.ee_transform` to 1e-9.

Only one thing breaks the planarity: the end-effector origin carries a -1.6 mm
lateral offset, which joint5's roll swings out of the plane. It is modelled exactly
(`lateral_offset` / `inplane_offset`) rather than ignored, so the IK round-trips to
machine precision at any roll angle.
"""

from __future__ import annotations

import math

import numpy as np

PAN_AXIS_XY = (-0.01125, 0.0)
SHOULDER_HEIGHT = 0.034 + 0.0635  # joint1 origin z + joint2 origin z, in the base frame

L_A = math.hypot(0.0415, 0.11315)  # joint2 -> joint3, a bent link
ALPHA_A = math.atan2(-0.11315, 0.0415)  # its built-in pitch offset
L_B = 0.162  # joint3 -> joint4
L_C = 0.0287 + 0.09193  # joint4 -> joint5 -> EE origin, both along local x
_EE_Y = -0.0016  # the EE origin's lateral offset, swung by joint5

REACH_MIN = abs(L_A - L_B)
REACH_MAX = L_A + L_B

# Targets this close to the pan axis have no well-defined arm plane: joint1's angle
# stops being determined by the target, and the roll/plane coupling that the EE's
# lateral offset creates stops contracting. The region sits inside the robot's own
# base, so nothing is grasped there — the IK reports it unreachable rather than
# returning a pose it cannot actually hold.
AXIS_SINGULARITY_RADIUS_M = 0.010


def lateral_offset(roll: float) -> float:
    """Signed distance of the EE from the arm's plane, as a function of joint5."""
    return _EE_Y * math.cos(roll)


def inplane_offset(roll: float) -> float:
    """The part of the EE origin's lateral offset that joint5 swings back into the
    plane; it acts along the local z of the final link."""
    return _EE_Y * math.sin(roll)


def planar_fk(t2: float, t3: float, t4: float, roll: float = 0.0) -> tuple[float, float]:
    """(radial, height) of the EE within the arm plane, measured from the pan axis."""
    psi_a = t2 + ALPHA_A
    psi_b = t2 + t3
    pitch = t2 + t3 + t4
    dz = inplane_offset(roll)
    radial = L_A * math.cos(psi_a) + L_B * math.cos(psi_b) + L_C * math.cos(pitch) + dz * math.sin(pitch)
    height = (
        SHOULDER_HEIGHT
        - L_A * math.sin(psi_a)
        - L_B * math.sin(psi_b)
        - L_C * math.sin(pitch)
        + dz * math.cos(pitch)
    )
    return radial, height


def approach_direction(azimuth: float, pitch: float) -> np.ndarray:
    """Unit vector the gripper points along. Independent of roll: joint5 spins about
    this axis, it does not move it."""
    return np.array(
        [math.cos(azimuth) * math.cos(pitch), math.sin(azimuth) * math.cos(pitch), -math.sin(pitch)]
    )


def radial_unit(azimuth: float) -> np.ndarray:
    return np.array([math.cos(azimuth), math.sin(azimuth), 0.0])


def lateral_unit(azimuth: float) -> np.ndarray:
    return np.array([-math.sin(azimuth), math.cos(azimuth), 0.0])


def pitch_for_approach(azimuth: float, approach: np.ndarray) -> float:
    """The pitch whose approach direction is closest to `approach`, given the plane is
    already fixed by `azimuth`. Exact: it projects onto the plane and reads off the
    angle, so any residual is purely the out-of-plane component."""
    return math.atan2(-approach[2], float(np.dot(approach, radial_unit(azimuth))))


def solve_planar(
    radial: float, height: float, pitch: float, roll: float = 0.0, elbow_up: bool = True
) -> tuple[float, float, float] | None:
    """Closed-form (t2, t3, t4) placing the EE at (radial, height) with the final link
    at `pitch`. None when the 2-link sub-chain cannot span the required distance."""
    dz = inplane_offset(roll)
    wrist_r = radial - L_C * math.cos(pitch) - dz * math.sin(pitch)
    wrist_h = height + L_C * math.sin(pitch) - dz * math.cos(pitch)
    dr = wrist_r
    dh = wrist_h - SHOULDER_HEIGHT
    dist = math.hypot(dr, dh)
    if not (REACH_MIN <= dist <= REACH_MAX):
        return None
    cos_delta = (dist**2 - L_A**2 - L_B**2) / (2.0 * L_A * L_B)
    delta = math.acos(max(-1.0, min(1.0, cos_delta)))
    if not elbow_up:
        delta = -delta
    phi = math.atan2(-dh, dr)
    psi_a = phi - math.atan2(L_B * math.sin(delta), L_A + L_B * math.cos(delta))
    psi_b = psi_a + delta
    t2 = psi_a - ALPHA_A
    t3 = psi_b - t2
    t4 = pitch - psi_b
    return t2, t3, t4


def reachable_pitches(radial: float, height: float, roll: float, grid: np.ndarray) -> np.ndarray:
    """Boolean mask over `grid`: which final pitches can place the EE at this point."""
    dz = inplane_offset(roll)
    wrist_r = radial - L_C * np.cos(grid) - dz * np.sin(grid)
    wrist_h = height + L_C * np.sin(grid) - dz * np.cos(grid) - SHOULDER_HEIGHT
    dist = np.hypot(wrist_r, wrist_h)
    return (dist >= REACH_MIN) & (dist <= REACH_MAX)
