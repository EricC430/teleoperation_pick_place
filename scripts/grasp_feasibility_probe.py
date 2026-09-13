"""Exploratory probe (2026-09-05, no D-number yet): how much of a general 6-DoF
grasp pose can the OMX 5-DOF arm actually reach?

Context: before spending time standing up GSNet/EconomicGrasp on GraspNet-1Billion,
check whether the arm's kinematic structure is even a plausible consumer of that
kind of output.

THE STRUCTURE (from reach_logger/fk.py, D026):

    joint1 z (base yaw)  -> sets the vertical PLANE the rest of the arm lives in
    joint2/3/4 y         -> a 3-link planar arm inside that plane
    joint5 x             -> roll about the end-effector's own pointing axis

Two consequences, and the second one is the whole answer:

1. ROLL IS FREE. joint5 rolls about the EE's local x, which is also the direction
   the EE offset points. For a parallel-jaw gripper that axis is the approach
   axis, and rolling about it is exactly "which way the fingers are oriented".
   So the arm never fails on that DOF.

2. Therefore the orientation error of the best achievable pose equals the ANGLE
   BETWEEN APPROACH DIRECTIONS, nothing more. (The rotations sharing one approach
   direction form a fiber of R -> R e_x; the geodesic distance from any target
   rotation to that fiber is exactly the angle between the two approach vectors.)

So the only question is: which approach directions can the arm produce at a given
position? Approach = Rz(t1) Ry(t2+t3+t4) e_x = (cos t1 cos b, sin t1 cos b, -sin b)
with b = t2+t3+t4. t1 is pinned by the target position (it picks the plane), so the
achievable approach directions form an ARC OF A GREAT CIRCLE — the arc, not the
whole circle, because b must also satisfy the position constraint.

Ceiling for intuition: a uniformly random direction lies within `tol` of a FULL
great circle with probability exactly sin(tol) — 17.4% at 10 deg, 50% at 30 deg.
The arc restriction can only pull the real number below that.

METHOD: no numerical IK. For each target the achievable pitch set B(p) is computed
by closed-form 2-link reachability (walk b over a fine grid; for each b place the
wrist and test |L_A - L_B| <= D <= L_A + L_B), then the answer is the minimum angle
between the target approach and the arc. An earlier version of this script used a
multi-start Gauss-Newton IK instead and reported 0-2%, then 4-7%, both of which were
optimizer artifacts, not geometry. Self-checks below tie the planar model back to
reach_logger.fk.ee_transform so the numbers are anchored to the verified FK.

Joint limits are still ignored — omx_f.urdf's are placeholders (+/-2pi) per D026 —
so every number here is an UPPER bound on what the hardware can do.
"""

from __future__ import annotations

import argparse
import math

import numpy as np

from reach_logger.fk import ee_transform

# Planar model of the arm, reduced from the fk.py chain. Inside the plane chosen by
# t1, each segment's direction is (cos psi, -sin psi) for a cumulative pitch psi.
_PAN_AXIS_XY = (-0.01125, 0.0)
_SHOULDER_HEIGHT = 0.034 + 0.0635  # joint1 origin z + joint2 origin z, above the base
_L_A = math.hypot(0.0415, 0.11315)  # joint2 -> joint3, a bent link
_ALPHA_A = math.atan2(-0.11315, 0.0415)  # its built-in angle offset, in the same convention
_L_B = 0.162  # joint3 -> joint4
_L_C = 0.0287 + 0.09193  # joint4 -> joint5 -> end effector, both along local x
_REACH_MIN = abs(_L_A - _L_B)
_REACH_MAX = _L_A + _L_B


def planar_fk(t2: float, t3: float, t4: float) -> tuple[float, float]:
    """(radial, height) of the EE relative to the pan axis and the arm's base plane.
    Ignores the 1.6 mm lateral offset in the EE origin (the arm's only out-of-plane
    term); see test_matches_fk()."""
    psi_a = t2 + _ALPHA_A
    psi_b = t2 + t3
    psi_c = t2 + t3 + t4
    radial = _L_A * math.cos(psi_a) + _L_B * math.cos(psi_b) + _L_C * math.cos(psi_c)
    height = _SHOULDER_HEIGHT - (
        _L_A * math.sin(psi_a) + _L_B * math.sin(psi_b) + _L_C * math.sin(psi_c)
    )
    return radial, height


def solve_planar(radial: float, height: float, pitch: float, elbow_up: bool = True) -> tuple[float, float, float] | None:
    """Closed-form (t2, t3, t4) reaching (radial, height) with final pitch `pitch`,
    or None if the 2-link sub-chain cannot span the required distance."""
    wrist_r = radial - _L_C * math.cos(pitch)
    wrist_h = height + _L_C * math.sin(pitch)
    dr = wrist_r
    dh = wrist_h - _SHOULDER_HEIGHT
    dist = math.hypot(dr, dh)
    if not (_REACH_MIN <= dist <= _REACH_MAX):
        return None
    cos_delta = (dist**2 - _L_A**2 - _L_B**2) / (2.0 * _L_A * _L_B)
    delta = math.acos(max(-1.0, min(1.0, cos_delta)))
    if not elbow_up:
        delta = -delta
    phi = math.atan2(-dh, dr)
    psi_a = phi - math.atan2(_L_B * math.sin(delta), _L_A + _L_B * math.cos(delta))
    psi_b = psi_a + delta
    t2 = psi_a - _ALPHA_A
    t3 = psi_b - t2
    t4 = pitch - psi_b
    return t2, t3, t4


def achievable_pitch_mask(radial: float, height: float, pitch_grid: np.ndarray) -> np.ndarray:
    """Boolean mask over pitch_grid: which final pitches can reach (radial, height)."""
    wrist_r = radial - _L_C * np.cos(pitch_grid)
    wrist_h = height + _L_C * np.sin(pitch_grid) - _SHOULDER_HEIGHT
    dist = np.hypot(wrist_r, wrist_h)
    return (dist >= _REACH_MIN) & (dist <= _REACH_MAX)


def approach_directions(azimuth: float, pitch_grid: np.ndarray) -> np.ndarray:
    """(N, 3) unit approach vectors the arm produces at this azimuth, per pitch."""
    return np.column_stack(
        [np.cos(azimuth) * np.cos(pitch_grid), np.sin(azimuth) * np.cos(pitch_grid), -np.sin(pitch_grid)]
    )


def min_approach_error_rad(
    position: np.ndarray, target_approach: np.ndarray, pitch_grid: np.ndarray
) -> tuple[float, float]:
    """Smallest angle between `target_approach` and any approach the arm can produce
    while its EE sits at `position`. Also returns the fraction of the pitch circle
    that is achievable there, as a diagnostic. Returns (inf, 0.0) if the position
    itself is out of reach."""
    azimuth = math.atan2(position[1] - _PAN_AXIS_XY[1], position[0] - _PAN_AXIS_XY[0])
    radial = math.hypot(position[0] - _PAN_AXIS_XY[0], position[1] - _PAN_AXIS_XY[1])
    mask = achievable_pitch_mask(radial, position[2], pitch_grid)
    if not mask.any():
        return math.inf, 0.0
    dots = approach_directions(azimuth, pitch_grid[mask]) @ target_approach
    best = float(np.arccos(np.clip(dots.max(), -1.0, 1.0)))
    return best, float(mask.mean())


def sample_reachable_position(rng: np.random.Generator) -> tuple[np.ndarray, float, float]:
    """A position guaranteed reachable, drawn by running FK on random joint angles
    rather than guessing a (radius, height) box that would not match the real
    workspace shape. Returns (position, azimuth, the pitch it was reached with)."""
    limit = math.radians(100)
    t1 = rng.uniform(-math.pi, math.pi)
    t2, t3, t4 = rng.uniform(-limit, limit, size=3)
    position = ee_transform(np.array([t1, t2, t3, t4, 0.0]))[:3, 3]
    return position, t1, t2 + t3 + t4


def sample_tabletop_position(rng: np.random.Generator) -> np.ndarray:
    """A grasp point on the table the arm is bolted to, inside the radius band D023
    measured (r_inner..r_outer = 15..33 cm), a few cm up for the object's body. This
    is the distribution the task actually cares about; sample_reachable_position's
    random-joint draw wanders to heights and radii no trash object will sit at."""
    radius = rng.uniform(0.15, 0.33)
    azimuth = rng.uniform(-math.pi, math.pi)
    return np.array(
        [
            _PAN_AXIS_XY[0] + radius * math.cos(azimuth),
            _PAN_AXIS_XY[1] + radius * math.sin(azimuth),
            rng.uniform(0.0, 0.06),
        ]
    )


def random_unit_vectors(n: int, rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_matches_fk(rng: np.random.Generator) -> None:
    """The planar model and the closed-form IK must agree with the verified FK."""
    for _ in range(200):
        t2, t3, t4 = rng.uniform(-2.0, 2.0, size=3)
        radial, height = planar_fk(t2, t3, t4)
        pos = ee_transform(np.array([0.0, t2, t3, t4, 0.0]))[:3, 3]
        assert abs(radial - (pos[0] - _PAN_AXIS_XY[0])) < 1e-9, "planar radial disagrees with FK"
        assert abs(height - pos[2]) < 1e-9, "planar height disagrees with FK"

    recovered = 0
    for _ in range(200):
        t2, t3, t4 = rng.uniform(-2.0, 2.0, size=3)
        radial, height = planar_fk(t2, t3, t4)
        for elbow_up in (True, False):
            sol = solve_planar(radial, height, t2 + t3 + t4, elbow_up=elbow_up)
            if sol is None:
                continue
            r2, h2 = planar_fk(*sol)
            if abs(r2 - radial) < 1e-6 and abs(h2 - height) < 1e-6:
                recovered += 1
                break
    assert recovered == 200, f"closed-form IK recovered only {recovered}/200 configurations"

    # Approach direction from FK must match the (cos t1 cos b, sin t1 cos b, -sin b) form.
    for _ in range(200):
        q = rng.uniform(-2.0, 2.0, size=5)
        r_fk = ee_transform(q)[:3, :3][:, 0]
        b = q[1] + q[2] + q[3]
        predicted = np.array(
            [math.cos(q[0]) * math.cos(b), math.sin(q[0]) * math.cos(b), -math.sin(b)]
        )
        assert np.allclose(r_fk, predicted, atol=1e-9), "approach direction model disagrees with FK"


def ground_reach_band(
    mount_height: float, object_height: float, pitch_grid: np.ndarray, topdown_tol_deg: float = 10.0
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """D020 criterion 1, re-derived for OMX: with the arm base `mount_height` above
    the ground, which ground radii can it still pick from? Returns the (min, max)
    radius band for a top-down grasp and for any approach at all, or None.

    The grasp point sits `object_height` above the ground, so in the arm's own frame
    it is at z = object_height - mount_height (negative = below the base)."""
    height = object_height - mount_height
    radii = np.arange(0.03, 0.46, 0.005)
    topdown = []
    any_approach = []
    down_pitch = math.pi / 2  # approach (0, 0, -1)
    tol = math.radians(topdown_tol_deg)
    for radius in radii:
        mask = achievable_pitch_mask(radius, height, pitch_grid)
        if not mask.any():
            continue
        any_approach.append(radius)
        if (np.abs(pitch_grid[mask] - down_pitch) <= tol).any():
            topdown.append(radius)
    band = lambda xs: (float(xs[0]), float(xs[-1])) if xs else None  # noqa: E731
    return band(topdown), band(any_approach)


def report_mount_heights(object_height: float, pitch_grid: np.ndarray) -> None:
    print(f"D020 criterion 1 re-derived for OMX (grasp point {object_height * 100:.0f} cm above ground)")
    print("arm base height above ground -> ground radii it can still pick from\n")
    print(f"{'mount h':>9} {'top-down band':>22} {'any-approach band':>22}")
    for mount_cm in range(0, 62, 5):
        topdown, any_approach = ground_reach_band(mount_cm / 100.0, object_height, pitch_grid)
        fmt = lambda b: f"{b[0] * 100:.0f}-{b[1] * 100:.0f} cm" if b else "-- none --"  # noqa: E731
        print(f"{mount_cm:>7} cm {fmt(topdown):>22} {fmt(any_approach):>22}")
    print("\nUPPER BOUND ONLY: omx_f.urdf joint limits are placeholders (+/-2pi, D026), and")
    print("self-collision, the mounting bracket and the gripper body are not modelled.")


def main() -> None:
    parser = argparse.ArgumentParser(description="OMX 5-DOF geometry probes")
    parser.add_argument("--mode", choices=("grasp", "mount"), default="grasp")
    parser.add_argument("--n", type=int, default=2000, help="candidate poses (grasp mode)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pitch-steps", type=int, default=2000)
    parser.add_argument("--object-height-cm", type=float, default=3.0, help="mount mode")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    test_matches_fk(rng)
    print("self-check: planar model, closed-form IK and approach model all agree with fk.ee_transform\n")

    if args.mode == "mount":
        report_mount_heights(
            args.object_height_cm / 100.0, np.linspace(-math.pi, math.pi, args.pitch_steps)
        )
        return

    pitch_grid = np.linspace(-math.pi, math.pi, args.pitch_steps)
    straight_down = np.array([0.0, 0.0, -1.0])
    targets = random_unit_vectors(args.n, rng)
    random_errors = np.empty(args.n)
    down_errors = np.empty(args.n)
    arc_fractions = np.empty(args.n)

    for i in range(args.n):
        position = sample_tabletop_position(rng)
        random_errors[i], arc_fractions[i] = min_approach_error_rad(position, targets[i], pitch_grid)
        down_errors[i], _ = min_approach_error_rad(position, straight_down, pitch_grid)
    random_errors = np.degrees(random_errors)
    down_errors = np.degrees(down_errors)

    print(f"n = {args.n} grasp points on the table, r 15-33 cm (D023's band), z 0-6 cm")
    print(f"achievable pitch arc: median {100 * np.median(arc_fractions):.0f}% of the full circle")
    print(
        f"top-down (straight down) approach achievable within 10 deg: "
        f"{100 * (down_errors <= 10).mean():.1f}% of those points\n"
    )
    print("a base-agnostic detector's candidate, drawn with orientation independent of position:")
    print(f"{'tol':>6} {'feasible':>10} {'sin(tol) ceiling':>18}")
    for tol_deg in (5, 10, 15, 20, 30, 45):
        pct = 100.0 * (random_errors <= tol_deg).mean()
        ceiling = 100.0 * math.sin(math.radians(tol_deg))
        print(f"{tol_deg:>5}d {pct:>9.1f}% {ceiling:>17.1f}%")
    print(
        "\nper-candidate is the WRONG statistic for a go/no-go: an object yields many\n"
        "candidates and only one has to work. At 11% each, k independent candidates give"
    )
    for k in (10, 30, 100):
        print(f"  k={k:>3}: {100 * (1 - 0.89**k):.1f}% chance at least one is feasible")
    print("(independence is optimistic: real candidates cluster; needs real GSNet output to settle)")


if __name__ == "__main__":
    main()
