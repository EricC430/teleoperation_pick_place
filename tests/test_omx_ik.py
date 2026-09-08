"""omx_ik must agree with reach_logger.fk, which is the validated forward model."""

from __future__ import annotations

import math

import numpy as np
import pytest

from omx_ik import ik, kinematics as kin
from omx_ik.grasp_filter import GraspCandidate, filter_grasps, summarise
from omx_ik.workspace import CAMERA_MOUNT_SECTOR_2026_08_31, PanWindow
from reach_logger.fk import ee_transform

RNG = np.random.default_rng(20260905)


def random_joints(n: int) -> np.ndarray:
    """Random configurations, excluding those that park the EE on the pan axis where
    the arm plane is undefined (kinematics.AXIS_SINGULARITY_RADIUS_M). That region is
    inside the robot's own base; TestAxisSingularity covers it separately."""
    kept = []
    while len(kept) < n:
        joints = RNG.uniform(-2.0, 2.0, size=5)
        position = ee_transform(joints)[:3, 3]
        horizontal = math.hypot(
            position[0] - kin.PAN_AXIS_XY[0], position[1] - kin.PAN_AXIS_XY[1]
        )
        if horizontal > 1.5 * kin.AXIS_SINGULARITY_RADIUS_M:
            kept.append(joints)
    return np.array(kept)


class TestPlanarModelMatchesFK:
    def test_radial_and_height_match(self):
        for joints in random_joints(300):
            t1, t2, t3, t4, roll = joints
            radial, height = kin.planar_fk(t2, t3, t4, roll)
            pos = ee_transform(joints)[:3, 3]
            horizontal = math.hypot(pos[0] - kin.PAN_AXIS_XY[0], pos[1] - kin.PAN_AXIS_XY[1])
            expected = math.hypot(radial, kin.lateral_offset(roll))
            assert horizontal == pytest.approx(expected, abs=1e-9)
            assert pos[2] == pytest.approx(height, abs=1e-9)

    def test_approach_direction_matches(self):
        for joints in random_joints(300):
            predicted = kin.approach_direction(joints[0], joints[1] + joints[2] + joints[3])
            assert np.allclose(ee_transform(joints)[:3, 0], predicted, atol=1e-9)

    def test_planar_ik_inverts_planar_fk(self):
        for joints in random_joints(300):
            _, t2, t3, t4, roll = joints
            radial, height = kin.planar_fk(t2, t3, t4, roll)
            pitch = t2 + t3 + t4
            recovered = [
                kin.solve_planar(radial, height, pitch, roll, elbow_up=up) for up in (True, False)
            ]
            assert any(
                sol is not None
                and kin.planar_fk(*sol, roll)[0] == pytest.approx(radial, abs=1e-9)
                and kin.planar_fk(*sol, roll)[1] == pytest.approx(height, abs=1e-9)
                for sol in recovered
            )


class TestSolvePose:
    def test_round_trips_any_reachable_pose(self):
        for joints in random_joints(300):
            pose = ee_transform(joints)
            solution = ik.solve_pose(pose[:3, 3], pose[:3, :3])
            assert solution is not None
            assert solution.position_error_m < 1e-9
            assert solution.approach_error_rad < 1e-9

    def test_recovered_joints_reproduce_the_full_rotation(self):
        for joints in random_joints(100):
            pose = ee_transform(joints)
            solution = ik.solve_pose(pose[:3, 3], pose[:3, :3])
            assert np.allclose(ee_transform(solution.joints), pose, atol=1e-9)

    def test_far_out_of_reach_position_has_no_solution(self):
        rotation = ee_transform(np.zeros(5))[:3, :3]
        assert ik.solve_pose(np.array([2.0, 0.0, 0.0]), rotation) is None


class TestAxisSingularity:
    """On the pan axis joint1 is undetermined, so the solver must decline rather than
    return a pose it cannot hold. Scanned over 20k random configurations, this is the
    only region where solve_pose fails to round-trip."""

    def test_targets_on_the_pan_axis_are_declined(self):
        rotation = ee_transform(np.zeros(5))[:3, :3]
        for offset in (0.0, 0.002, 0.009):
            position = np.array([kin.PAN_AXIS_XY[0] + offset, kin.PAN_AXIS_XY[1], 0.25])
            assert ik.solve_pose(position, rotation) is None

    def test_the_excluded_region_is_inside_the_base(self):
        assert kin.AXIS_SINGULARITY_RADIUS_M < 0.02


class TestApproachErrorIsPurelyOutOfPlane:
    """The arm cannot yaw its approach independently of where the target is, so the
    residual must equal exactly the out-of-plane component and nothing else."""

    def test_great_circle_distance_is_a_lower_bound_and_usually_tight(self):
        """The achievable approaches form a great circle, so the out-of-plane angle is
        a hard floor. It is only *reached* when the pitch it needs is also holdable at
        that position — the reachable pitch arc is not the whole circle."""
        tight = 0
        checked = 0
        for joints in random_joints(200):
            position = ee_transform(joints)[:3, 3]
            approach = RNG.normal(size=3)
            approach /= np.linalg.norm(approach)
            solution = ik.solve_closest(position, approach, roll=0.0)
            if solution is None or solution.position_error_m > 1e-6:
                continue
            azimuth = solution.joints[0]
            floor = math.asin(abs(float(np.dot(approach, kin.lateral_unit(azimuth)))))
            assert solution.approach_error_rad >= floor - 1e-9
            checked += 1
            tight += solution.approach_error_rad == pytest.approx(floor, abs=1e-6)
        assert checked > 100
        # The invariant is that the arc restriction is real but not total. The actual
        # rate is data, not a law: ~46% for isotropic approaches at random reachable
        # positions, measured 2026-09-05.
        assert 0.2 < tight / checked < 1.0

    def test_in_plane_approach_is_matched_exactly(self):
        for joints in random_joints(200):
            position = ee_transform(joints)[:3, 3]
            azimuth = joints[0]
            pitch = joints[1] + joints[2] + joints[3]
            solution = ik.solve(position, kin.approach_direction(azimuth, pitch), roll=joints[4])
            assert solution is not None
            assert solution.approach_error_rad < 1e-9


class TestGraspFilter:
    def _candidates(self, n: int, isotropic: bool) -> list[GraspCandidate]:
        candidates = []
        for i, joints in enumerate(random_joints(n)):
            pose = ee_transform(joints)
            if isotropic:
                approach = RNG.normal(size=3)
                approach /= np.linalg.norm(approach)
                reference = np.array([0.0, 0.0, 1.0])
                if abs(np.dot(approach, reference)) > 0.99:
                    reference = np.array([0.0, 1.0, 0.0])
                z_axis = np.cross(approach, reference)
                z_axis /= np.linalg.norm(z_axis)
                rotation = np.column_stack([approach, np.cross(z_axis, approach), z_axis])
            else:
                rotation = pose[:3, :3]
            candidates.append(
                GraspCandidate(pose[:3, 3], rotation, object_id=f"obj{i // 20}", score=float(i))
            )
        return candidates

    def test_arm_generated_poses_are_all_geometrically_solvable(self):
        """Poses taken straight off the arm are reachable by construction — but not all
        of them are valid *targets*: some configurations park the EE behind the base,
        and the default filter rejects those. Geometry and task rules are separate."""
        candidates = self._candidates(60, isotropic=False)
        permissive = filter_grasps(candidates, allow_fold_through_base=True)
        assert all(r.feasible for r in permissive)
        default = filter_grasps(candidates)
        rejected = [d for d, p in zip(default, permissive) if p.feasible and not d.feasible]
        assert all(r.solution.folds_through_base for r in rejected)

    def test_isotropic_orientations_mostly_fail_but_not_all(self):
        summary = summarise(filter_grasps(self._candidates(300, isotropic=True)))
        assert 0.0 < summary["feasible_pct"] < 40.0
        assert summary["object_coverage_pct"] >= summary["feasible_pct"]

    def test_summary_counts_objects(self):
        summary = summarise(filter_grasps(self._candidates(40, isotropic=False)))
        assert summary["candidates"] == 40
        assert summary["objects"] == 2
        assert summary["object_coverage_pct"] == 100.0


class TestPanWindow:
    """Where the arm may point, on top of what it can reach: the 2026-08-31 sector is
    a camera-mount collision limit, not a cable or field-of-view limit."""

    def test_wraps_correctly(self):
        window = PanWindow(-90.0, 45.0)
        assert window.contains(0.0)
        assert window.contains(math.radians(-89.0))
        assert window.contains(math.radians(44.0))
        assert not window.contains(math.radians(90.0))
        assert not window.contains(math.radians(180.0))
        # same pose, wound a full turn
        assert window.contains(math.radians(-90.0) + 2 * math.pi)

    def test_window_constrains_joint1_but_does_not_remove_fold_through_solutions(self):
        """Pinned deliberately: a fold-through solution can sit inside the joint1
        window while the target is behind the arm. The two constraints are separate."""
        window = PanWindow(-90.0, 45.0)
        folds_inside_window = 0
        for joints in random_joints(200):
            pose = ee_transform(joints)
            limited = ik.solve_pose(pose[:3, 3], pose[:3, :3], pan_window=window)
            if limited is None:
                continue
            assert window.contains(limited.joints[0])
            if limited.folds_through_base:
                folds_inside_window += 1
        assert folds_inside_window > 0

    def test_filter_rejects_fold_through_by_default(self):
        candidates = TestGraspFilter()._candidates(200, isotropic=False)
        default = filter_grasps(candidates)
        assert all(
            not r.solution.folds_through_base for r in default if r.feasible
        )
        permissive = filter_grasps(candidates, allow_fold_through_base=True)
        assert sum(r.feasible for r in permissive) >= sum(r.feasible for r in default)

    def test_filter_honours_the_window(self):
        candidates = TestGraspFilter()._candidates(200, isotropic=True)
        window = PanWindow(-90.0, 45.0)
        wide = summarise(filter_grasps(candidates))
        narrow = summarise(filter_grasps(candidates, pan_window=window))
        assert narrow["feasible"] <= wide["feasible"]

    def test_measured_sector_matches_the_spec(self):
        assert CAMERA_MOUNT_SECTOR_2026_08_31.min_deg == -90.0
        assert CAMERA_MOUNT_SECTOR_2026_08_31.max_deg == 45.0
        assert CAMERA_MOUNT_SECTOR_2026_08_31.width_deg == 135.0
        assert "camera" in CAMERA_MOUNT_SECTOR_2026_08_31.note
