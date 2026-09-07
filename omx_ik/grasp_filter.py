"""Filter a grasp detector's candidates down to the ones OMX can actually execute.

Feeds 路徑 A (IK -> pre-grasp -> GSNet pose -> IK transport). A detector like GSNet
scores grasps from object geometry alone and has no idea where the robot base is, so
a large share of what it proposes asks the gripper to arrive from a direction this
arm cannot produce.

Read the per-candidate rate with care — it is not the go/no-go number. An object
yields many candidates and only one has to be executable, so `summarise` reports
per-object coverage too when candidates carry object ids. Structurally: for OMX the
achievable approach directions at a given position form an arc of a great circle, so
per-candidate feasibility for isotropically-oriented candidates runs at single-digit
to low-double-digit percent while per-object coverage is far higher.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from omx_ik.ik import IKSolution, solve_pose
from omx_ik.workspace import PanWindow


@dataclass(frozen=True)
class GraspCandidate:
    position: np.ndarray  # (3,) in the arm's base frame
    rotation: np.ndarray  # (3, 3); first column is the approach axis
    object_id: str | None = None
    score: float | None = None


@dataclass(frozen=True)
class FilterResult:
    candidate: GraspCandidate
    solution: IKSolution | None
    feasible: bool


def filter_grasps(
    candidates: Sequence[GraspCandidate],
    position_tol_m: float = 1e-3,
    approach_tol_deg: float = 10.0,
    pan_window: PanWindow | None = None,
    allow_fold_through_base: bool = False,
) -> list[FilterResult]:
    """Two independent restrictions, both off by default in the solver and applied here:

    `pan_window` — where the arm may point (camera-mount collision sector, see
    omx_ik.workspace). Constrains joint1.

    `allow_fold_through_base` — a `back` solution reaches around through the base axis
    and leaves the object *behind* the arm. Defaults to rejecting them: targets belong
    in front of or beside the arm. Note this is NOT implied by `pan_window`; a fold
    solution can hold joint1 inside the window while the target sits 180 degrees away.
    """
    results = []
    for candidate in candidates:
        solution = solve_pose(
            candidate.position,
            candidate.rotation,
            pan_window=pan_window,
            allow_fold_through_base=allow_fold_through_base,
        )
        feasible = solution is not None and solution.within(position_tol_m, approach_tol_deg)
        results.append(FilterResult(candidate, solution, feasible))
    return results


def summarise(results: Sequence[FilterResult]) -> dict[str, float | int | None]:
    total = len(results)
    feasible = sum(1 for r in results if r.feasible)
    # NOT "position unreachable": solve_pose declines when the pitch the requested
    # approach implies cannot be held at that point, which is a different and much
    # commoner thing than the point itself being out of range.
    no_solution = sum(1 for r in results if r.solution is None)
    folding = sum(1 for r in results if r.feasible and r.solution.folds_through_base)

    approach_errors = [
        r.solution.approach_error_deg for r in results if r.solution is not None
    ]
    object_ids = {r.candidate.object_id for r in results if r.candidate.object_id is not None}
    covered = {
        r.candidate.object_id for r in results if r.feasible and r.candidate.object_id is not None
    }

    return {
        "candidates": total,
        "feasible": feasible,
        "feasible_pct": 100.0 * feasible / total if total else 0.0,
        "no_solution_at_requested_pose": no_solution,
        "feasible_only_by_folding_through_base": folding,
        "median_approach_error_deg": float(np.median(approach_errors)) if approach_errors else None,
        "objects": len(object_ids) or None,
        "objects_with_a_feasible_grasp": len(covered) if object_ids else None,
        "object_coverage_pct": (100.0 * len(covered) / len(object_ids)) if object_ids else None,
    }


def best_solution(results: Sequence[FilterResult]) -> FilterResult | None:
    """The executable candidate to actually run: highest detector score among the
    feasible ones, falling back to the smallest approach error when unscored."""
    feasible = [r for r in results if r.feasible and r.solution is not None]
    if not feasible:
        return None
    scored = [r for r in feasible if r.candidate.score is not None]
    if scored:
        return max(scored, key=lambda r: r.candidate.score)
    return min(feasible, key=lambda r: r.solution.approach_error_rad)
