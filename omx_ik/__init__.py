"""Closed-form IK and grasp-candidate filtering for the OMX 5-DOF arm.

Local trial code (2026-09-05), not yet tied to a decision record. Serves 路徑 A/B,
both of which need IK that D027 records as "no IK implementation started".
"""

from omx_ik.ik import IKSolution, solve, solve_closest, solve_pose

__all__ = ["IKSolution", "solve", "solve_closest", "solve_pose"]
