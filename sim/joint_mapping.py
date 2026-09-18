"""Real recording <-> sim joint mapping. Shared by S4 (live teleop-in-sim) and S5 (replay + DR).

Both specs need to turn one row of a LeRobot `action` / `observation.state` column -- six floats,
one per motor, in the fixed order `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll, gripper` -- into Isaac Lab joint position targets in radians. This is that one
conversion, written once so S4's live collector and S5's replay/gain-fit scripts cannot drift
apart on it silently.

🔴 [已查證 2026-09-18] the recorded `action`/`observation.state` columns are DEGREES, not a
normalized -100..100 or 0..100 range, despite `RANGE_0_100` being the gripper's *encoder*
normalization mode per `S4_sim_teleop_collect.md` §4. Read directly off
`data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/meta/stats.json`: e.g.
`shoulder_lift` spans roughly -66..+41 across 60 episodes, which sits inside that joint's factory
travel (-120..+90 deg, `omx_constants.py`) and would not make sense as a percentage or as
raw-tick counts. Cross-checked against episode 0's parquet rows directly: the `gripper` channel
reads ~59.5 while visibly open (start/end of the episode) and drops to ~47.5 while visibly closed
(mid-episode) -- a ~12-unit swing, degree-shaped, not a 0..100 open/closed percentage either.
`omx_constants.Joint.factory_lo_deg/hi_deg` already treats the gripper the same way (its factory
range is written as `0.0, 100.0` and run through `math.radians()` like every other joint) --
this module keeps that existing convention rather than inventing a second one.

⚠️ [未確認] SIGN. Whether "+" in the recorded degrees matches "+" in the URDF/sim joint frame is
NOT verified for any of the six joints. `S4_sim_teleop_collect.md` §5 item 1 names this as the
single easiest way to record a dataset that looks fine and trains a silently mirrored policy, and
gates it on a five-pose comparison test (home / J1-only / J2-only / J3-only / gripper open-close)
that has not been run as of 2026-09-18 -- only the mimic-gearing half of that test
(`verify_mimic_gearing.py`, gap 2) exists so far. `SIGN` below defaults to +1 for all six joints:
an assumption, not a result. Anything downstream (S5 gap-1 gain fit, gap-3 grasp trigger) that
reports "sim tracks real to N degrees" is only claiming that for a sign-unconfirmed transform of
real -- it is not yet a sim-vs-real claim. Flip an entry here once the five-pose test settles it,
the same way `--mimic-gearing` exists for gap 2's sign.
"""

from __future__ import annotations

import math

import omx_constants as K

# The dataset's `action`/`observation.state` column order (LeRobot feature `names`, e.g.
# `data/huggingface/lerobot/ericc430/omx_pick_place_pilot_uvc_60/meta/info.json`).
DATASET_JOINT_ORDER: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# Must match omx_constants.JOINTS 1:1, in order -- if someone reorders JOINTS this mapping goes
# silently wrong, so fail loudly at import time instead.
assert tuple(j.lerobot_name for j in K.JOINTS) == DATASET_JOINT_ORDER, (
    "omx_constants.JOINTS order no longer matches the dataset's action/observation.state column "
    "order -- fix joint_mapping.py before trusting anything built on it."
)

# [未確認] see module docstring. Flip to -1.0 for a joint once the five-pose test decides it.
SIGN: dict[str, float] = {name: 1.0 for name in DATASET_JOINT_ORDER}


def row_to_sim_rad(row: "list[float] | tuple[float, ...]") -> list[float]:
    """One dataset row (6 floats, degrees, `DATASET_JOINT_ORDER`) -> sim joint radians, same order."""
    if len(row) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(row)}")
    return [SIGN[name] * math.radians(float(v)) for name, v in zip(DATASET_JOINT_ORDER, row)]


def sim_rad_to_row(rad: "list[float] | tuple[float, ...]") -> list[float]:
    """Inverse of `row_to_sim_rad` -- sim joint radians -> dataset-unit degrees. Mostly for
    printing sim state back out in the same units the real dataset uses, for comparison tables."""
    if len(rad) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(rad)}")
    return [math.degrees(float(v)) / SIGN[name] for name, v in zip(DATASET_JOINT_ORDER, rad)]


if __name__ == "__main__":
    print("dataset joint order:", DATASET_JOINT_ORDER)
    print("sign (all [未確認], defaults to +1 -- see module docstring):", SIGN)
    sample = [-7.8, -30.9, 20.2, -17.6, -0.9, 55.8]  # ~ dataset mean row, from stats.json
    rad = row_to_sim_rad(sample)
    print(f"sample dataset row (deg): {sample}")
    print(f"-> sim radians:           {[round(r, 4) for r in rad]}")
    print(f"-> back to deg (roundtrip): {[round(d, 3) for d in sim_rad_to_row(rad)]}")
