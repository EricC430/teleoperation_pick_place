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

✅ [已查證 2026-09-18, shoulder_lift only] SIGN for `shoulder_lift` is **+1**, confirmed by render
with a control. `sim/render_state_replay.py` posed the arm at episode 0's recorded
`observation.state` and rendered it; `scripts/compare_sim_real_frames.py` put those beside the
real recorded video at the same timestamps. At frame 226 (the real arm reaching down to the cup):
SIGN=+1 renders the arm extended forward at table height, matching the real frame; the control run
with `--sign-override shoulder_lift=-1` renders it pointing nearly straight UP, grossly wrong.
Outputs: `outputs/sign_check_ep0/` and `outputs/sign_check_ep0_liftneg/`. This agrees with two
earlier independent lines of evidence (the effort-limit diagnostic in `fit_drive_gains.py`, and
the `reach_logger/fk.py` geometry argument) -- see `sim/README.md` "Two gaps" §1.

⚠️ [未確認] SIGN for the other five joints. The same six renders show no configuration mismatch
against the real video, which IS evidence that none of them is flipped -- a flipped
`shoulder_pan`, `elbow_flex` or `wrist_flex` would visibly distort the arm the same way
`shoulder_lift` did. But **no per-joint control run was done for them**, and the evidence is
weakest exactly where the visual effect is smallest: `wrist_roll` (rotates the gripper about its
own axis -- subtler than an arm-configuration change) and `gripper` (whose sim amplitude is
independently known to be wrong, see gap 2's residual). Running a control for any of them is one
`render_state_replay.py` invocation with `--sign-override`, ~4 minutes.

Whether "+" in the recorded degrees matches "+" in the URDF/sim joint frame was, before the above,
NOT verified for any of the six joints. `S4_sim_teleop_collect.md` §5 item 1 names this as the
single easiest way to record a dataset that looks fine and trains a silently mirrored policy, and
gates it on a five-pose comparison test (home / J1-only / J2-only / J3-only / gripper open-close)
that has not been run as of 2026-09-18 on real hardware. `SIGN` below still reads +1 for all six
joints -- for `shoulder_lift` that is now a result (see above), for the rest it remains a default
that the render comparison supports but no control run has isolated. Flip an entry here if a
control or the five-pose test ever contradicts it, the same way `--mimic-gearing` exists for gap
2's sign.

Note the render check does NOT replace the five-pose test for calibration-grade questions: it
catches gross errors (flipped sign, swapped joint), not angle offsets or scale errors, and the sim
camera pose is still a PLACEHOLDER (S5 gap 4). S4 §5-5 draws the same line and calls this class of
check T4, a smoke test, explicitly not an acceptance test.
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

# 🔴 ZERO OFFSETS -- known to be NON-ZERO, values unknown. Added 2026-09-21.
#
# The recorded degrees come from lerobot's calibration (homing offset + encoder ticks). The URDF's
# zero pose is a particular mechanical configuration. Nothing ever made those two agree, and the
# data says they do not:
#   * `[Eric說 2026-09-21]` the arm base sits 15 cm above the table. Applying that alone makes the
#     replayed gripper land ~30 cm above the table at the grasp instant -- impossible for a ~9 cm
#     paper cup, so a term is missing.
#   * A sign error is ruled out: flipping elbow_flex turns the x-reach correlation against the
#     known placements from +0.63 to -0.34.
#   * A constant-offset model reconciles it: fitting offsets over all 60 episodes' grasp frames
#     solves a grasp height of 8.2 cm above the table (i.e. 6.8 cm BELOW the base) -- exactly the
#     shape of reaching down to a cup.
# 🔴 The fitted values are NOT usable: shoulder_lift / elbow_flex / wrist_flex all rotate about the
#    same y axis in the same plane, so their offsets trade off against each other (the fit returned
#    elbow +32.5 deg and wrist_flex +32.4 deg, which is the degeneracy talking, not a measurement),
#    and 9.3 cm of horizontal residual remains. **This cannot be resolved from recorded data alone.**
#    It needs S4 §5-1's five-pose test: move ONE joint to a physically unambiguous pose (a link
#    exactly vertical or horizontal, checked with a protractor / phone angle app), read what the
#    leader reports, and the difference IS that joint's offset. Two poses per joint confirms it is
#    an offset rather than a scale error.
# Until measured these stay 0.0, which is an assumption, not a result -- same status the SIGN table
# had before the render check settled shoulder_lift.
OFFSET_RAD: dict[str, float] = {name: 0.0 for name in DATASET_JOINT_ORDER}


def row_to_sim_rad(row: "list[float] | tuple[float, ...]") -> list[float]:
    """One dataset row (6 floats, degrees, `DATASET_JOINT_ORDER`) -> sim joint radians, same order."""
    if len(row) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(row)}")
    return [SIGN[name] * math.radians(float(v)) + OFFSET_RAD[name]
            for name, v in zip(DATASET_JOINT_ORDER, row)]


def sim_rad_to_row(rad: "list[float] | tuple[float, ...]") -> list[float]:
    """Inverse of `row_to_sim_rad` -- sim joint radians -> dataset-unit degrees. Mostly for
    printing sim state back out in the same units the real dataset uses, for comparison tables."""
    if len(rad) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(rad)}")
    return [math.degrees((float(v) - OFFSET_RAD[name]) / SIGN[name])
            for name, v in zip(DATASET_JOINT_ORDER, rad)]


if __name__ == "__main__":
    print("dataset joint order:", DATASET_JOINT_ORDER)
    print("sign:", SIGN)
    print("  shoulder_lift = +1 is [已查證 2026-09-18] (render vs real video, with a flipped control)")
    print("  the other five are [未確認]: consistent with the same renders, but no control run isolated them")
    sample = [-7.8, -30.9, 20.2, -17.6, -0.9, 55.8]  # ~ dataset mean row, from stats.json
    rad = row_to_sim_rad(sample)
    print(f"sample dataset row (deg): {sample}")
    print(f"-> sim radians:           {[round(r, 4) for r in rad]}")
    print(f"-> back to deg (roundtrip): {[round(d, 3) for d in sim_rad_to_row(rad)]}")
