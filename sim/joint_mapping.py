"""LeRobot `.pos` units <-> URDF joint radians, for the OMX-F. Pure numpy — runs on the host AND in
the isaac-lab container, so the two sides cannot disagree about units.

Used by S5 (`scripts/s5_prepare_replay.py`, `sim/fit_drive_gains*.py`, `sim/verify_grasp_attach.py`)
and meant to be reused by S4. Two call styles, one conversion underneath:
  * array:  `lerobot_to_urdf_rad(pos)` / `urdf_rad_to_lerobot(q)`   — (..., 6) numpy
  * row:    `row_to_sim_rad(row)` / `sim_rad_to_row(rad)`            — one 6-float list

🔴 2026-09-18 merge correction: a parallel version of this file (main 33d7e08) read the recorded
`.pos` values as DEGREES, reasoning from `stats.json` ranges (-66..+41 "fits a joint's travel").
That range fits -100..100 just as well, so it did not tell the two apart; the code does (below).
Degrees-as-is under-reads body angles ~1.8x (1 unit = 4095/200 ticks = 1.8 deg) and puts the
gripper on the wrong zero and scale. Anything computed with that version — the S5 gap-1 "5 combos
on uvc_60 ep 0" and verify_grasp_attach's ep 0 run — needs a rerun before its numbers mean anything.

What is VERIFIED here and what is NOT
-------------------------------------
`[已查證 2026-09-18]` the LeRobot side, from `lerobot/` commit a16f34c0:
  * `OmxFollowerConfig.use_degrees` defaults to False, and no config in `configs/` sets it
    -> the five body joints are `MotorNormMode.RANGE_M100_100`, the gripper is `RANGE_0_100`
    (`robots/omx_follower/omx_follower.py:50-62`).
  * normalisation is `norm = (raw - range_min) / (range_max - range_min) * 200 - 100`
    (`motors/motors_bus.py:_normalize`), and `-norm` when `drive_mode` is set.
  * every follower calibration in `calibration/` from 2026-09-13 on is the factory default
    (`range_min=0, range_max=4095, drive_mode=0, homing_offset=0`), which is what makes the
    conversion below a constant. `check_calibration()` refuses anything else.
  * XL330/XL430 position resolution is 4096 ticks per revolution.

`[未確認]` the URDF side — SIGN (per joint), BODY_ZERO_DEG, GRIPPER_ZERO_DEG:
  whether raw tick 2048 is the URDF's zero, and whether +tick is +URDF-angle, is NOT read off any
  spec. The defaults (+1, 0) are the simplest guess. They are settled by the S4 §5-1 five-pose
  comparison (home / J1 only / J2 only / J3 only / gripper), not by this file. Everything downstream
  (gain fit, replay) is only as right as these eight numbers.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

LEROBOT_NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
URDF_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "gripper_joint_1")

TICKS_PER_REV = 4096
RANGE_MIN, RANGE_MAX = 0, 4095
CENTER_TICK = 2048

# The dataset's `action`/`observation.state` column order, under the name S4/S5 sim scripts use.
DATASET_JOINT_ORDER = LEROBOT_NAMES

# [未確認] — see the module docstring. The ONE sign knob: flip an entry to -1.0 once the five-pose
# test decides it. `fit_drive_gains.py --sign-override` mutates this dict for a single run.
SIGN: dict[str, float] = {name: 1.0 for name in LEROBOT_NAMES}
# Order follows LEROBOT_NAMES[:5].
BODY_ZERO_DEG = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
# [未確認] Real uvc_60 data: open ~59, closed on a paper cup ~47-50 (gripper.pos units). With the
# defaults below that is +32 deg open / -11 deg closed, and the USD limits gripper_joint_1 to
# 0..100 deg — so at least one of these two numbers is probably wrong. The mimic check prints the
# finger gap per angle; pick the offset that makes "59 = open" land on an open gap.
GRIPPER_ZERO_DEG = 0.0

try:  # sim/ is on the path for every caller; fail loudly if omx_constants reorders its joints
    import omx_constants as _K
except ImportError:
    _K = None
if _K is not None:
    assert tuple(j.lerobot_name for j in _K.JOINTS) == LEROBOT_NAMES, (
        "omx_constants.JOINTS order no longer matches the dataset's action/observation.state column "
        "order -- fix joint_mapping.py before trusting anything built on it.")


def _body_sign() -> np.ndarray:
    return np.array([SIGN[n] for n in LEROBOT_NAMES[:5]])


def check_calibration(path: str | Path) -> list[str]:
    """Return problems; empty list = the constant conversion in this module is valid."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for name in LEROBOT_NAMES:
        c = cal.get(name)
        if c is None:
            problems.append(f"{name}: missing")
            continue
        if (c["range_min"], c["range_max"]) != (RANGE_MIN, RANGE_MAX):
            problems.append(f"{name}: range {c['range_min']}..{c['range_max']} != factory {RANGE_MIN}..{RANGE_MAX}")
        if c["drive_mode"] != 0:
            problems.append(f"{name}: drive_mode={c['drive_mode']} (follower is expected to be 0)")
    return problems


def _norm_to_raw(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (x - lo) / (hi - lo) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN


def _raw_to_motor_deg(raw: np.ndarray) -> np.ndarray:
    return (raw - CENTER_TICK) * 360.0 / TICKS_PER_REV


def lerobot_to_urdf_rad(pos: np.ndarray) -> np.ndarray:
    """(..., 6) LeRobot `.pos` values in LEROBOT_NAMES order -> (..., 6) radians in URDF_NAMES order."""
    pos = np.asarray(pos, dtype=np.float64)
    out = np.empty_like(pos)
    body_deg = _raw_to_motor_deg(_norm_to_raw(pos[..., :5], -100.0, 100.0))
    out[..., :5] = np.radians(_body_sign() * body_deg + BODY_ZERO_DEG)
    grip_deg = _raw_to_motor_deg(_norm_to_raw(pos[..., 5], 0.0, 100.0))
    out[..., 5] = math.radians(1.0) * (SIGN["gripper"] * grip_deg + GRIPPER_ZERO_DEG)
    return out


def urdf_rad_to_lerobot(q: np.ndarray) -> np.ndarray:
    """Inverse of `lerobot_to_urdf_rad` (no tick quantisation, no clamping)."""
    q = np.asarray(q, dtype=np.float64)
    out = np.empty_like(q)
    body_deg = (np.degrees(q[..., :5]) - BODY_ZERO_DEG) / _body_sign()
    raw = body_deg * TICKS_PER_REV / 360.0 + CENTER_TICK
    out[..., :5] = (raw - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * 200.0 - 100.0
    grip_deg = (np.degrees(q[..., 5]) - GRIPPER_ZERO_DEG) / SIGN["gripper"]
    raw = grip_deg * TICKS_PER_REV / 360.0 + CENTER_TICK
    out[..., 5] = (raw - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * 100.0
    return out


def row_to_sim_rad(row: "list[float] | tuple[float, ...]") -> list[float]:
    """One dataset row (6 LeRobot `.pos` values, DATASET_JOINT_ORDER) -> sim joint radians, same order."""
    if len(row) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(row)}")
    return lerobot_to_urdf_rad(np.asarray(row, dtype=np.float64)).tolist()


def sim_rad_to_row(rad: "list[float] | tuple[float, ...]") -> list[float]:
    """Inverse of `row_to_sim_rad` -- sim joint radians -> LeRobot `.pos` units (not degrees)."""
    if len(rad) != 6:
        raise ValueError(f"expected 6 values in {DATASET_JOINT_ORDER}, got {len(rad)}")
    return urdf_rad_to_lerobot(np.asarray(rad, dtype=np.float64)).tolist()


if __name__ == "__main__":
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    probe = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 50.0], [100.0, -100.0, 50.0, -50.0, 10.0, 59.0]])
    q = lerobot_to_urdf_rad(probe)
    back = urdf_rad_to_lerobot(q)
    np.set_printoptions(precision=3, suppress=True)
    print("lerobot .pos         ->", probe.tolist())
    print("urdf deg             ->", np.degrees(q).tolist())
    print("round-trip max error ->", float(np.abs(back - probe).max()))
    row = [-7.8, -30.9, 20.2, -17.6, -0.9, 55.8]
    row_back = sim_rad_to_row(row_to_sim_rad(row))
    print("row API round-trip   ->", max(abs(a - b) for a, b in zip(row, row_back)))
    print("🔴 SIGN / BODY_ZERO_DEG / GRIPPER_ZERO_DEG are [未確認] — S4 §5-1 settles them.")
