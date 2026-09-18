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

SIGN evidence (main f64bbbe, merged 2026-09-18)
-----------------------------------------------
⚠️ The renders below were made BEFORE the unit correction above, i.e. with `.pos` read as degrees
(body angles ~1.8x too small). A +1 vs -1 flip of shoulder_lift is a gross difference and the
control run discriminated it, so that conclusion very likely survives [AI推論]; the "no mismatch
for the other five" observation was made at the wrong scale and should be re-rendered with this
file's conversion. The same re-render is also an empirical check of the unit correction itself.

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

Whether "+" in the recorded `.pos` values matches "+" in the URDF/sim joint frame was, before the above,
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
    print("sign:", SIGN)
    print("  shoulder_lift = +1: render vs real video with a flipped control (main f64bbbe; rendered pre-unit-fix)")
    print("  the other five, BODY_ZERO_DEG, GRIPPER_ZERO_DEG: [未確認] — S4 §5-1 settles them")
