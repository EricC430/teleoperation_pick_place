"""OMX-F physical constants — the single source of truth for every simulation config.

Nothing here is guessed. Every number is either read out of the URDF, taken from the
ROBOTIS specification page, or explicitly marked as PROVISIONAL with the reason.

Sources
-------
`[已查證 2026-09-03]` https://docs.robotis.com/docs/systems/omx/specifications/hardware/
`[已查證 2026-09-03]` https://emanual.robotis.com/docs/en/dxl/x/xl430-w250/
`[已查證 2026-09-03]` https://emanual.robotis.com/docs/en/dxl/x/xl330-m288/
`[已查證 2026-08-31]` assets/omx_f/omx_f.urdf   (D026)
`docs/decisions.md` D025 §2026-09-03, D029.

🔴 The URDF's own joint limits are PLACEHOLDERS (+/-6.283 rad, effort 1000, velocity 4.8).
   Anything that reads limits from the URDF is wrong. Read them from here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# --------------------------------------------------------------------------------------
# Actuators
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Motor:
    """A DYNAMIXEL model. Torque/speed are quoted at the OMX's operating voltage."""

    model: str
    volts: float
    stall_torque_nm: float
    no_load_rpm: float
    gear_ratio: float

    @property
    def velocity_limit_rad_s(self) -> float:
        return self.no_load_rpm * 2.0 * math.pi / 60.0


# ⚠️ [未確認] OMX-F is specified as a 12 VDC system, but the XL330 tolerates only 3.7-6.0 V,
#    so the XL330s must sit behind a regulated 5 V rail. The 5 V column is used below.
#    Confirm against the power board before trusting the effort limits.
XL430_W250 = Motor("XL430-W250-T", volts=12.0, stall_torque_nm=1.5, no_load_rpm=61.0, gear_ratio=258.5)
XL330_M288 = Motor("XL330-M288-T", volts=5.0, stall_torque_nm=0.52, no_load_rpm=103.0, gear_ratio=288.4)
XL330_M077 = Motor("XL330-M077-T", volts=5.0, stall_torque_nm=0.20, no_load_rpm=350.0, gear_ratio=93.0)

# --------------------------------------------------------------------------------------
# Joints
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Joint:
    urdf_name: str
    lerobot_name: str
    motor: Motor
    factory_lo_deg: float
    factory_hi_deg: float

    @property
    def factory_limits_rad(self) -> tuple[float, float]:
        return math.radians(self.factory_lo_deg), math.radians(self.factory_hi_deg)


JOINTS: tuple[Joint, ...] = (
    Joint("joint1", "shoulder_pan", XL430_W250, -270.0, 360.0),
    Joint("joint2", "shoulder_lift", XL330_M288, -120.0, 90.0),
    Joint("joint3", "elbow_flex", XL330_M288, -120.0, 90.0),
    Joint("joint4", "wrist_flex", XL330_M288, -100.0, 100.0),
    Joint("joint5", "wrist_roll", XL330_M288, -270.0, 270.0),
    Joint("gripper_joint_1", "gripper", XL330_M077, 0.0, 100.0),
)

# `gripper_joint_2` mirrors `gripper_joint_1` with multiplier -1.
# Source: open_manipulator_description/ros2_control/omx_f.ros2_control.xacro (use_sim branch).
# The URDF importer keeps mimic joints unless `convert_mimic_joints_to_normal_joints=True`.
MIMIC_JOINT = ("gripper_joint_2", "gripper_joint_1", -1.0)

# --------------------------------------------------------------------------------------
# Site limits — the arm cannot use its full factory travel in our cell
# --------------------------------------------------------------------------------------

# 🔴 `docs/experiment_spec.md` §3: the usable azimuth sector is ~135 deg wide,
#    theta in [-90, +45] with 0 = straight ahead, + = left. The boundary is a PHYSICAL
#    collision with the third-person camera rig, not a field-of-view limit.
#    Modelling it as a shoulder_pan limit is the closest single-joint approximation.
#
# ⚠️ [未確認] the sign convention. The URDF gives joint1 axis (0,0,1); whether +joint1 is
#    "left" in the operator's frame is NOT verified. The 5-pose comparison test
#    (S4 §5-1) must confirm it before this narrowing is trusted.
# ⚠️ Rebind to the cameras: if the rig moves, re-measure. This is not a property of the arm.
SITE_AZIMUTH_DEG = (-90.0, 45.0)

# Workspace radii, tape-measured 2026-08-31 (S1), metres.
SITE_R_INNER_M = 0.22
SITE_R_OUTER_M = 0.41
# Cross-check: ROBOTIS quotes OMX-F full reach 400 mm. The two agree within the
# measurement's own resolution — treat that as corroboration, not as a coincidence.
SPEC_FULL_REACH_M = 0.40

# Payload, from the same spec page. An empty aluminium can (~15 g) is fine;
# a FULL can (~330 g) exceeds the arm's rating, especially near full reach.
PAYLOAD_FULL_REACH_KG = 0.100
PAYLOAD_NORMAL_REACH_KG = 0.250


def effective_limits_rad(joint: Joint, apply_site_limits: bool = True) -> tuple[float, float]:
    """Factory travel intersected with what the cell physically allows."""
    lo, hi = joint.factory_limits_rad
    if apply_site_limits and joint.lerobot_name == "shoulder_pan":
        lo = max(lo, math.radians(SITE_AZIMUTH_DEG[0]))
        hi = min(hi, math.radians(SITE_AZIMUTH_DEG[1]))
    return lo, hi


# --------------------------------------------------------------------------------------
# Drive gains — PROVISIONAL, and the biggest single sim2real gap
# --------------------------------------------------------------------------------------

# 🔴 A URDF has no concept of drive stiffness/damping, so SOMETHING has to be invented here
#    and the importer's default (stiffness 100 / damping 1 for every joint) is invented badly:
#    it gives the 0.2 Nm gripper the same gains as the 1.5 Nm base yaw.
#
# The rule used below is at least dimensionally honest and reproducible:
#
#     stiffness = stall_torque / TRACKING_ERROR      "full torque at 5 deg of lag"
#     damping   = DAMPING_FRACTION * stiffness
#
# ⚠️ This is a STARTING POINT, not a calibration. D029 records that closing this gap
#    requires fitting against real recorded trajectories. Until that fit exists, do not
#    claim the simulated arm's dynamics resemble the real one's.
GAIN_TRACKING_ERROR_RAD = math.radians(5.0)
GAIN_DAMPING_FRACTION = 0.05


def stiffness(joint: Joint) -> float:
    return joint.motor.stall_torque_nm / GAIN_TRACKING_ERROR_RAD


def damping(joint: Joint) -> float:
    return GAIN_DAMPING_FRACTION * stiffness(joint)


def joint_table() -> list[dict]:
    """One row per joint — what every downstream config and audit compares against."""
    rows = []
    for j in JOINTS:
        lo_f, hi_f = j.factory_limits_rad
        lo_e, hi_e = effective_limits_rad(j)
        rows.append(
            {
                "urdf": j.urdf_name,
                "lerobot": j.lerobot_name,
                "motor": j.motor.model,
                "factory_deg": (j.factory_lo_deg, j.factory_hi_deg),
                "effective_deg": (math.degrees(lo_e), math.degrees(hi_e)),
                "effective_rad": (lo_e, hi_e),
                "effort_nm": j.motor.stall_torque_nm,
                "velocity_rad_s": j.motor.velocity_limit_rad_s,
                "stiffness": stiffness(j),
                "damping": damping(j),
                "narrowed": (lo_e, hi_e) != (lo_f, hi_f),
            }
        )
    return rows


if __name__ == "__main__":
    hdr = f"{'urdf':<16}{'lerobot':<15}{'motor':<15}{'factory(deg)':<20}{'effective(deg)':<20}{'effort':<9}{'vel':<8}{'stiff':<8}{'damp':<7}"
    print(hdr)
    print("-" * len(hdr))
    for r in joint_table():
        f = f"{r['factory_deg'][0]:.0f} .. {r['factory_deg'][1]:.0f}"
        e = f"{r['effective_deg'][0]:.0f} .. {r['effective_deg'][1]:.0f}"
        mark = "  <- narrowed by site" if r["narrowed"] else ""
        print(
            f"{r['urdf']:<16}{r['lerobot']:<15}{r['motor']:<15}{f:<20}{e:<20}"
            f"{r['effort_nm']:<9.3f}{r['velocity_rad_s']:<8.2f}{r['stiffness']:<8.2f}{r['damping']:<7.2f}{mark}"
        )
    print()
    print(f"mimic: {MIMIC_JOINT[0]} = {MIMIC_JOINT[2]} * {MIMIC_JOINT[1]}")
    print(f"payload: {PAYLOAD_FULL_REACH_KG*1000:.0f} g @ full reach / {PAYLOAD_NORMAL_REACH_KG*1000:.0f} g @ normal reach")
    print("drive gains are PROVISIONAL — see the module docstring and D029.")
