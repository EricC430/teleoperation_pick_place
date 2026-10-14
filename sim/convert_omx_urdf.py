"""Convert `omx_f.urdf` -> USD reproducibly, and patch what the URDF gets wrong.

Run this INSIDE the isaac-lab container (see `sim/run_in_container.sh`).

Why this script exists
----------------------
`assets/wildbot_with_omxaiarm.usd` was produced by importing the URDF through the Isaac Sim
GUI on 2026-08-28. A GUI import is not reproducible: the settings that produced it are not
recorded anywhere, and the next person to re-import gets a different asset. Worse, several of
the importer's defaults are wrong for this arm (D029 §URDF->USD):

  1. joint limits      URDF placeholders +/-6.283 rad  -> the arm can fold through itself
  2. effort limit      URDF placeholder 1000 Nm        -> infinite strength
  3. velocity limit    URDF placeholder 4.8 rad/s      -> wrong for every joint
  4. drive gains       absent from URDF entirely       -> importer default 100/1 for all joints
  5. collision meshes  default convex hull             -> the gripper fingers fill in solid
  6. self-collision    default off

This script fixes 1-4 by patching the generated USD and 5-6 by passing explicit converter
settings, then prints every "original -> new" pair so the change is auditable
(`docs/specs/S4_sim_teleop_collect.md` §7).

Everything numeric comes from `sim/omx_constants.py`. Nothing is invented here.
"""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--urdf", required=True, help="path to omx_f.urdf")
parser.add_argument("--out", required=True, help="path of the USD to write")
parser.add_argument("--fix-base", action="store_true", default=True, help="weld link0 to the world (default: on)")
parser.add_argument("--free-base", dest="fix_base", action="store_false", help="leave the base free (for the mobile-base scene)")
parser.add_argument("--self-collision", action="store_true", default=False, help="enable self-collision (costs perf)")
parser.add_argument("--no-site-limits", action="store_true", help="use factory travel only, ignore the camera-rig sector")
parser.add_argument("--limits", choices=("site", "factory", "urdf"), default=None,
                    help="joint1-5 limits; default site (factory with --no-site-limits). urdf = +/-360, for replay")
parser.add_argument("--dry-run", action="store_true", help="print the plan and exit without touching Isaac Sim")
parser.add_argument(
    "--mimic-gearing",
    type=float,
    default=1.0,
    help=(
        "PhysxMimicJointAPI gearing for gripper_joint_2. Default 1.0 reproduces what Isaac Sim's own "
        "GUI importer produced from this same URDF on 2026-08-28. The URDF says multiplier=-1, but "
        "PhysX's mimic constraint and URDF's mimic tag do not share a sign convention, so this is NOT "
        "a straight copy. 🔴 The sign is confirmed by the gripper open/close row of the 5-pose test "
        "(S4 §5-1), not by reading either spec."
    ),
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import omx_constants as K  # noqa: E402

APPLY_SITE = not args.no_site_limits


def plan_rows():
    rows = []
    for j in K.JOINTS:
        lo, hi = K.effective_limits_rad(j, apply_site_limits=APPLY_SITE)
        rows.append(
            {
                "name": j.urdf_name,
                "lower_deg": lo * 180.0 / 3.141592653589793,
                "upper_deg": hi * 180.0 / 3.141592653589793,
                "max_force": j.motor.stall_torque_nm,
                "max_vel_deg_s": j.motor.velocity_limit_rad_s * 180.0 / 3.141592653589793,
                "stiffness": K.stiffness(j),
                "damping": K.damping(j),
            }
        )
    return rows


def print_plan(rows):
    print("=" * 100)
    print("CONVERSION PLAN  (site limits applied: %s)" % APPLY_SITE)
    print("=" * 100)
    print(f"{'joint':<18}{'limits (deg)':<22}{'maxForce (Nm)':<16}{'maxVel (deg/s)':<17}{'stiffness':<12}{'damping'}")
    for r in rows:
        lim = f"{r['lower_deg']:.1f} .. {r['upper_deg']:.1f}"
        print(f"{r['name']:<18}{lim:<22}{r['max_force']:<16.3f}{r['max_vel_deg_s']:<17.1f}{r['stiffness']:<12.2f}{r['damping']:.2f}")
    print(f"{K.MIMIC_JOINT[0]:<18}mimic of {K.MIMIC_JOINT[1]} x {K.MIMIC_JOINT[2]}")
    print("-" * 100)
    print("collider_type              = convex_decomposition   (importer default convex_hull fills the gripper)")
    print("self_collision             = %s" % args.self_collision)
    print("merge_fixed_joints         = False                  (keep end_effector_joint as a usable frame)")
    print("convert_mimic_to_normal    = False                  (keep gripper_joint_2 slaved)")
    print("fix_base                   = %s" % args.fix_base)
    print("=" * 100)


rows = plan_rows()
print_plan(rows)

if args.dry_run:
    print("\n--dry-run: nothing was converted.")
    raise SystemExit(0)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg  # noqa: E402
from pxr import Usd  # noqa: E402

RAD2DEG = 180.0 / 3.141592653589793

urdf_path = os.path.abspath(args.urdf)
out_path = os.path.abspath(args.out)
os.makedirs(os.path.dirname(out_path), exist_ok=True)

stiff = {j.urdf_name: K.stiffness(j) for j in K.JOINTS}
damp = {j.urdf_name: K.damping(j) for j in K.JOINTS}
# 🔴 2026-09-18 fix: gripper_joint_2 is a mimic follower and must be UNDRIVEN at the USD level --
# it is meant to be governed only by the PhysxMimicJointAPI constraint applied further down.
# This previously copied gripper_joint_1's (nonzero) gains onto gripper_joint_2 "so the pair is
# consistent" -- but the URDF converter's joint_drive config applies those gains as an actual
# PhysX position drive targeting gripper_joint_2's initial pose (0 rad, from
# omx_scene_cfg.py's ArticulationCfg.init_state). That drive actively held the joint near 0
# and fought the (much softer, naturalFrequency=25/dampingRatio=0.005) mimic constraint -- the
# leading hypothesis (not yet re-verified as of this edit) for why gripper_joint_2 moved <0.15deg
# while gripper_joint_1 swept 90deg with BOTH +1.0 and -1.0 gearing (sim/verify_mimic_gearing.py
# --no-render). Re-run that script after this change before trusting it. Probably not a sign or
# axis bug alone -- see sim/inspect_mimic_axis.py for the referenceJointAxis fix made alongside
# this one, which by itself did NOT change the measured ratio.
stiff[K.MIMIC_JOINT[0]] = 0.0
damp[K.MIMIC_JOINT[0]] = 0.0

cfg = UrdfConverterCfg(
    asset_path=urdf_path,
    usd_dir=os.path.dirname(out_path),
    usd_file_name=os.path.basename(out_path),
    force_usd_conversion=True,
    fix_base=args.fix_base,
    merge_fixed_joints=False,
    convert_mimic_joints_to_normal_joints=False,
    collider_type="convex_decomposition",
    self_collision=args.self_collision,
    joint_drive=UrdfConverterCfg.JointDriveCfg(
        target_type="position",
        drive_type="force",
        gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=stiff, damping=damp),
    ),
)

print("\nconverting ...")
converter = UrdfConverter(cfg)
usd_path = converter.usd_path
print(f"converter wrote: {usd_path}")

# ----------------------------------------------------------------------------------------
# Patch what the URDF could not supply: limits, maxForce, max velocity, drive gains (per DEGREE in USD), the mimic
# constraint the converter drops, and the zero-mass end_effector_link (1 kg in PhysX). One implementation shared with
# sim/patch_omx_usd.py (plain Isaac Sim, on an existing asset): sim/omx_usd_patch.py.
# ----------------------------------------------------------------------------------------
import omx_usd_patch  # noqa: E402

stage = Usd.Stage.Open(usd_path)
patch_lines = omx_usd_patch.patch(stage, limits=args.limits or ("site" if APPLY_SITE else "factory"),
                                  mimic_gearing=args.mimic_gearing)
stage.GetRootLayer().Save()

print("\n" + "=" * 118)
print("PATCHED  (original -> new).  Anything not listed was NOT changed.")
print("=" * 118)
for ln in patch_lines:
    print("  " + ln)
print("=" * 118)
print(f"\ndone: {usd_path}")
import os
os._exit(0)
