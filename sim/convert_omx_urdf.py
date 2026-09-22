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
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

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
# Patch the limits the URDF could not supply.
# UsdPhysics angular limits and PhysX max joint velocity are in DEGREES; drive maxForce for an
# angular drive is a torque in N*m and needs no conversion.
# ----------------------------------------------------------------------------------------
stage = Usd.Stage.Open(usd_path)
by_name = {r["name"]: r for r in rows}
patched, missing = [], set(by_name)

for prim in stage.Traverse():
    if not prim.IsA(UsdPhysics.RevoluteJoint):
        continue
    name = prim.GetName()
    row = by_name.get(name)
    if row is None:
        continue
    missing.discard(name)
    joint = UsdPhysics.RevoluteJoint(prim)

    old_lo = joint.GetLowerLimitAttr().Get()
    old_hi = joint.GetUpperLimitAttr().Get()
    joint.GetLowerLimitAttr().Set(float(row["lower_deg"]))
    joint.GetUpperLimitAttr().Set(float(row["upper_deg"]))

    drive = UsdPhysics.DriveAPI.Get(prim, "angular")
    old_force = drive.GetMaxForceAttr().Get() if drive else None
    if drive:
        drive.GetMaxForceAttr().Set(float(row["max_force"]))

    physx = PhysxSchema.PhysxJointAPI.Get(stage, prim.GetPath())
    if not physx:
        physx = PhysxSchema.PhysxJointAPI.Apply(prim)
    old_vel = physx.GetMaxJointVelocityAttr().Get()
    physx.CreateMaxJointVelocityAttr().Set(float(row["max_vel_deg_s"]))

    patched.append((name, old_lo, old_hi, row["lower_deg"], row["upper_deg"], old_force, row["max_force"], old_vel, row["max_vel_deg_s"]))

# ----------------------------------------------------------------------------------------
# Re-create the mimic constraint.
# The URDF declares <mimic joint="gripper_joint_1" multiplier="-1"/> on gripper_joint_2, but the
# Isaac Lab 5.1 converter does NOT carry it into the USD even with
# convert_mimic_joints_to_normal_joints=False — verified 2026-09-03 by auditing the output.
# Without it the second finger is a FREE joint and the gripper cannot close.
# ----------------------------------------------------------------------------------------
follower_name, master_name, _urdf_multiplier = K.MIMIC_JOINT
follower = master = None
for prim in stage.Traverse():
    if prim.GetName() == follower_name:
        follower = prim
    elif prim.GetName() == master_name:
        master = prim

mimic_note = ""
if follower is None or master is None:
    mimic_note = f"🔴 could not find {follower_name}/{master_name} — mimic NOT applied"
elif any("Mimic" in sch for sch in follower.GetAppliedSchemas()):
    mimic_note = f"{follower_name} already carries a mimic schema — left alone"
else:
    api = PhysxSchema.PhysxMimicJointAPI.Apply(follower, "rotZ")
    api.CreateReferenceJointRel().SetTargets([master.GetPath()])
    # 🔴 2026-09-18 fix: both gripper joints' UsdPhysics.RevoluteJoint.axis is Z (matches the
    # URDF's <axis xyz="0 0 1"/> on both), confirmed with sim/inspect_mimic_axis.py. This was
    # previously "rotX", which referenced an axis gripper_joint_1 does not rotate about, so the
    # mimic constraint tracked a reference quantity that never changed -- gripper_joint_2 stayed
    # within ~0.1deg of zero regardless of gearing sign (measured with verify_mimic_gearing.py on
    # both +1.0 and -1.0). Not a sign bug; a reference-axis bug.
    api.CreateReferenceJointAxisAttr().Set("rotZ")
    api.CreateGearingAttr().Set(float(args.mimic_gearing))
    api.CreateOffsetAttr().Set(0.0)
    # naturalFrequency / dampingRatio are present on the GUI-imported asset but are not exposed
    # by this schema version's python bindings — create them by name, with the same values.
    from pxr import Sdf  # noqa: PLC0415

    follower.CreateAttribute("physxMimicJoint:rotZ:naturalFrequency", Sdf.ValueTypeNames.Float).Set(25.0)
    follower.CreateAttribute("physxMimicJoint:rotZ:dampingRatio", Sdf.ValueTypeNames.Float).Set(0.005)
    mimic_note = (
        f"applied PhysxMimicJointAPI:rotZ on {follower_name} -> {master_name}, "
        f"gearing={args.mimic_gearing} (sign to be confirmed by the 5-pose test)"
    )

stage.GetRootLayer().Save()

print("\n" + "=" * 118)
print("PATCHED  (original -> new).  Anything not listed was NOT changed.")
print("=" * 118)
print(f"{'joint':<18}{'limit lo':<20}{'limit hi':<20}{'maxForce Nm':<24}{'maxVel deg/s'}")
for n, ol, oh, nl, nh, of_, nf, ov, nv in patched:
    print(
        f"{n:<18}{f'{ol} -> {nl:.1f}':<20}{f'{oh} -> {nh:.1f}':<20}"
        f"{f'{of_} -> {nf:.3f}':<24}{f'{ov} -> {nv:.1f}'}"
    )
print(f"\nmimic: {mimic_note}")
if missing:
    print(f"\n🔴 NOT FOUND in the USD (expected joints that were never patched): {sorted(missing)}")
    print("   Do not use this asset until that is explained.")
print("=" * 118)
print(f"\ndone: {usd_path}")
import os
os._exit(0)
